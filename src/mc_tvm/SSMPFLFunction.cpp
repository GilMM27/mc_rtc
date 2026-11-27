#include <mc_tvm/SSMPFLFunction.h>

#include <mc_rbdyn/Robot.h>
#include <mc_tvm/Robot.h>

#include <RBDyn/Jacobian.h>
#include <RBDyn/MultiBodyConfig.h>

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc_tvm
{

SSMPFLFunction::SSMPFLFunction(const mc_rbdyn::Robot & robot,
                               std::vector<std::string> links,
                               double dt,
                               std::map<std::string, RegionParameters> regions,
                               double stoppingTime,
                               double minMobility,
                               double minMassEigenvalue,
                               tvm::VariablePtr slack)
: tvm::function::abstract::LinearFunction(static_cast<int>(regions.size() * links.size())), robot_(robot), dt_(dt),
  regions_(std::move(regions)), stoppingTime_(stoppingTime), minMobility_(minMobility),
  minMassEigenvalue_(minMassEigenvalue), slack_(std::move(slack)), inertia_(robot.mb()),
  fullJacobian_(6, robot.mb().nrDof()), fullJacobianDot_(6, robot.mb().nrDof()),
  rawA_(Eigen::MatrixXd::Zero(regions_.size() * links.size(), robot.mb().nrDof())),
  rawUpper_(Eigen::VectorXd::Zero(regions_.size() * links.size())), scaledA_(rawA_),
  slackA_(Eigen::MatrixXd::Zero(rawA_.rows(), rawA_.rows())), scaledUpper_(rawUpper_), nextRawA_(rawA_),
  nextRawUpper_(rawUpper_), nextScaledA_(scaledA_), nextSlackA_(slackA_), nextScaledUpper_(scaledUpper_)
{
  if(links.empty() || regions_.empty() || !std::isfinite(dt) || dt <= 0 || !std::isfinite(stoppingTime)
     || stoppingTime <= 0 || !std::isfinite(minMobility) || minMobility < 0 || !std::isfinite(minMassEigenvalue)
     || minMassEigenvalue <= 0)
    throw std::invalid_argument("invalid SSM+PFL function settings");
  for(const auto & [name, p] : regions_)
  {
    if(name.empty() || !std::isfinite(p.forceMaxN) || p.forceMaxN < 0 || !std::isfinite(p.stiffnessNPerM)
       || p.stiffnessNPerM <= 0 || !std::isfinite(p.distanceMarginM) || p.distanceMarginM < 0)
      throw std::invalid_argument("invalid SSM+PFL region parameters");
  }
  for(const auto & regionEntry : regions_)
  {
    const auto & region = regionEntry.first;
    for(const auto & link : links)
    {
      if(link.empty() || !pairIndices_.emplace(std::make_pair(region, link), pairs_.size()).second)
        throw std::invalid_argument("invalid or duplicate SSM+PFL link");
      pairs_.emplace_back(region, link);
    }
  }
  if(slack_ && slack_->size() != static_cast<int>(pairs_.size()))
    throw std::invalid_argument("SSM+PFL slack size does not match pair count");
  nextRows_.reserve(pairs_.size());

  registerUpdates(Update::Jacobian, &SSMPFLFunction::updateJacobian, Update::B, &SSMPFLFunction::updateb);
  addOutputDependency<SSMPFLFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<SSMPFLFunction>(Output::B, Update::B);
  addVariable(robot.tvmRobot().alphaD(), true);
  if(slack_) { addVariable(slack_, true); }
  velocity_.setZero();
}

void SSMPFLFunction::clear()
{
  rawA_.setZero();
  rawUpper_.setZero();
  scaledA_.setZero();
  slackA_.setZero();
  scaledUpper_.setZero();
  rowData_.clear();
  measurements_.clear();
}

void SSMPFLFunction::setMeasurements(const std::vector<Measurement> & measurements)
{
  if(measurements.size() != pairs_.size()) { throw std::invalid_argument("incomplete SSM+PFL measurement batch"); }
  std::vector<Measurement> ordered(pairs_.size());
  std::vector<bool> seen(pairs_.size(), false);
  for(const auto & measurement : measurements)
  {
    const auto it = pairIndices_.find({measurement.region, measurement.link});
    if(it == pairIndices_.end()) throw std::domain_error("unknown SSM+PFL pair");
    const auto index = it->second;
    if(seen[index]) throw std::domain_error("duplicate SSM+PFL pair");
    seen[index] = true;
    ordered[index] = measurement;
  }
  const auto & mb = robot_.mb();
  const auto & mbc = robot_.mbc();
  const int dofs = mb.nrDof();
  const Eigen::VectorXd qdot = rbd::dofToVector(mb, robot_.alpha());
  if(qdot.size() != dofs || !qdot.allFinite()) { throw std::domain_error("invalid robot velocity"); }
  inertia_.computeH(mb, mbc);
  const Eigen::MatrixXd & mass = inertia_.H();
  if(mass.rows() != dofs || mass.cols() != dofs || !mass.allFinite()) throw std::domain_error("invalid robot inertia");
  const Eigen::LDLT<Eigen::MatrixXd> factor(mass);
  if(factor.info() != Eigen::Success || factor.vectorD().minCoeff() < minMassEigenvalue_)
    throw std::domain_error("robot inertia is not positive definite");

  auto & nextRawA = nextRawA_;
  auto & nextRawUpper = nextRawUpper_;
  auto & nextScaledA = nextScaledA_;
  auto & nextSlackA = nextSlackA_;
  auto & nextScaledUpper = nextScaledUpper_;
  auto & nextRows = nextRows_;
  nextRawA.setZero();
  nextRawUpper.setZero();
  nextScaledA.setZero();
  nextSlackA.setZero();
  nextScaledUpper.setZero();
  nextRows.clear();
  for(size_t i = 0; i < measurements.size(); ++i)
  {
    const auto & m = ordered[i];
    const auto region = regions_.find(m.region);
    if(region == regions_.end() || !std::isfinite(m.distance) || m.distance < 0 || !m.robotPointBodyLocal.allFinite()
       || !m.normal.allFinite() || !m.humanVelocity.allFinite()
       || (m.witnessesValid && (!m.robotPointWorld.allFinite() || !m.humanPointWorld.allFinite())))
      throw std::domain_error("invalid SSM+PFL pair measurement");
    const double normalNorm = m.normal.norm();
    if(!std::isfinite(normalNorm) || normalNorm < 1e-9) throw std::domain_error("invalid SSM+PFL pair normal");
    const Eigen::Vector3d normal = m.normal / normalNorm;
    auto jacobianIt = jacobians_.find(m.robotBody);
    if(jacobianIt == jacobians_.end())
      jacobianIt = jacobians_.try_emplace(m.robotBody, mb, m.robotBody).first;
    auto & jacobian = jacobianIt->second;
    jacobian.point(m.robotPointBodyLocal);
    jacobian.fullJacobian(mb, jacobian.jacobian(mb, mbc), fullJacobian_);
    jacobian.fullJacobian(mb, jacobian.jacobianDot(mb, mbc), fullJacobianDot_);
    if(!fullJacobian_.allFinite() || !fullJacobianDot_.allFinite())
      throw std::domain_error("invalid SSM+PFL point Jacobian");
    RowData row;
    row.directionalJacobian = normal.transpose() * fullJacobian_.bottomRows(3);
    const double mobility = row.directionalJacobian.dot(factor.solve(row.directionalJacobian.transpose()));
    if(!std::isfinite(mobility) || mobility < -minMobility_)
      throw std::domain_error("invalid SSM+PFL directional inverse mass");
    const auto & p = region->second;
    const double pflSpeed = p.forceMaxN / std::sqrt(p.stiffnessNPerM) * std::sqrt(std::max(0.0, mobility));
    const double combinedSpeed = std::max(m.distance - p.distanceMarginM, 0.0) / stoppingTime_ + pflSpeed;
    row.robotSpeedBudget = std::max(0.0, combinedSpeed + normal.dot(m.humanVelocity));
    row.robotApproach = row.directionalJacobian.dot(qdot);
    row.jacobianDrift = (normal.transpose() * fullJacobianDot_.bottomRows(3) * qdot).value();
    row.accelerationBound = row.robotSpeedBudget - row.robotApproach - dt_ * row.jacobianDrift;
    if(!std::isfinite(pflSpeed) || !std::isfinite(combinedSpeed) || !std::isfinite(row.robotSpeedBudget)
       || !std::isfinite(row.robotApproach) || !std::isfinite(row.jacobianDrift)
       || !std::isfinite(row.accelerationBound))
      throw std::domain_error("nonfinite SSM+PFL row");
    nextRawA.row(static_cast<int>(i)) = dt_ * row.directionalJacobian;
    nextRawUpper[static_cast<int>(i)] = row.accelerationBound;
    const double scale = 1.0 / std::max(1e-3, nextRawA.row(static_cast<int>(i)).norm());
    nextScaledA.row(static_cast<int>(i)) = scale * nextRawA.row(static_cast<int>(i));
    nextScaledUpper[static_cast<int>(i)] = scale * row.accelerationBound;
    if(slack_) { nextSlackA(static_cast<int>(i), static_cast<int>(i)) = -scale; }
    nextRows.push_back(std::move(row));
  }
  rawA_.swap(nextRawA);
  rawUpper_.swap(nextRawUpper);
  scaledA_.swap(nextScaledA);
  slackA_.swap(nextSlackA);
  scaledUpper_.swap(nextScaledUpper);
  rowData_.swap(nextRows);
  measurements_.swap(ordered);
}

void SSMPFLFunction::updateJacobian()
{
  jacobian_[robot_.tvmRobot().alphaD().get()] = scaledA_;
  if(slack_) { jacobian_[slack_.get()] = slackA_; }
}

void SSMPFLFunction::updateb()
{
  b_ = -scaledUpper_;
}

} // namespace mc_tvm
