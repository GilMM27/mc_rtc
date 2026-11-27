#pragma once
#include <mc_rbdyn/Robot.h>
#include <mc_tvm/Robot.h>
#include <mc_tvm/SSMPFLFunction.h>

#include <tvm/Variable.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <RBDyn/FD.h>
#include <RBDyn/Jacobian.h>
#include <RBDyn/MultiBodyConfig.h>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mc_solver::details
{
using Constraint = mc_tvm::SSMPFLFunction;
using Measurement = Constraint::Measurement;
using RegionParameters = Constraint::RegionParameters;

// A sampled local model: the nearest feature and human witness velocity are
// held fixed over one QP step. q's derivative is the robot point Jacobian.
class DistanceFunction final : public tvm::function::abstract::Function
{
public:
  SET_OUTPUTS(DistanceFunction, Budget)
  SET_UPDATES(DistanceFunction, Value, Velocity, Jacobian, NormalAcceleration, Budget)

  DistanceFunction(const mc_rbdyn::Robot & robot,
                   std::vector<std::string> links,
                   double dt,
                   std::map<std::string, RegionParameters> regions,
                   double stoppingTime,
                   double minMobility,
                   double minMassEigenvalue,
                   tvm::VariablePtr slackRoot = nullptr)
  : Function(static_cast<int>(links.size() * regions.size())), robot_(robot), q_(robot.tvmRobot().q()),
    slackRoot_(std::move(slackRoot)), dt_(dt), stoppingTime_(stoppingTime), minMobility_(minMobility),
    minMassEigenvalue_(minMassEigenvalue), regions_(std::move(regions)), inertia_(robot.mb()),
    fullJ_(6, robot.mb().nrDof()), fullJdot_(6, robot.mb().nrDof()),
    nextJ_(Eigen::MatrixXd::Zero(size(), robot.mb().nrDof())), J_(nextJ_), nextDrift_(Eigen::VectorXd::Zero(size())),
    drift_(nextDrift_), nextBudget_(Eigen::VectorXd::Zero(size())), budget_(nextBudget_),
    nextHuman_(Eigen::VectorXd::Zero(size())), human_(nextHuman_), nextDistance_(Eigen::VectorXd::Zero(size())),
    distance_(nextDistance_)
  {
    if(links.empty() || regions_.empty() || !std::isfinite(dt_) || dt_ <= 0 || !std::isfinite(1.0 / dt_)
       || !std::isfinite(stoppingTime_) || stoppingTime_ <= 0 || !std::isfinite(minMobility_) || minMobility_ < 0
       || !std::isfinite(minMassEigenvalue_) || minMassEigenvalue_ <= 0)
      throw std::invalid_argument("invalid SSM+PFL function settings");
    for(const auto & [region, p] : regions_)
    {
      if(region.empty() || !std::isfinite(p.forceMaxN) || p.forceMaxN < 0 || !std::isfinite(p.stiffnessNPerM)
         || p.stiffnessNPerM <= 0 || !std::isfinite(p.distanceMarginM) || p.distanceMarginM < 0)
        throw std::invalid_argument("invalid region parameters");
      for(const auto & link : links)
      {
        if(link.empty() || !indices_.emplace(std::make_pair(region, link), pairs_.size()).second)
          throw std::invalid_argument("duplicate or empty pair");
        pairs_.emplace_back(region, link);
      }
    }
    rawA_.setZero(size(), robot_.mb().nrDof());
    rawUpper_.setZero(size());
    if(slackRoot_ && slackRoot_->size() != size()) throw std::invalid_argument("slack size mismatch");
    registerUpdates(Update::Value, &DistanceFunction::updateValue, Update::Velocity, &DistanceFunction::updateVelocity,
                    Update::Jacobian, &DistanceFunction::updateJacobian, Update::NormalAcceleration,
                    &DistanceFunction::updateNormalAcceleration, Update::Budget, &DistanceFunction::updateBudget);
    addOutputDependency<DistanceFunction>(Output::Value, Update::Value);
    addOutputDependency<DistanceFunction>(Output::Velocity, Update::Velocity);
    addOutputDependency<DistanceFunction>(Output::Jacobian, Update::Jacobian);
    addOutputDependency<DistanceFunction>(Output::NormalAcceleration, Update::NormalAcceleration);
    addOutputDependency<DistanceFunction>(Output::Budget, Update::Budget);
    addVariable(q_, false);
    if(slackRoot_) addVariable(slackRoot_, false);
  }

  void setMeasurements(const std::vector<Measurement> & batch)
  {
    if(batch.size() != pairs_.size()) throw std::invalid_argument("incomplete measurement batch");
    std::vector<Measurement> ordered(batch.size());
    std::vector<bool> seen(batch.size(), false);
    for(const auto & m : batch)
    {
      const auto it = indices_.find({m.region, m.link});
      if(it == indices_.end() || seen[it->second]) throw std::domain_error("unknown or duplicate pair");
      seen[it->second] = true;
      ordered[it->second] = m;
    }
    const auto & mb = robot_.mb();
    const auto & mbc = robot_.mbc();
    const Eigen::VectorXd qdot = rbd::dofToVector(mb, robot_.alpha());
    if(qdot.size() != mb.nrDof() || !qdot.allFinite()) throw std::domain_error("invalid velocity");
    inertia_.computeH(mb, mbc);
    const auto & mass = inertia_.H();
    if(mass.rows() != mb.nrDof() || mass.cols() != mb.nrDof() || !mass.allFinite())
      throw std::domain_error("invalid mass matrix");
    const Eigen::LDLT<Eigen::MatrixXd> factor(mass);
    if(factor.info() != Eigen::Success || factor.vectorD().minCoeff() < minMassEigenvalue_)
      throw std::domain_error("mass matrix is not positive definite");
    for(size_t i = 0; i < ordered.size(); ++i)
    {
      const auto & m = ordered[i];
      if(m.robotBody.empty() || !std::isfinite(m.distance) || m.distance < 0 || !m.robotPointBodyLocal.allFinite()
         || !m.normal.allFinite() || !m.humanVelocity.allFinite()
         || (m.witnessesValid && (!m.robotPointWorld.allFinite() || !m.humanPointWorld.allFinite())))
        throw std::domain_error("invalid measurement");
      const double norm = m.normal.norm();
      if(!std::isfinite(norm) || norm < 1e-9) throw std::domain_error("invalid normal");
      const Eigen::Vector3d n = m.normal / norm;
      auto it = jacobians_.find(m.robotBody);
      if(it == jacobians_.end()) it = jacobians_.try_emplace(m.robotBody, mb, m.robotBody).first;
      auto & jac = it->second;
      jac.point(m.robotPointBodyLocal);
      jac.fullJacobian(mb, jac.jacobian(mb, mbc), fullJ_);
      jac.fullJacobian(mb, jac.jacobianDot(mb, mbc), fullJdot_);
      if(!fullJ_.allFinite() || !fullJdot_.allFinite()) throw std::domain_error("invalid Jacobian");
      nextJ_.row(static_cast<int>(i)) = n.transpose() * fullJ_.bottomRows(3);
      nextDrift_[static_cast<int>(i)] = (n.transpose() * fullJdot_.bottomRows(3) * qdot).value();
      const double mobility =
          nextJ_.row(static_cast<int>(i)).dot(factor.solve(nextJ_.row(static_cast<int>(i)).transpose()));
      if(!std::isfinite(mobility) || mobility < -minMobility_)
        throw std::domain_error("invalid directional inverse mass");
      const auto & p = regions_.at(m.region);
      const double pfl = p.forceMaxN / std::sqrt(p.stiffnessNPerM) * std::sqrt(std::max(0.0, mobility));
      const double ssm = std::max(m.distance - p.distanceMarginM, 0.0) / stoppingTime_;
      nextHuman_[static_cast<int>(i)] = n.dot(m.humanVelocity);
      if(!std::isfinite(pfl) || !std::isfinite(ssm) || !std::isfinite(nextHuman_[static_cast<int>(i)]))
        throw std::domain_error("nonfinite SSM+PFL budget inputs");
      const double combined = ssm + pfl + nextHuman_[static_cast<int>(i)];
      if(!std::isfinite(combined)) throw std::domain_error("nonfinite SSM+PFL budget");
      nextBudget_[static_cast<int>(i)] = std::max(0.0, combined);
      nextDistance_[static_cast<int>(i)] = m.distance;
      if(!std::isfinite(nextDrift_[static_cast<int>(i)]) || !std::isfinite(nextBudget_[static_cast<int>(i)]))
        throw std::domain_error("nonfinite local model");
    }
    std::vector<Constraint::RowData> nextRows;
    nextRows.reserve(ordered.size());
    Eigen::MatrixXd nextRawA = dt_ * nextJ_;
    if(!nextRawA.allFinite()) throw std::domain_error("nonfinite SSM+PFL Jacobian row");
    Eigen::VectorXd nextRawUpper(size());
    for(int i = 0; i < size(); ++i)
    {
      Constraint::RowData row;
      row.directionalJacobian = nextJ_.row(i);
      row.robotApproach = row.directionalJacobian.dot(qdot);
      row.jacobianDrift = nextDrift_[i];
      row.robotSpeedBudget = nextBudget_[i];
      row.accelerationBound = row.robotSpeedBudget - row.robotApproach - dt_ * row.jacobianDrift;
      const double separationRate = nextHuman_[i] - row.robotApproach;
      const double accelerationTarget = (nextHuman_[i] - row.robotSpeedBudget - separationRate) / dt_;
      if(!std::isfinite(row.robotApproach) || !std::isfinite(row.accelerationBound) || !std::isfinite(separationRate)
         || !std::isfinite(accelerationTarget))
        throw std::domain_error("nonfinite SSM+PFL row");
      nextRawUpper[i] = row.accelerationBound;
      nextRows.push_back(std::move(row));
    }
    J_.swap(nextJ_);
    drift_.swap(nextDrift_);
    budget_.swap(nextBudget_);
    human_.swap(nextHuman_);
    distance_.swap(nextDistance_);
    measurements_.swap(ordered);
    rowData_.swap(nextRows);
    rawA_.swap(nextRawA);
    rawUpper_.swap(nextRawUpper);
    if(slackRoot_)
    {
      slackRoot_->set(Eigen::VectorXd::Zero(size()));
      tvm::dot(slackRoot_)->set(Eigen::VectorXd::Zero(size()));
    }
    active_ = true;
  }

  void clear()
  {
    J_.setZero();
    drift_.setZero();
    budget_.setZero();
    human_.setZero();
    distance_.setZero();
    publishedBudget_.setZero();
    if(slackRoot_)
    {
      slackRoot_->set(Eigen::VectorXd::Zero(size()));
      tvm::dot(slackRoot_)->set(Eigen::VectorXd::Zero(size()));
    }
    measurements_.clear();
    rowData_.clear();
    rawA_.setZero(size(), robot_.mb().nrDof());
    rawUpper_.setZero(size());
    active_ = false;
  }

  void updateValue()
  {
    value_ = distance_;
    if(slackRoot_) value_ += slackRoot_->value();
  }
  void updateVelocity()
  {
    velocity_ = human_ - J_ * rbd::dofToVector(robot_.mb(), robot_.alpha());
    if(slackRoot_) velocity_ += tvm::dot(slackRoot_)->value();
  }
  void updateJacobian()
  {
    jacobian_[q_.get()] = -J_;
    if(slackRoot_)
    {
      jacobian_[slackRoot_.get()].setZero();
      if(active_) jacobian_[slackRoot_.get()].diagonal().setOnes();
    }
  }
  void updateNormalAcceleration() { normalAcceleration_ = -drift_; }
  void updateBudget() { publishedBudget_ = budget_; }
  const Eigen::VectorXd & budget() const { return publishedBudget_; }
  const Eigen::MatrixXd & directionalJacobian() const { return J_; }
  const Eigen::VectorXd & drift() const { return drift_; }
  const Eigen::VectorXd & humanSpeed() const { return human_; }
  const std::vector<Measurement> & measurements() const { return measurements_; }
  const auto & pairs() const { return pairs_; }
  const auto & regions() const { return regions_; }
  const auto & rowData() const { return rowData_; }
  const Eigen::MatrixXd & rows() const { return rawA_; }
  const Eigen::VectorXd & bounds() const { return rawUpper_; }
  bool active() const { return active_; }
  double dt() const { return dt_; }

private:
  const mc_rbdyn::Robot & robot_;
  tvm::VariablePtr q_, slackRoot_;
  double dt_, stoppingTime_, minMobility_, minMassEigenvalue_;
  std::map<std::string, RegionParameters> regions_;
  std::vector<std::pair<std::string, std::string>> pairs_;
  std::map<std::pair<std::string, std::string>, size_t> indices_;
  std::vector<Measurement> measurements_;
  rbd::ForwardDynamics inertia_;
  std::map<std::string, rbd::Jacobian> jacobians_;
  Eigen::MatrixXd fullJ_, fullJdot_, nextJ_, J_;
  Eigen::VectorXd nextDrift_, drift_, nextBudget_, budget_, nextHuman_, human_;
  Eigen::VectorXd nextDistance_, distance_, publishedBudget_;
  std::vector<Constraint::RowData> rowData_;
  Eigen::MatrixXd rawA_;
  Eigen::VectorXd rawUpper_;
  bool active_ = false;
};

class BudgetDynamics final : public tvm::task_dynamics::abstract::TaskDynamics
{
public:
  explicit BudgetDynamics(std::shared_ptr<DistanceFunction> f) : f_(std::move(f)) {}
  class Impl final : public tvm::task_dynamics::abstract::TaskDynamicsImpl
  {
  public:
    Impl(tvm::FunctionPtr f,
         tvm::constraint::Type type,
         const Eigen::VectorXd & rhs,
         std::shared_ptr<DistanceFunction> distance)
    : TaskDynamicsImpl(tvm::task_dynamics::Order::Two, f, type, rhs), distance_(std::move(distance))
    {
      addInputDependency(Update::UpdateValue, *distance_, DistanceFunction::Output::Budget);
    }
    void updateValue() override
    {
      // d_dot + dt*d_ddot >= h - b. With slack root x, dt*x_ddot is slack [m/s].
      value_ = (distance_->humanSpeed() - distance_->budget() - function().velocity()) / distance_->dt();
    }

  private:
    std::shared_ptr<DistanceFunction> distance_;
  };

protected:
  std::unique_ptr<tvm::task_dynamics::abstract::TaskDynamicsImpl> impl_(tvm::FunctionPtr f,
                                                                        tvm::constraint::Type type,
                                                                        const Eigen::VectorXd & rhs) const override
  {
    return std::make_unique<Impl>(f, type, rhs, f_);
  }
  tvm::task_dynamics::Order order_() const override { return tvm::task_dynamics::Order::Two; }

private:
  std::shared_ptr<DistanceFunction> f_;
};

} // namespace mc_solver::details
