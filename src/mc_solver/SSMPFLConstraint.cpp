#include <mc_solver/SSMPFLConstraint.h>
#include "SSMPFLPositionFunction.h"

#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/TVMQPSolver.h>

#include <mc_rbdyn/Robot.h>
#include <mc_rtc/gui/Arrow.h>
#include <mc_rtc/gui/Checkbox.h>
#include <mc_rtc/gui/Label.h>
#include <mc_rtc/logging.h>

#include <tvm/Space.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mc_solver
{

SSMPFLConstraint::SSMPFLConstraint(const mc_rbdyn::Robot & robot,
                                   std::vector<std::string> links,
                                   double dt,
                                   std::map<std::string, RegionParameters> regions,
                                   double stoppingTime,
                                   double minMobility,
                                   double minMassEigenvalue,
                                   double weight)
: weight_(weight), dt_(dt),
  slackRoot_(weight > 0 ? tvm::Space(static_cast<int>(regions.size() * links.size())).createVariable("SSMPFL slack root") : nullptr),
  slack_(slackRoot_ ? tvm::dot(slackRoot_, 2) : nullptr),
  function_(std::make_shared<details::DistanceFunction>(robot,
                                                     std::move(links),
                                                     dt,
                                                     std::move(regions),
                                                     stoppingTime,
                                                     minMobility,
                                                     minMassEigenvalue,
                                                     slackRoot_)),
  monitored_(function_->pairs().size(), false), monitorRobotPoints_(function_->pairs().size(), Eigen::Vector3d::Zero()),
  monitorHumanPoints_(function_->pairs().size(), Eigen::Vector3d::Zero())
{
  if(slackRoot_)
  {
    slackRoot_->set(Eigen::VectorXd::Zero(pairs().size()));
    tvm::dot(slackRoot_)->set(Eigen::VectorXd::Zero(pairs().size()));
    slack_->set(Eigen::VectorXd::Zero(pairs().size()));
  }
  if(backend_ != QPSolver::Backend::TVM) { throw std::invalid_argument("SSMPFLConstraint requires TVM backend"); }
  if(!std::isfinite(weight_) || weight_ < 0) { throw std::invalid_argument("invalid SSM+PFL weight"); }
  if(weight_ > 0 && (!std::isfinite(weight_ * dt_ * dt_) || weight_ * dt_ * dt_ == 0))
    throw std::invalid_argument("SSM+PFL weight and timestep produce an invalid slack penalty");
}

const std::vector<std::pair<std::string, std::string>> & SSMPFLConstraint::pairs() const noexcept { return function_->pairs(); }
const std::map<std::string, SSMPFLConstraint::RegionParameters> & SSMPFLConstraint::regions() const noexcept { return function_->regions(); }
const std::vector<SSMPFLConstraint::RowData> & SSMPFLConstraint::rowData() const noexcept { return function_->rowData(); }
const Eigen::MatrixXd & SSMPFLConstraint::rows() const noexcept { return function_->rows(); }
const Eigen::VectorXd & SSMPFLConstraint::bounds() const noexcept { return function_->bounds(); }

void SSMPFLConstraint::setMeasurements(const std::vector<Measurement> & measurements)
{
  function_->setMeasurements(measurements);
  const auto & ordered = function_->measurements();
  for(size_t i = 0; i < ordered.size(); ++i)
  {
    monitorRobotPoints_[i] = ordered[i].robotPointWorld;
    monitorHumanPoints_[i] = ordered[i].humanPointWorld;
  }
}

void SSMPFLConstraint::clear()
{
  function_->clear();
  std::fill(monitorRobotPoints_.begin(), monitorRobotPoints_.end(), Eigen::Vector3d::Zero());
  std::fill(monitorHumanPoints_.begin(), monitorHumanPoints_.end(), Eigen::Vector3d::Zero());
}

double SSMPFLConstraint::softSlack(size_t row) const
{
  if(row >= pairs().size()) { throw std::out_of_range("SSM+PFL row index"); }
  if(row >= function_->rowData().size()) { return 0.0; }
  return slack_ ? std::max(0.0, dt_ * slack_->value()[static_cast<int>(row)]) : 0.0;
}

void SSMPFLConstraint::addToSolverImpl(QPSolver & solver)
{
  auto & problem = TVMQPSolver::from_solver(solver).problem();
  gui_ = solver.gui();
  logger_ = solver.logger();
  task_ = problem.add(function_ >= 0., details::BudgetDynamics(function_), {tvm::requirements::PriorityLevel(0)});
  if(slack_)
  {
    slackBound_ = problem.add(slack_ >= 0., {tvm::requirements::PriorityLevel(0)});
    slackObjective_ =
        problem.add(slack_ == 0., {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(weight_ * dt_ * dt_)});
  }
  addPairDiagnostics(solver);
}

void SSMPFLConstraint::removeFromSolverImpl(QPSolver & solver)
{
  removePairDiagnostics();
  auto & problem = TVMQPSolver::from_solver(solver).problem();
  if(slackObjective_)
  {
    problem.remove(*slackObjective_);
    slackObjective_.reset();
  }
  if(slackBound_)
  {
    problem.remove(*slackBound_);
    slackBound_.reset();
  }
  if(task_)
  {
    problem.remove(*task_);
    task_.reset();
  }
}

double SSMPFLConstraint::pairValue(size_t row, bool budget) const
{
  if(row >= function_->rowData().size()) return std::numeric_limits<double>::quiet_NaN();
  return budget ? function_->rowData()[row].robotSpeedBudget : function_->measurements()[row].distance;
}

void SSMPFLConstraint::addPairDiagnostics(QPSolver &)
{
  for(size_t i = 0; i < pairs().size(); ++i)
  {
    const auto & [region, link] = pairs()[i];
    const auto prefix = "SSMPFLPair_" + region + "_" + link + "_";
    if(logger_)
    {
      const auto budgetKey = prefix + "robot_speed_budget_mps";
      const auto slackKey = prefix + "soft_slack_mps";
      const auto distanceKey = prefix + "distance_m";
      logger_->addLogEntry(budgetKey, [this, i]() { return pairValue(i, true); });
      logger_->addLogEntry(slackKey, [this, i]() { return i < function_->rowData().size() ? softSlack(i) : 0.0; });
      logger_->addLogEntry(distanceKey, [this, i]() { return pairValue(i, false); });
      pairLogKeys_.insert(pairLogKeys_.end(), {budgetKey, slackKey, distanceKey});
    }
    if(gui_)
    {
      gui_->addElement(
          {"SSM+PFL", "Pairs", region},
          mc_rtc::gui::Checkbox(link, [this, i]() { return monitored_[i]; }, [this, i]() { toggleMonitor(i); }));
    }
  }
}

void SSMPFLConstraint::toggleMonitor(size_t row)
{
  if(!gui_) return;
  const auto & [region, link] = pairs()[row];
  const std::vector<std::string> category = {"SSM+PFL", "Pair monitors", region};
  if(monitored_[row])
  {
    gui_->removeElement(category, link);
    gui_->removeElement({"SSM+PFL", "Pair arrows", region}, link);
  }
  else
  {
    gui_->addElement(category, mc_rtc::gui::Label(link,
                                                  [this, row]()
                                                  {
                                                    const auto distance = pairValue(row, false);
                                                    return std::isfinite(distance) ? std::to_string(distance) + " m"
                                                                                   : std::string("unavailable");
                                                  }));
    gui_->addElement({"SSM+PFL", "Pair arrows", region},
                     mc_rtc::gui::Arrow(
                         link, [this, row]() -> const Eigen::Vector3d & { return monitorRobotPoints_[row]; },
                         [this, row]() -> const Eigen::Vector3d & { return monitorHumanPoints_[row]; }));
  }
  monitored_[row] = !monitored_[row];
}

void SSMPFLConstraint::removePairDiagnostics()
{
  if(gui_) gui_->removeCategory({"SSM+PFL", "Pairs"});
  if(gui_) gui_->removeCategory({"SSM+PFL", "Pair monitors"});
  if(gui_) gui_->removeCategory({"SSM+PFL", "Pair arrows"});
  if(logger_)
    for(const auto & key : pairLogKeys_) logger_->removeLogEntry(key);
  pairLogKeys_.clear();
  std::fill(monitored_.begin(), monitored_.end(), false);
  gui_.reset();
  logger_.reset();
}

} // namespace mc_solver

namespace
{

static auto registered = mc_solver::ConstraintSetLoader::register_load_function(
    "ssmPfl",
    [](mc_solver::QPSolver & solver, const mc_rtc::Configuration & config)
    {
      if(solver.backend() != mc_solver::QPSolver::Backend::TVM)
        mc_rtc::log::error_and_throw("SSM+PFL requires the TVM solver backend");
      const unsigned int robotIndex =
          robotIndexFromConfig(config, solver.robots(), "ssmPfl", false, "robotIndex", "robot", "");
      if(config.has("capacity"))
        mc_rtc::log::error_and_throw("SSM+PFL capacity is no longer supported; configure Links");
      const std::vector<std::string> links = config("Links");
      const double stoppingTime = config("StoppingTimeSeconds");
      const double minMobility = config("MinDirectionalInverseMass", 1e-12);
      const double minMassEigenvalue = config("MinMassEigenvalue", 1e-9);
      const double weight = config("Weight", 0.0);
      if(config.has("Weight") && (!std::isfinite(weight) || weight <= 0))
        mc_rtc::log::error_and_throw("SSM+PFL Weight must be positive and finite when supplied");
      const auto regionConfig = config("Regions");
      std::map<std::string, mc_solver::SSMPFLConstraint::RegionParameters> regions;
      for(const auto & name : regionConfig.keys())
      {
        const auto entry = regionConfig(name);
        mc_solver::SSMPFLConstraint::RegionParameters p;
        p.forceMaxN = entry("ForceMaxN");
        p.stiffnessNPerM = entry("StiffnessNPerM");
        p.distanceMarginM = entry("DistanceMarginM", 0.0);
        regions.emplace(name, p);
      }
      return std::make_shared<mc_solver::SSMPFLConstraint>(solver.robot(robotIndex), links, solver.dt(),
                                                           std::move(regions), stoppingTime, minMobility,
                                                           minMassEigenvalue, weight);
    });

} // namespace
