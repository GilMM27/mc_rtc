#pragma once

#include <mc_solver/ConstraintSet.h>
#include <mc_tvm/SSMPFLFunction.h>

#include <mc_rtc/gui/StateBuilder.h>
#include <mc_rtc/log/Logger.h>
#include <tvm/ControlProblem.h>

namespace mc_solver
{
namespace details { class DistanceFunction; }

/** TVM SSM + PFL speed constraint with a fixed number of region/link pairs.
 *
 * Each measured distance is represented by a sampled position-level function;
 * an order-two task dynamic limits next-step robot approach speed. The local
 * model holds the measured witness, normal and human velocity fixed during a
 * controller step. A moving human can reduce the robot approach budget to zero.
 * This directional model does not bound tangential point velocity or contact
 * force. A positive weight enables one-sided soft speed slack in m/s.
 */
struct MC_SOLVER_DLLAPI SSMPFLConstraint : public ConstraintSet
{
  using RegionParameters = mc_tvm::SSMPFLFunction::RegionParameters;
  using Measurement = mc_tvm::SSMPFLFunction::Measurement;
  using RowData = mc_tvm::SSMPFLFunction::RowData;

  SSMPFLConstraint(const mc_rbdyn::Robot & robot,
                   std::vector<std::string> links,
                   double dt,
                   std::map<std::string, RegionParameters> regions,
                   double stoppingTime,
                   double minMobility = 1e-12,
                   double minMassEigenvalue = 1e-9,
                   double weight = 0);

  /** Validate and replace a complete batch atomically.
   * An exception leaves the previous batch active. The caller must clear the
   * constraint and skip the QP when a measurement batch is invalid or stale.
   */
  void setMeasurements(const std::vector<Measurement> & measurements);
  /** Deactivate all fixed pair rows as 0 <= 0. */
  void clear();
  double softSlack(size_t row) const;

  const std::vector<std::pair<std::string, std::string>> & pairs() const noexcept;
  const std::map<std::string, RegionParameters> & regions() const noexcept;
  const std::vector<RowData> & rowData() const noexcept;
  const Eigen::MatrixXd & rows() const noexcept;
  const Eigen::VectorXd & bounds() const noexcept;

  void addToSolverImpl(QPSolver & solver) override;
  void removeFromSolverImpl(QPSolver & solver) override;

private:
  void addPairDiagnostics(QPSolver & solver);
  void removePairDiagnostics();
  void toggleMonitor(size_t row);
  double pairValue(size_t row, bool budget) const;
  double weight_;
  double dt_;
  tvm::VariablePtr slackRoot_;
  tvm::VariablePtr slack_;
  std::shared_ptr<details::DistanceFunction> function_;
  tvm::TaskWithRequirementsPtr task_;
  tvm::TaskWithRequirementsPtr slackBound_;
  tvm::TaskWithRequirementsPtr slackObjective_;
  std::shared_ptr<mc_rtc::gui::StateBuilder> gui_;
  std::shared_ptr<mc_rtc::Logger> logger_;
  std::vector<std::string> pairLogKeys_;
  std::vector<bool> monitored_;
  std::vector<Eigen::Vector3d> monitorRobotPoints_;
  std::vector<Eigen::Vector3d> monitorHumanPoints_;
};

} // namespace mc_solver
