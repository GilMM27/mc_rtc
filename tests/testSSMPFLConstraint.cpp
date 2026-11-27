#include <mc_rbdyn/RobotLoader.h>

#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/SSMPFLConstraint.h>
#include <mc_solver/TVMQPSolver.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_tvm/Robot.h>

#include <RBDyn/FD.h>
#include <RBDyn/Jacobian.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <limits>

#include "utils.h"

BOOST_AUTO_TEST_CASE(TestSSMPFLConstraint)
{
  configureRobotLoader();
  auto rm = mc_rbdyn::RobotLoader::get_robot_module("JVRC1");
  auto em = mc_rbdyn::RobotLoader::get_robot_module("env/ground");
  auto robots = mc_rbdyn::loadRobotAndEnv(*rm, *em);
  mc_solver::TVMQPSolver solver(robots, 0.005);
  auto gui = std::make_shared<mc_rtc::gui::StateBuilder>();
  solver.gui(gui);
  auto config = mc_rtc::Configuration::fromYAMLData(R"(
type: ssmPfl
Links: [wrist, elbow]
StoppingTimeSeconds: 0.5
Regions:
  hand: {ForceMaxN: 140, StiffnessNPerM: 75000, DistanceMarginM: 0.05}
)");
  auto obsoleteConfig = config;
  obsoleteConfig.add("capacity", 2);
  BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load<mc_solver::SSMPFLConstraint>(solver, obsoleteConfig),
                    std::runtime_error);
  config.remove("capacity"); // Configuration copies share their underlying data.
  auto constraint = mc_solver::ConstraintSetLoader::load<mc_solver::SSMPFLConstraint>(solver, config);
  BOOST_REQUIRE(constraint);
  BOOST_CHECK_EQUAL(constraint->pairs().size(), 2);
  const auto originalTasks = solver.problem().tasks().size();
  solver.addConstraintSet(*constraint);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), originalTasks + 1);
  BOOST_CHECK(gui->hasElement({"SSM+PFL", "Pairs", "hand"}, "wrist"));

  mc_solver::SSMPFLConstraint::Measurement measurement;
  measurement.region = "hand";
  measurement.link = "wrist";
  measurement.robotBody = "R_WRIST_Y_S";
  measurement.normal = Eigen::Vector3d::UnitX();
  measurement.humanVelocity = Eigen::Vector3d(0.02, 0, 0);
  measurement.distance = 0.2;
  auto other = measurement;
  other.link = "elbow";
  other.distance = 0.3;
  const auto taskCount = solver.problem().tasks().size();
  constraint->setMeasurements({other, measurement});
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), taskCount);
  BOOST_CHECK_EQUAL(constraint->pairs()[0].second, "wrist");
  BOOST_CHECK_EQUAL(constraint->pairs()[1].second, "elbow");
  BOOST_REQUIRE_EQUAL(constraint->rowData().size(), 2);
  BOOST_CHECK_GT(constraint->bounds()[1], constraint->bounds()[0]);
  const Eigen::VectorXd firstBounds = constraint->bounds();
  auto replacement = measurement;
  replacement.distance = 0.4;
  constraint->setMeasurements({replacement, other});
  BOOST_CHECK_GT(constraint->bounds()[0], firstBounds[0]);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), taskCount);
  constraint->setMeasurements({measurement, other});
  BOOST_CHECK_SMALL((constraint->bounds() - firstBounds).norm(), 1e-12);
  const Eigen::MatrixXd firstRows = constraint->rows();
  auto shifted = measurement;
  shifted.robotPointBodyLocal = Eigen::Vector3d(0.2, 0.3, 0.4);
  constraint->setMeasurements({shifted, other});
  BOOST_CHECK_GT((constraint->rows().row(0) - firstRows.row(0)).norm(), 1e-8);
  constraint->setMeasurements({measurement, other});
  BOOST_CHECK_SMALL((constraint->rows() - firstRows).norm(), 1e-12);
  const auto row = constraint->rowData()[0];
  BOOST_CHECK_SMALL((constraint->rows().row(0) - 0.005 * row.directionalJacobian).norm(), 1e-12);
  BOOST_CHECK_SMALL(constraint->bounds()[0] - row.accelerationBound, 1e-12);

  rbd::ForwardDynamics inertia(solver.robot().mb());
  inertia.computeH(solver.robot().mb(), solver.robot().mbc());
  const auto factor = inertia.H().ldlt();
  const double inverseMass = row.directionalJacobian.dot(factor.solve(row.directionalJacobian.transpose()));
  const double expectedBudget = 0.15 / 0.5 + 140 / std::sqrt(75000.0) * std::sqrt(std::max(0.0, inverseMass)) + 0.02;
  BOOST_CHECK_SMALL(row.robotSpeedBudget - expectedBudget, 1e-10);
  BOOST_CHECK_SMALL(row.accelerationBound - (row.robotSpeedBudget - row.robotApproach - 0.005 * row.jacobianDrift),
                    1e-12);

  // A lower-priority objective asks to cross the limit. The solved acceleration must respect the hard row.
  const Eigen::VectorXd rowDirection = constraint->rows().row(0).transpose();
  BOOST_REQUIRE_GT(rowDirection.squaredNorm(), 1e-12);
  const Eigen::VectorXd desiredAcceleration =
      rowDirection * ((constraint->bounds()[0] + 0.1) / rowDirection.squaredNorm());
  const auto acceleration = solver.robot().tvmRobot().alphaD();
  auto objective = solver.problem().add(acceleration == desiredAcceleration,
                                        {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.0)});
  BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
  BOOST_CHECK_LE(constraint->rows().row(0).dot(acceleration->value()) - constraint->bounds()[0], 1e-7);
  solver.problem().remove(*objective);

  auto invalid = measurement;
  invalid.distance = -1;
  BOOST_CHECK_THROW(constraint->setMeasurements({invalid, other}), std::domain_error);
  auto invalidSecond = other;
  invalidSecond.distance = -1;
  BOOST_CHECK_THROW(constraint->setMeasurements({replacement, invalidSecond}), std::domain_error);
  BOOST_CHECK_SMALL((constraint->bounds() - firstBounds).norm(), 1e-12);
  BOOST_CHECK_THROW(constraint->setMeasurements({measurement}), std::invalid_argument);
  BOOST_CHECK_THROW(constraint->setMeasurements({measurement, measurement}), std::domain_error);
  BOOST_CHECK_SMALL((constraint->rows().row(0) - 0.005 * row.directionalJacobian).norm(), 1e-12);
  constraint->clear();
  BOOST_CHECK_SMALL(constraint->rows().norm(), 1e-12);
  BOOST_CHECK_SMALL(constraint->bounds().norm(), 1e-12);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), taskCount);
  auto unconstrained = solver.problem().add(acceleration == desiredAcceleration,
                                            {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.0)});
  BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
  BOOST_CHECK_SMALL((acceleration->value() - desiredAcceleration).norm(), 1e-5);
  solver.problem().remove(*unconstrained);
  constraint->setMeasurements({measurement, other});
  BOOST_CHECK_GT(constraint->rows().norm(), 1e-12);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), taskCount);
  solver.removeConstraintSet(*constraint);
  BOOST_CHECK(!gui->hasElement({"SSM+PFL", "Pairs", "hand"}, "wrist"));
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), originalTasks);

  config.add("Weight", 1e-6);
  auto soft = mc_solver::ConstraintSetLoader::load<mc_solver::SSMPFLConstraint>(solver, config);
  solver.addConstraintSet(*soft);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), originalTasks + 3);
  soft->setMeasurements({measurement, other});
  BOOST_CHECK_SMALL(soft->softSlack(0), 1e-12);
  const Eigen::VectorXd softDirection = soft->rows().row(0).transpose();
  const Eigen::VectorXd violatingTarget = softDirection * ((soft->bounds()[0] + 0.1) / softDirection.squaredNorm());
  auto softObjective = solver.problem().add(acceleration == violatingTarget,
                                            {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1e6)});
  BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
  BOOST_CHECK_GT(soft->softSlack(0), 0.01);
  BOOST_CHECK_LE(soft->rows().row(0).dot(acceleration->value()) - soft->bounds()[0] - soft->softSlack(0), 1e-7);
  solver.problem().remove(*softObjective);
  solver.removeConstraintSet(*soft);
  BOOST_CHECK_EQUAL(solver.problem().tasks().size(), originalTasks);

  mc_solver::TasksQPSolver tasksSolver(robots, 0.005);
  BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load(tasksSolver, config), std::runtime_error);
}
