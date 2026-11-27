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

// Exercise the order-two function over successive solves: a changed budget
// must reach TVM even when the measured distance stays constant.
BOOST_AUTO_TEST_CASE(TestSSMPFLPositionDynamics)
{
  configureRobotLoader();
  auto rm = mc_rbdyn::RobotLoader::get_robot_module("JVRC1");
  auto em = mc_rbdyn::RobotLoader::get_robot_module("env/ground");
  for(double dt : {0.001, 0.005, 0.02})
  {
    auto robots = mc_rbdyn::loadRobotAndEnv(*rm, *em);
    mc_solver::TVMQPSolver solver(robots, dt);
    mc_solver::SSMPFLConstraint constraint(solver.robot(), {"wrist"}, dt,
        {{"hand", {140, 75000, 0.05}}}, 0.5);
    solver.addConstraintSet(constraint);
    mc_solver::SSMPFLConstraint::Measurement m;
    m.region = "hand"; m.link = "wrist"; m.robotBody = "R_WRIST_Y_S";
    m.normal = Eigen::Vector3d::UnitX(); m.distance = 0.08;
    constraint.setMeasurements({m});
    const Eigen::VectorXd direction = constraint.rows().row(0).transpose();
    BOOST_REQUIRE_GT(direction.squaredNorm(), 1e-12);
    const Eigen::VectorXd target = direction * ((constraint.bounds()[0] + 0.1) / direction.squaredNorm());
    const auto acceleration = solver.robot().tvmRobot().alphaD();
    auto objective = solver.problem().add(acceleration == target,
        {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.0)});
    const double firstBudget = constraint.rowData()[0].robotSpeedBudget;
    BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
    BOOST_CHECK_LE(constraint.rows().row(0).dot(acceleration->value()) - constraint.bounds()[0], 1e-7);
    m.humanVelocity = Eigen::Vector3d(-100, 0, 0);
    constraint.setMeasurements({m});
    BOOST_CHECK_GT(firstBudget, 0);
    BOOST_CHECK_SMALL(constraint.rowData()[0].robotSpeedBudget, 1e-12);
    const double changedBound = constraint.bounds()[0];
    BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
    BOOST_CHECK_LE(constraint.rows().row(0).dot(acceleration->value()) - changedBound, 1e-7);
    const Eigen::VectorXd publishedBounds = constraint.bounds();
    m.distance = std::numeric_limits<double>::max();
    BOOST_CHECK_THROW(constraint.setMeasurements({m}), std::domain_error);
    BOOST_CHECK_SMALL((constraint.bounds() - publishedBounds).norm(), 1e-12);
    // The speed budget is finite here, but its acceleration target overflows.
    m.humanVelocity.setZero();
    m.distance = 2e306;
    BOOST_CHECK_THROW(constraint.setMeasurements({m}), std::domain_error);
    BOOST_CHECK_SMALL((constraint.bounds() - publishedBounds).norm(), 1e-12);
    m.distance = 0.08;
    m.normal.x() = std::numeric_limits<double>::quiet_NaN();
    BOOST_CHECK_THROW(constraint.setMeasurements({m}), std::domain_error);
    BOOST_CHECK_SMALL((constraint.bounds() - publishedBounds).norm(), 1e-12);
    constraint.clear();
    BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
    BOOST_CHECK_SMALL((acceleration->value() - target).norm(), 1e-5);
    solver.problem().remove(*objective);
    solver.removeConstraintSet(constraint);
  }
}

BOOST_AUTO_TEST_CASE(TestSSMPFLSoftSlackAcrossTimesteps)
{
  configureRobotLoader();
  auto rm = mc_rbdyn::RobotLoader::get_robot_module("JVRC1");
  auto em = mc_rbdyn::RobotLoader::get_robot_module("env/ground");
  for(double dt : {0.001, 0.005, 0.02})
  {
    for(double weight : {1e-6, 1.0})
    {
      auto robots = mc_rbdyn::loadRobotAndEnv(*rm, *em);
      mc_solver::TVMQPSolver solver(robots, dt);
      mc_solver::SSMPFLConstraint constraint(solver.robot(), {"wrist"}, dt,
          {{"hand", {140, 75000, 0.05}}}, 0.5, 1e-12, 1e-9, weight);
      solver.addConstraintSet(constraint);
      mc_solver::SSMPFLConstraint::Measurement m;
      m.region = "hand"; m.link = "wrist"; m.robotBody = "R_WRIST_Y_S";
      m.normal = Eigen::Vector3d::UnitX(); m.distance = 0.05;
      constraint.setMeasurements({m});
      const Eigen::VectorXd direction = constraint.rows().row(0).transpose();
      BOOST_REQUIRE_GT(direction.squaredNorm(), 1e-12);
      const Eigen::VectorXd target = direction * ((constraint.bounds()[0] + 0.1) / direction.squaredNorm());
      const auto acceleration = solver.robot().tvmRobot().alphaD();
      auto objective = solver.problem().add(acceleration == target,
          {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1e6)});
      const Eigen::MatrixXd row = constraint.rows();
      const Eigen::VectorXd bound = constraint.bounds();
      BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
      BOOST_CHECK_GT(constraint.softSlack(0), 0.01);
      BOOST_CHECK_LE(row.row(0).dot(acceleration->value()) - bound[0] - constraint.softSlack(0), 1e-7);
      constraint.clear();
      BOOST_CHECK_SMALL(constraint.softSlack(0), 1e-12);
      BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
      BOOST_CHECK_SMALL((acceleration->value() - target).norm(), 1e-5);
      solver.problem().remove(*objective);
      solver.removeConstraintSet(constraint);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestSSMPFLGeometryChangesAndZeroMobility)
{
  configureRobotLoader();
  auto rm = mc_rbdyn::RobotLoader::get_robot_module("JVRC1Fixed");
  auto em = mc_rbdyn::RobotLoader::get_robot_module("env/ground");
  for(double position : {-0.15, 0.0, 0.15})
  {
    auto robots = mc_rbdyn::loadRobotAndEnv(*rm, *em);
    mc_solver::TVMQPSolver solver(robots, 0.005);
    for(auto & joint : solver.robot().q()) if(joint.size() == 1) joint[0] = position;
    for(auto & joint : solver.robot().alpha()) if(joint.size() == 1) joint[0] = 0.03;
    solver.robot().forwardKinematics();
    solver.robot().forwardVelocity();
    mc_solver::SSMPFLConstraint constraint(solver.robot(), {"wrist"}, 0.005,
        {{"hand", {140, 75000, 0.05}}}, 0.5);
    solver.addConstraintSet(constraint);
    const auto taskCount = solver.problem().tasks().size();
    mc_solver::SSMPFLConstraint::Measurement m;
    m.region = "hand"; m.link = "wrist"; m.robotBody = "R_WRIST_Y_S";
    m.distance = 0.1;
    for(int step = 0; step < 3; ++step)
    {
      m.normal = Eigen::Vector3d(1.0, 0.4 * step, 0.2 * (step - 1)).normalized();
      m.robotPointBodyLocal = Eigen::Vector3d(0.04 * step, -0.03 * step, 0.02 * step);
      m.humanVelocity = (step - 1) * 0.08 * m.normal;
      constraint.setMeasurements({m});
      const Eigen::MatrixXd row = constraint.rows();
      const Eigen::VectorXd bound = constraint.bounds();
      const Eigen::VectorXd direction = row.row(0).transpose();
      BOOST_REQUIRE_GT(direction.squaredNorm(), 1e-14);
      const Eigen::VectorXd target = direction * ((bound[0] + 0.05) / direction.squaredNorm());
      const auto acceleration = solver.robot().tvmRobot().alphaD();
      auto objective = solver.problem().add(acceleration == target,
          {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.0)});
      BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
      BOOST_CHECK_LE(row.row(0).dot(acceleration->value()) - bound[0], 1e-7);
      solver.problem().remove(*objective);
      BOOST_CHECK_EQUAL(solver.problem().tasks().size(), taskCount);
    }
    solver.removeConstraintSet(constraint);
  }
  auto robots = mc_rbdyn::loadRobotAndEnv(*rm, *em);
  mc_solver::TVMQPSolver solver(robots, 0.005);
  mc_solver::SSMPFLConstraint fixed(solver.robot(), {"base"}, 0.005,
      {{"hand", {140, 75000, 0.05}}}, 0.5);
  solver.addConstraintSet(fixed);
  mc_solver::SSMPFLConstraint::Measurement m;
  m.region = "hand"; m.link = "base"; m.robotBody = solver.robot().mb().body(0).name();
  m.normal = Eigen::Vector3d::UnitX(); m.distance = 0.0; m.humanVelocity = -m.normal;
  fixed.setMeasurements({m});
  BOOST_CHECK_SMALL(fixed.rows().norm(), 1e-12);
  BOOST_CHECK_SMALL(fixed.bounds().norm(), 1e-12);
  BOOST_CHECK_SMALL(fixed.rowData()[0].robotSpeedBudget, 1e-12);

  // A single-joint witness almost orthogonal to its point velocity has a
  // small positive directional inverse mass, without dividing by mobility.
  Eigen::Vector3d pointVelocity = Eigen::Vector3d::Zero();
  for(int i = 0; i < solver.robot().mb().nrBodies(); ++i)
  {
    const auto & name = solver.robot().mb().body(i).name();
    rbd::Jacobian jac(solver.robot().mb(), name);
    if(jac.dof() != 1) continue;
    jac.point(Eigen::Vector3d(0.1, 0.2, 0.3));
    pointVelocity = jac.jacobian(solver.robot().mb(), solver.robot().mbc()).bottomRows(3).col(0);
    if(pointVelocity.norm() > 1e-6) { m.robotBody = name; break; }
  }
  BOOST_REQUIRE_GT(pointVelocity.norm(), 1e-6);
  const Eigen::Vector3d direction = pointVelocity.normalized();
  m.normal = (direction.unitOrthogonal() + 1e-7 * direction).normalized();
  m.robotPointBodyLocal = Eigen::Vector3d(0.1, 0.2, 0.3);
  m.distance = 0.05;
  m.humanVelocity.setZero();
  fixed.setMeasurements({m});
  BOOST_CHECK_GT(fixed.rowData()[0].directionalJacobian.norm(), 0.0);
  BOOST_CHECK_LT(fixed.rowData()[0].directionalJacobian.norm(), 1e-5);
  BOOST_CHECK_GT(fixed.rowData()[0].robotSpeedBudget, 0.0);
  BOOST_CHECK_LT(fixed.rowData()[0].robotSpeedBudget, 1e-5);
  BOOST_CHECK(fixed.bounds().allFinite());
  auto objective = solver.problem().add(solver.robot().tvmRobot().alphaD() == 0.,
      {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(1.0)});
  BOOST_REQUIRE(solver.run(mc_solver::FeedbackType::None));
  solver.problem().remove(*objective);
  solver.removeConstraintSet(fixed);
}
