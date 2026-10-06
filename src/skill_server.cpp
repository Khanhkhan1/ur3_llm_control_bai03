#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_model/revolute_joint_model.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "ur3_llm_control/srv/home.hpp"
#include "ur3_llm_control/srv/pick.hpp"
#include "ur3_llm_control/srv/place.hpp"
#include "ur3_llm_control/srv/update_scene.hpp"

using moveit::planning_interface::MoveGroupInterface;
using GripperCommand = control_msgs::action::GripperCommand;

namespace {

constexpr int kMaxAttempts = 5;

// Robotiq 2F-85 knuckle angle [rad]: 0.0 fully open, 0.8 fully closed (the joint limits).
// Commands stay clear of both limits: in Gazebo (DART) a joint resting exactly on a limit
// ignores its velocity command, so a gripper sent to 0.0 would never close again.
constexpr double kGripperOpen = 0.03;
constexpr double kGripperClose = 0.70;
// Closing on a 5 cm cube stops the fingers near 0.33 rad; closing on nothing reaches ~0.69.
constexpr double kGripperMissPosition = 0.62;
constexpr double kGripperMinGrasp = 0.15;
// The attached cube is modelled slightly smaller than the real one so that setting it
// down on the table is not reported as a collision.
constexpr double kAttachedShrink = 0.002;

const std::vector<std::string> kGripperLinks = {
  "tool0", "ur_to_robotiq_link", "robotiq_adapter_ref", "robotiq_85_base_link",
  "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
  "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
  "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
  "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"};

// Display color of each planning-scene object in RViz (MoveIt's default is plain green
// for everything), matching the cubes in Gazebo.
std_msgs::msg::ColorRGBA objectColor(const std::string & id)
{
  static const std::map<std::string, std::array<float, 3>> colors{
    {"red_cube", {0.85f, 0.05f, 0.05f}}, {"yellow_cube", {0.9f, 0.8f, 0.05f}},
    {"blue_cube", {0.05f, 0.2f, 0.9f}},  {"green_cube", {0.05f, 0.7f, 0.1f}},
    {"purple_cube", {0.55f, 0.1f, 0.8f}}, {"table", {0.62f, 0.62f, 0.6f}}};
  std_msgs::msg::ColorRGBA c;
  auto it = colors.find(id);
  const auto rgb = it != colors.end() ? it->second : std::array<float, 3>{0.6f, 0.6f, 0.6f};
  c.r = rgb[0];
  c.g = rgb[1];
  c.b = rgb[2];
  c.a = 1.0f;
  return c;
}

moveit_msgs::msg::ObjectColor objectColorMsg(const std::string & id)
{
  moveit_msgs::msg::ObjectColor oc;
  oc.id = id;
  oc.color = objectColor(id);
  return oc;
}

// tool0 pointing straight down, rotated `yaw` about z (the fingers close along tool0 x).
geometry_msgs::msg::Pose topDownPose(double x, double y, double z, double yaw)
{
  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = z;
  tf2::Quaternion q;
  q.setRPY(M_PI, 0, yaw);
  p.orientation = tf2::toMsg(q);
  return p;
}

// A cube looks the same every 90 deg; grasp it with the wrist turned as little as possible.
double normalizeCubeYaw(double yaw)
{
  return std::remainder(yaw, M_PI / 2.0);
}

// One IK branch for every top-down pose over the table (elbow up, wrist_2 at -pi/2):
// each IK answer must lie within kBranchTolerance of kBranchCenter, and IK is seeded from
// kReadyJoints (an arm pose above the table center) when the arm is not already in the
// branch. Without this, KDL's random restarts hop between the eight UR solutions and the
// arm swings through odd configurations from one move to the next.
const std::vector<std::string> kArmJoints = {
  "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
const std::vector<double> kReadyJoints = {-0.36, -1.52, 1.21, -1.25, -1.571, -1.93};
const std::vector<double> kBranchCenter = {-0.36, -1.6, 1.5, -1.4, -1.571, 0.0};
const std::vector<double> kBranchTolerance = {1.5, 1.4, 1.4, 1.9, 0.6, M_PI};

// MoveIt keeps the continuous wrist_3_joint inside [-pi, pi] (planning state, IK answers,
// interpolation across +-pi) while the controller tracks the unwrapped angle, so a plan
// can start 2*pi away from where the wrist really is, or jump by 2*pi mid-way -- the
// controller then aborts or spins the wrist a full turn. Re-express each continuous joint
// relative to its measured position and to the previous point, so the trajectory is
// continuous in the controller's joint space.
void makeContinuous(
  trajectory_msgs::msg::JointTrajectory & traj, const sensor_msgs::msg::JointState & current,
  const moveit::core::RobotModelConstPtr & model)
{
  for (size_t j = 0; j < traj.joint_names.size(); ++j) {
    const auto * jm = model->getJointModel(traj.joint_names[j]);
    const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(jm);
    if (!revolute || !revolute->isContinuous()) continue;
    auto it = std::find(current.name.begin(), current.name.end(), traj.joint_names[j]);
    if (it == current.name.end()) continue;
    double prev = current.position[std::distance(current.name.begin(), it)];
    for (auto & point : traj.points) {
      double & v = point.positions[j];
      v = prev + std::remainder(v - prev, 2.0 * M_PI);
      prev = v;
    }
  }
}

bool withRetries(
  const std::function<bool()> & attempt, const rclcpp::Logger & logger, const std::string & label)
{
  for (int i = 1; i <= kMaxAttempts; ++i) {
    if (attempt()) return true;
    RCLCPP_WARN(logger, "%s: attempt %d/%d failed, retrying...", label.c_str(), i, kMaxAttempts);
  }
  RCLCPP_ERROR(logger, "%s: giving up after %d attempts", label.c_str(), kMaxAttempts);
  return false;
}

}  // namespace

// Motion skills for the UR3 + Robotiq 2F-85: home / pick / place, plus keeping MoveIt's
// planning scene (table + the cubes the camera sees) up to date for collision checking.
// It never decides *where* anything is -- the coordinates of every pick and place come
// from the overhead camera via task_runner.
class SkillServerNode : public rclcpp::Node
{
public:
  SkillServerNode() : rclcpp::Node("skill_server")
  {
    // MoveGroupInterface::getCurrentState() rejects /joint_states whose stamp is older
    // than the request, and joint_state_broadcaster was observed to stamp 0 here, so
    // the latest message is kept directly (see buildKnownState()). Own callback group:
    // the skills run inside service callbacks and need fresh joint states meanwhile.
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        latest_joint_state_ = msg;
      },
      options);
  }

  // Called once the node is spinning: --params-file overrides were observed (ROS 2 Humble /
  // RoboStack) to be missing from get_parameter() right after construction, so they are
  // read from the raw override map instead.
  void initMoveGroup()
  {
    auto overrides = get_node_parameters_interface()->get_parameter_overrides();
    auto ov_double = [&overrides](const std::string & name, double fallback) {
      auto it = overrides.find(name);
      return it != overrides.end() ? it->second.get<double>() : fallback;
    };
    auto ov_vector = [&overrides](const std::string & name, std::vector<double> fallback) {
      auto it = overrides.find(name);
      return it != overrides.end() ? it->second.get<std::vector<double>>() : fallback;
    };
    if (auto it = overrides.find("planning_group"); it != overrides.end()) {
      planning_group_ = it->second.get<std::string>();
    }
    table_center_ = ov_vector("table_center", table_center_);
    table_size_ = ov_vector("table_size", table_size_);
    cube_size_ = ov_double("cube_size", cube_size_);
    grasp_z_ = ov_double("grasp_z", grasp_z_);
    place_clearance_ = ov_double("place_clearance", place_clearance_);
    approach_height_ = ov_double("approach_height", approach_height_);
    vel_scale_ = ov_double("velocity_scaling", vel_scale_);
    accel_scale_ = ov_double("acceleration_scaling", accel_scale_);

    move_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), planning_group_);
    move_group_->setMaxVelocityScalingFactor(vel_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(10);

    addTable();

    // Own reentrant group: pick/place wait inside a service callback (default, mutually
    // exclusive group) for IK and gripper results, which must still be delivered.
    client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    ik_client_ = create_client<moveit_msgs::srv::GetPositionIK>(
      "compute_ik", rmw_qos_profile_services_default, client_cb_group_);
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(
      this, "robotiq_gripper_controller/gripper_cmd", client_cb_group_);

    using std::placeholders::_1;
    using std::placeholders::_2;
    home_srv_ = create_service<ur3_llm_control::srv::Home>(
      "skill/home", std::bind(&SkillServerNode::handleHome, this, _1, _2));
    pick_srv_ = create_service<ur3_llm_control::srv::Pick>(
      "skill/pick", std::bind(&SkillServerNode::handlePick, this, _1, _2));
    place_srv_ = create_service<ur3_llm_control::srv::Place>(
      "skill/place", std::bind(&SkillServerNode::handlePlace, this, _1, _2));
    scene_srv_ = create_service<ur3_llm_control::srv::UpdateScene>(
      "skill/update_scene", std::bind(&SkillServerNode::handleUpdateScene, this, _1, _2));

    RCLCPP_INFO(
      get_logger(), "skill_server ready: /skill/home, /skill/pick, /skill/place, /skill/update_scene");
  }

private:
  // ---------------------------------------------------------------- planning scene

  moveit_msgs::msg::CollisionObject box(
    const std::string & id, const std::vector<double> & size, double x, double y, double z,
    double yaw)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = move_group_->getPlanningFrame();
    obj.id = id;
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    prim.dimensions.assign(size.begin(), size.end());
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    pose.orientation = tf2::toMsg(q);
    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    return obj;
  }

  double tableTop() const { return table_center_[2] + table_size_[2] / 2.0; }

  // A cube resting on the table, as a MoveIt collision object.
  moveit_msgs::msg::CollisionObject cubeObject(const std::string & name, double x, double y, double yaw)
  {
    return box(
      name, {cube_size_, cube_size_, cube_size_}, x, y, tableTop() + cube_size_ / 2.0, yaw);
  }

  void removeObject(const std::string & name)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = move_group_->getPlanningFrame();
    obj.id = name;
    obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    planning_scene_interface_.applyCollisionObject(obj);
    scene_cubes_.erase(name);
  }

  // The table from worlds/tabletop.sdf: without it OMPL would happily route the arm
  // through it, since MoveIt only knows the robot's own URDF.
  void addTable()
  {
    planning_scene_interface_.applyCollisionObject(
      box("table", table_size_, table_center_[0], table_center_[1], table_center_[2], 0.0),
      objectColor("table"));
    RCLCPP_INFO(get_logger(), "Added 'table' collision object to the planning scene");
  }

  // Attach `name` below tool0, where a cube gripped at grasp_z hangs, so later plans
  // (moving it across the table) account for the cube as part of the robot.
  void attachCube(const std::string & name)
  {
    const double s = cube_size_ - kAttachedShrink;
    moveit_msgs::msg::AttachedCollisionObject aco;
    aco.link_name = "tool0";
    aco.object.header.frame_id = "tool0";
    aco.object.id = name;
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    prim.dimensions = {s, s, s};
    geometry_msgs::msg::Pose pose;
    // tool0's z axis points down while gripping: the cube center is that far along it.
    pose.position.z = grasp_z_ - (tableTop() + cube_size_ / 2.0);
    pose.orientation.w = 1.0;
    aco.object.primitives.push_back(prim);
    aco.object.primitive_poses.push_back(pose);
    aco.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    aco.touch_links = kGripperLinks;
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    diff.robot_state.is_diff = true;
    diff.robot_state.attached_collision_objects.push_back(aco);
    diff.object_colors.push_back(objectColorMsg(name));
    planning_scene_interface_.applyPlanningScene(diff);
  }

  void detachCube(const std::string & name)
  {
    moveit_msgs::msg::AttachedCollisionObject aco;
    aco.link_name = "tool0";
    aco.object.id = name;
    aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    planning_scene_interface_.applyAttachedCollisionObject(aco);
    // Detaching leaves the object in the world; drop that copy, the caller re-adds it
    // where the cube actually is.
    removeObject(name);
  }

  void handleUpdateScene(
    const std::shared_ptr<ur3_llm_control::srv::UpdateScene::Request> req,
    std::shared_ptr<ur3_llm_control::srv::UpdateScene::Response> res)
  {
    if (req->x.size() != req->names.size() || req->y.size() != req->names.size() ||
        req->yaw.size() != req->names.size()) {
      res->status = "INVALID_REQUEST";
      return;
    }
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    std::vector<moveit_msgs::msg::ObjectColor> colors;
    std::set<std::string> seen;
    for (size_t i = 0; i < req->names.size(); ++i) {
      if (req->names[i] == held_object_) continue;
      objects.push_back(cubeObject(req->names[i], req->x[i], req->y[i], req->yaw[i]));
      colors.push_back(objectColorMsg(req->names[i]));
      seen.insert(req->names[i]);
    }
    for (const auto & name : scene_cubes_) {
      if (!seen.count(name)) {
        moveit_msgs::msg::CollisionObject obj;
        obj.header.frame_id = move_group_->getPlanningFrame();
        obj.id = name;
        obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        objects.push_back(obj);
      }
    }
    planning_scene_interface_.applyCollisionObjects(objects, colors);
    scene_cubes_ = seen;
    RCLCPP_INFO(get_logger(), "Planning scene: %zu cubes from the camera", seen.size());
    res->status = "SUCCESS";
  }

  // ---------------------------------------------------------------- motion

  sensor_msgs::msg::JointState::SharedPtr jointState()
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    return latest_joint_state_;
  }

  moveit::core::RobotStatePtr buildKnownState()
  {
    auto state = std::make_shared<moveit::core::RobotState>(move_group_->getRobotModel());
    state->setToDefaultValues();
    if (auto js = jointState()) {
      state->setVariableValues(*js);
    }
    state->update();
    return state;
  }

  // Executes a planned trajectory after making it continuous w.r.t. the measured joints.
  bool execute(moveit_msgs::msg::RobotTrajectory trajectory)
  {
    if (auto js = jointState()) {
      makeContinuous(trajectory.joint_trajectory, *js, move_group_->getRobotModel());
    }
    MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    return move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
  }

  // Plan (OMPL, collision-checked against the planning scene) to the current target, then execute.
  bool planAndExecute()
  {
    MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) return false;
    return execute(plan.trajectory_);
  }

  bool moveNamedTarget(const std::string & name)
  {
    return withRetries(
      [this, &name]() {
        move_group_->setStartStateToCurrentState();
        move_group_->setNamedTarget(name);
        return planAndExecute();
      },
      get_logger(), "moveNamedTarget(" + name + ")");
  }

  // True when the arm joints are inside the IK branch window (see kBranchCenter).
  bool inBranch(const sensor_msgs::msg::JointState & js)
  {
    for (size_t k = 0; k < kArmJoints.size(); ++k) {
      auto it = std::find(js.name.begin(), js.name.end(), kArmJoints[k]);
      if (it == js.name.end()) return false;
      const double v = js.position[std::distance(js.name.begin(), it)];
      const double d = kArmJoints[k] == "wrist_3_joint" ? std::remainder(v - kBranchCenter[k], 2 * M_PI)
                                                         : v - kBranchCenter[k];
      if (std::abs(d) > kBranchTolerance[k]) return false;
    }
    return true;
  }

  // Collision-aware IK for `pose` (joint name -> angle) from MoveIt's /compute_ik.
  // MoveIt's own pose goal may pick an elbow-down solution whose forearm hits the table.
  bool ikJointSolution(const geometry_msgs::msg::Pose & pose, std::map<std::string, double> & out)
  {
    if (!ik_client_->wait_for_service(std::chrono::seconds(2))) {
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    req->ik_request.group_name = move_group_->getName();
    req->ik_request.ik_link_name = "tool0";
    req->ik_request.avoid_collisions = true;
    req->ik_request.timeout = rclcpp::Duration::from_seconds(1.0);
    req->ik_request.pose_stamped.header.frame_id = move_group_->getPlanningFrame();
    req->ik_request.pose_stamped.pose = pose;
    // Seed: the current arm pose when it is already in the branch (shortest motion),
    // otherwise the ready pose. The rest of the robot state comes from move_group.
    req->ik_request.robot_state.is_diff = true;
    auto js = jointState();
    if (!js || !inBranch(*js)) {
      req->ik_request.robot_state.joint_state.name = kArmJoints;
      req->ik_request.robot_state.joint_state.position = kReadyJoints;
    }
    for (size_t k = 0; k < kArmJoints.size(); ++k) {
      moveit_msgs::msg::JointConstraint jc;
      jc.joint_name = kArmJoints[k];
      jc.position = kBranchCenter[k];
      jc.tolerance_above = jc.tolerance_below = kBranchTolerance[k];
      jc.weight = 1.0;
      req->ik_request.constraints.joint_constraints.push_back(jc);
    }
    auto future = ik_client_->async_send_request(req);
    if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
      return false;
    }
    auto res = future.get();
    if (res->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      return false;
    }
    const auto group_joints = move_group_->getJointNames();
    const auto & sol = res->solution.joint_state;
    for (size_t i = 0; i < sol.name.size(); ++i) {
      if (std::find(group_joints.begin(), group_joints.end(), sol.name[i]) != group_joints.end()) {
        out[sol.name[i]] = sol.position[i];
      }
    }
    return out.size() == group_joints.size();
  }

  bool movePoseTarget(const geometry_msgs::msg::Pose & pose, const std::string & label)
  {
    return withRetries(
      [this, &pose]() {
        move_group_->setStartStateToCurrentState();
        std::map<std::string, double> joints;
        if (!ikJointSolution(pose, joints)) {
          RCLCPP_WARN(get_logger(), "no IK solution in the arm's working branch");
          return false;
        }
        move_group_->setJointValueTarget(joints);
        return planAndExecute();
      },
      get_logger(), label);
  }

  // Straight-line (Cartesian) tool motion from `from` to `target`, collision-checked by
  // computeCartesianPath against the planning scene, then time-parameterized.
  bool cartesianTo(
    const geometry_msgs::msg::Pose & from, const geometry_msgs::msg::Pose & target,
    const std::string & label)
  {
    return withRetries(
      [this, &from, &target]() {
        std::vector<geometry_msgs::msg::Pose> waypoints{from, target};

        move_group_->setStartStateToCurrentState();
        moveit_msgs::msg::RobotTrajectory traj_msg;
        double fraction = move_group_->computeCartesianPath(waypoints, 0.005, 0.0, traj_msg);
        if (fraction < 0.95) {
          RCLCPP_WARN(get_logger(), "cartesian path covers only %.0f%%", fraction * 100.0);
          return false;
        }

        auto js = jointState();
        if (js) makeContinuous(traj_msg.joint_trajectory, *js, move_group_->getRobotModel());

        auto start_state = buildKnownState();
        robot_trajectory::RobotTrajectory rt(move_group_->getRobotModel(), planning_group_);
        rt.setRobotTrajectoryMsg(*start_state, traj_msg);
        trajectory_processing::TimeOptimalTrajectoryGeneration totg;
        if (!totg.computeTimeStamps(rt, vel_scale_, accel_scale_)) return false;

        moveit_msgs::msg::RobotTrajectory retimed;
        rt.getRobotTrajectoryMsg(retimed);
        return execute(retimed);
      },
      get_logger(), label);
  }

  // ---------------------------------------------------------------- gripper

  // Current Robotiq knuckle angle from /joint_states (-1 when not known yet).
  double fingerPosition()
  {
    auto js = jointState();
    if (!js) return -1.0;
    for (size_t i = 0; i < js->name.size(); ++i) {
      if (js->name[i] == "robotiq_85_left_knuckle_joint") return js->position[i];
    }
    return -1.0;
  }

  // Moves the fingers towards `position` and returns where they stopped in `reached`:
  // at the target, or earlier on a cube. Squeezing a cube makes the joint velocity
  // jitter, so the controller's stall detection does not always end the goal; the
  // finger position settling in /joint_states counts as done too. The goal is left
  // active, so a grasped cube stays squeezed until the next gripper command.
  bool setGripper(double position, double * reached = nullptr)
  {
    if (!gripper_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "gripper action server not available");
      return false;
    }
    GripperCommand::Goal goal;
    goal.command.position = position;
    goal.command.max_effort = 100.0;
    auto goal_future = gripper_client_->async_send_goal(goal);
    if (goal_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      return false;
    }
    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      return false;
    }
    auto result_future = gripper_client_->async_get_result(goal_handle);
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto last_change = start;
    double last = fingerPosition();
    while (Clock::now() - start < std::chrono::seconds(12)) {
      if (result_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
        auto result = result_future.get();
        if (reached) *reached = result.result->position;
        return result.code == rclcpp_action::ResultCode::SUCCEEDED;
      }
      const double now_pos = fingerPosition();
      const auto now = Clock::now();
      if (std::abs(now_pos - last) > 0.003) {
        last = now_pos;
        last_change = now;
      } else if (now - start > std::chrono::milliseconds(1500) &&
                 now - last_change > std::chrono::milliseconds(700)) {
        RCLCPP_INFO(get_logger(), "gripper goal %.3f: fingers settled at %.3f", position, now_pos);
        if (reached) *reached = now_pos;
        return true;
      }
    }
    RCLCPP_WARN(get_logger(), "gripper goal %.3f: fingers still moving after 12 s", position);
    if (reached) *reached = fingerPosition();
    return false;
  }

  // ---------------------------------------------------------------- skills

  void handleHome(
    const std::shared_ptr<ur3_llm_control::srv::Home::Request>,
    std::shared_ptr<ur3_llm_control::srv::Home::Response> res)
  {
    RCLCPP_INFO(get_logger(), "SKILL home()");
    res->status = moveNamedTarget("up") ? "SUCCESS" : "PLANNING_FAILED";
  }

  void handlePick(
    const std::shared_ptr<ur3_llm_control::srv::Pick::Request> req,
    std::shared_ptr<ur3_llm_control::srv::Pick::Response> res)
  {
    RCLCPP_INFO(
      get_logger(), "SKILL pick(%s) at (%.3f, %.3f), yaw %.1f deg", req->object.c_str(), req->x,
      req->y, req->yaw * 180.0 / M_PI);
    if (!held_object_.empty()) {
      RCLCPP_WARN(get_logger(), "pick() rejected: already holding '%s'", held_object_.c_str());
      res->status = "ALREADY_HOLDING";
      return;
    }

    const double yaw = normalizeCubeYaw(req->yaw);
    const auto approach_pose = topDownPose(req->x, req->y, grasp_z_ + approach_height_, yaw);
    const auto grasp_pose = topDownPose(req->x, req->y, grasp_z_, yaw);

    if (!setGripper(kGripperOpen)) {
      res->status = "GRIPPER_FAILED";
      return;
    }
    if (!movePoseTarget(approach_pose, "pick: approach above " + req->object)) {
      res->status = "PLANNING_FAILED";
      return;
    }
    // The fingers are about to surround this cube: take it out of the collision world
    // (it is attached to the gripper once grasped). The other cubes stay obstacles.
    removeObject(req->object);
    if (!cartesianTo(approach_pose, grasp_pose, "pick: descend to " + req->object)) {
      res->status = "PLANNING_FAILED";
      return;
    }

    double finger = 0.0;
    setGripper(kGripperClose, &finger);
    if (finger > kGripperMissPosition || finger < kGripperMinGrasp) {
      RCLCPP_WARN(
        get_logger(), "pick(%s): fingers closed to %.3f rad, nothing in grip", req->object.c_str(),
        finger);
      setGripper(kGripperOpen);
      cartesianTo(grasp_pose, approach_pose, "pick: retreat");
      res->status = "GRASP_FAILED";
      res->finger_position = finger;
      return;
    }
    RCLCPP_INFO(get_logger(), "pick(%s): fingers stopped at %.3f rad on the cube", req->object.c_str(), finger);
    attachCube(req->object);
    held_object_ = req->object;

    if (!cartesianTo(grasp_pose, approach_pose, "pick: lift " + req->object)) {
      res->status = "PLANNING_FAILED";
      return;
    }
    // Still holding it after the lift? A slipped cube lets the fingers close further.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    finger = fingerPosition();
    res->finger_position = finger;
    if (finger > kGripperMissPosition) {
      RCLCPP_WARN(get_logger(), "pick(%s): cube slipped out while lifting", req->object.c_str());
      detachCube(req->object);
      held_object_.clear();
      res->status = "GRASP_FAILED";
      return;
    }
    res->status = "SUCCESS";
  }

  void handlePlace(
    const std::shared_ptr<ur3_llm_control::srv::Place::Request> req,
    std::shared_ptr<ur3_llm_control::srv::Place::Response> res)
  {
    RCLCPP_INFO(get_logger(), "SKILL place(%s) at (%.3f, %.3f)", req->object.c_str(), req->x, req->y);
    if (held_object_ != req->object) {
      RCLCPP_WARN(
        get_logger(), "place() rejected: not holding '%s' (holding '%s')", req->object.c_str(),
        held_object_.empty() ? "<nothing>" : held_object_.c_str());
      res->status = "NOT_HOLDING";
      return;
    }

    const double z = grasp_z_ + place_clearance_;
    const auto approach_pose = topDownPose(req->x, req->y, grasp_z_ + approach_height_, 0.0);
    const auto place_pose = topDownPose(req->x, req->y, z, 0.0);

    if (!movePoseTarget(approach_pose, "place: approach above target")) {
      res->status = "PLANNING_FAILED";
      return;
    }
    if (!cartesianTo(approach_pose, place_pose, "place: descend")) {
      res->status = "PLANNING_FAILED";
      return;
    }
    if (!setGripper(kGripperOpen)) {
      res->status = "GRIPPER_FAILED";
      return;
    }
    detachCube(req->object);
    held_object_.clear();
    if (!cartesianTo(place_pose, approach_pose, "place: lift")) {
      res->status = "PLANNING_FAILED";
      return;
    }
    // Back in the collision world where it was set down (the camera refines this later).
    planning_scene_interface_.applyCollisionObject(
      cubeObject(req->object, req->x, req->y, 0.0), objectColor(req->object));
    scene_cubes_.insert(req->object);
    res->status = "SUCCESS";
  }

  std::shared_ptr<MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::mutex joint_state_mutex_;
  sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;

  std::string planning_group_{"ur_manipulator"};
  std::vector<double> table_center_{0.29, 0.0, 0.05};
  std::vector<double> table_size_{0.42, 0.72, 0.10};
  double cube_size_{0.05};
  double grasp_z_{0.282};
  double place_clearance_{0.004};
  double approach_height_{0.10};
  double vel_scale_{0.3};
  double accel_scale_{0.3};

  std::set<std::string> scene_cubes_;
  std::string held_object_;

  rclcpp::Service<ur3_llm_control::srv::Home>::SharedPtr home_srv_;
  rclcpp::Service<ur3_llm_control::srv::Pick>::SharedPtr pick_srv_;
  rclcpp::Service<ur3_llm_control::srv::Place>::SharedPtr place_srv_;
  rclcpp::Service<ur3_llm_control::srv::UpdateScene>::SharedPtr scene_srv_;
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SkillServerNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initMoveGroup();

  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
