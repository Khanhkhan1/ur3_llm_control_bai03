cmake_minimum_required(VERSION 3.8)
project(ur3_llm_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Home.srv"
  "srv/Pick.srv"
  "srv/Place.srv"
  "srv/UpdateScene.srv"
  "srv/DetectObjects.srv"
)

add_executable(skill_server src/skill_server.cpp)
ament_target_dependencies(skill_server
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_ros_planning
  moveit_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  visualization_msgs
  control_msgs
  rclcpp_action
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(skill_server "${cpp_typesupport_target}")

install(TARGETS skill_server
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/workspace.py
  scripts/scene_perception.py
  scripts/world_model.py
  scripts/llm_planner.py
  scripts/task_validator.py
  scripts/skill_executor.py
  scripts/task_runner.py
  scripts/demo_recorder.py
  scripts/run_command.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config worlds urdf srdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
