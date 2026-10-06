#!/usr/bin/env python3
"""Launch the Gazebo cell (UR3 + Robotiq 2F-85, table, 5 cubes, 3 zones, overhead camera)
+ MoveIt 2 + skill_server (motion skills) + scene_perception (camera). task_runner.py is
run separately, once this is up, for each natural-language command -- see README.

Reuses ur_simulation_gz's ur_sim_control.launch.py and ur_moveit_config's
ur_moveit.launch.py unmodified, the same way letter_writer's launch file
does (see that package's launch file docstring for the macOS/RoboStack
xacro-concurrency workaround this also relies on).
"""

import glob
import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _macos_python_preload_env():
    """RoboStack/macOS workaround -- see letter_writer's launch file for the
    full explanation. No-op on Linux."""
    if sys.platform != "darwin" or "CONDA_PREFIX" not in os.environ:
        return {}
    candidates = glob.glob(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libpython3.*.dylib"))
    return {"DYLD_INSERT_LIBRARIES": candidates[0]} if candidates else {}


def _python_node_settings():
    """How to start this package's Python nodes. On macOS/RoboStack the script's
    `#!/usr/bin/env python3` goes through SIP-protected /usr/bin/env, which strips
    DYLD_LIBRARY_PATH, and rclpy then cannot load this package's service typesupport
    (same issue as run_command.sh). Run the conda python directly with the path set."""
    if sys.platform != "darwin" or "CONDA_PREFIX" not in os.environ:
        return {}, None
    conda = os.environ["CONDA_PREFIX"]
    prefix = get_package_prefix("ur3_llm_control")
    env = {
        "DYLD_LIBRARY_PATH": os.pathsep.join(
            [os.path.join(prefix, "lib"), os.path.join(conda, "lib"), os.environ.get("DYLD_LIBRARY_PATH", "")]
        )
    }
    return env, os.path.join(conda, "bin", "python3")


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    startup_delay = LaunchConfiguration("startup_delay")

    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type", default_value="ur3", description="Type of UR robot to simulate."
        ),
        DeclareLaunchArgument(
            "gazebo_gui",
            default_value="false",
            description="Start Gazebo with its own GUI window (Linux only).",
        ),
        DeclareLaunchArgument(
            "startup_delay",
            default_value="15.0",
            description="Seconds to wait for Gazebo/controllers/MoveIt before starting skill_server.",
        ),
    ]

    # Gazebo resolves the gripper's package:// mesh URIs as model://robotiq_description/...,
    # so its share directory has to be on the Gazebo resource path.
    gz_resource_path = AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=PathJoinSubstitution([FindPackageShare("robotiq_description"), ".."]),
    )

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "gazebo_gui": gazebo_gui,
            "launch_rviz": "false",
            "description_package": "ur3_llm_control",
            "description_file": "ur3_with_gripper.urdf.xacro",
            "world_file": PathJoinSubstitution(
                [FindPackageShare("ur3_llm_control"), "worlds", "tabletop.sdf"]
            ),
            "controllers_file": PathJoinSubstitution(
                [FindPackageShare("ur3_llm_control"), "config", "ur_controllers_relaxed.yaml"]
            ),
        }.items(),
    )

    ur_moveit = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "launch",
                    "ur_moveit_config",
                    "ur_moveit.launch.py",
                    ["ur_type:=", ur_type],
                    "use_sim_time:=true",
                    "launch_rviz:=false",
                    "description_package:=ur3_llm_control",
                    "description_file:=ur3_with_gripper.urdf.xacro",
                    "moveit_config_package:=ur3_llm_control",
                    "moveit_config_file:=ur3_with_gripper.srdf.xacro",
                ],
                output="screen",
                name="ur_moveit_launch",
            )
        ],
    )

    # Our own RViz, with a custom config that adds the camera's detections
    # (/perception/markers) and annotated image (/perception/debug_image) --
    # Gazebo has no usable GUI on macOS, so this is the live view of what the
    # camera sees and what the robot is picking/placing.
    rviz = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_ur3_llm_control",
                output="log",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("ur3_llm_control"), "config", "ur3_llm_control.rviz"]
                    ),
                ],
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    gripper_controller_spawner = TimerAction(
        period=startup_delay,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["robotiq_gripper_controller", "-c", "/controller_manager"],
                output="screen",
            )
        ],
    )

    # Camera images from Gazebo (worlds/tabletop.sdf) into ROS 2: the overhead camera
    # feeds scene_perception, the side camera is only recorded for the demo video.
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_bridge",
        arguments=[
            "/overhead_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/overhead_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/video_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        ],
        parameters=[{"use_sim_time": True}],
        output="log",
    )

    skill_server_node = Node(
        package="ur3_llm_control",
        executable="skill_server",
        name="skill_server",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("ur3_llm_control"), "config", "scene.yaml"]),
            {"use_sim_time": True},
        ],
        additional_env=_macos_python_preload_env(),
    )

    delayed_skill_server = TimerAction(period=startup_delay, actions=[skill_server_node])

    py_env, py_prefix = _python_node_settings()
    scene_perception = Node(
        package="ur3_llm_control",
        executable="scene_perception.py",
        name="scene_perception",
        output="screen",
        parameters=[{"use_sim_time": True}],
        additional_env=py_env,
        prefix=py_prefix,
    )

    return LaunchDescription(
        declared_arguments
        + [
            gz_resource_path,
            ur_control,
            camera_bridge,
            scene_perception,
            ur_moveit,
            rviz,
            gripper_controller_spawner,
            delayed_skill_server,
        ]
    )
