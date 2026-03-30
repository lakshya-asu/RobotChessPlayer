"""ROS GZ-first bringup for the chess manipulator world."""

from __future__ import annotations

import os
from typing import Any, Dict

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    moveit_share = get_package_share_directory("moveit_resources_panda_moveit_config")
    panda_description_share = get_package_share_directory("moveit_resources_panda_description")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    moveit_enabled = LaunchConfiguration("moveit")
    rviz = LaunchConfiguration("rviz")
    perception_enabled = LaunchConfiguration("perception")

    pkg_resource_path = os.pathsep.join(
        [
            pkg_share,
            os.path.dirname(pkg_share),
            panda_description_share,
            os.path.dirname(panda_description_share),
            os.path.join(pkg_share, "models"),
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ]
    )
    joint_command_bridges = [
        f"/panda_joint{joint_index}_position_cmd@std_msgs/msg/Float64]gz.msgs.Double"
        for joint_index in range(1, 8)
    ]

    world_path = os.path.join(pkg_share, "worlds", "chesset_ros_gz.sdf")
    gui_config_path = os.path.join(pkg_share, "config", "ros_gz_gui.config")
    xacro_file = os.path.join(pkg_share, "description", "manipulator.ros_gz.urdf.xacro")
    robot_description = xacro.process_file(xacro_file).toxml()

    srdf_path = os.path.join(moveit_share, "config", "panda_arm.srdf.xacro")
    robot_description_semantic = xacro.process_file(srdf_path).toxml()
    moveit_parameters = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": _load_yaml(
            os.path.join(moveit_share, "config", "kinematics.yaml")
        ),
        "robot_description_planning": _load_yaml(
            os.path.join(moveit_share, "config", "joint_limits.yaml")
        ),
        "ompl": _load_yaml(os.path.join(moveit_share, "config", "ompl_planning.yaml")),
        "planning_pipelines": ["ompl"],
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "use_sim_time": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    moveit_parameters.update(
        _load_yaml(os.path.join(moveit_share, "config", "moveit_controllers.yaml"))
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={
            "gz_args": (
                f'-r "{world_path}" '
                f'--gui-config "{gui_config_path}" '
                "--render-engine-gui ogre "
                "--render-engine-server ogre"
            )
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )

    joint_state_filter = Node(
        package="chess_manipulator",
        executable="joint_state_filter",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "panda_chess_bot",
            "-topic",
            "robot_description",
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/world/default/model/panda_chess_bot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
            *joint_command_bridges,
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("/world/default/model/panda_chess_bot/joint_state", "/joint_states_raw"),
            (
                "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera/image",
                "/perception/camera/image_raw",
            ),
            (
                "/world/default/model/overhead_camera/link/camera_link/sensor/overhead_camera/camera_info",
                "/perception/camera/camera_info",
            ),
        ],
        output="screen",
    )

    trajectory_controller_node = Node(
        package="chess_manipulator",
        executable="ros_gz_trajectory_controller",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    perception_node = Node(
        package="chess_manipulator",
        executable="board_perception",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(perception_enabled),
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_parameters],
        arguments=["--ros-args", "--log-level", "info"],
        condition=IfCondition(moveit_enabled),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(moveit_share, "launch", "moveit.rviz")],
        output="screen",
        condition=IfCondition(rviz),
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": robot_description,
                "robot_description_semantic": robot_description_semantic,
                "robot_description_kinematics": moveit_parameters["robot_description_kinematics"],
                "robot_description_planning": moveit_parameters["robot_description_planning"],
            }
        ],
    )

    load_control_stack = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                TimerAction(
                    period=1.0,
                    actions=[
                        trajectory_controller_node,
                        move_group_node,
                    ],
                )
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("moveit", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("perception", default_value="true"),
            SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=pkg_resource_path),
            gz_sim,
            robot_state_publisher,
            clock_bridge,
            joint_state_filter,
            spawn_robot,
            load_control_stack,
            perception_node,
            rviz_node,
        ]
    )
