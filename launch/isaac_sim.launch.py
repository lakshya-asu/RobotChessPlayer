"""Isaac-first bringup for the ROS-side digital twin."""

from __future__ import annotations

import math
import os
from typing import Any, Dict

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ROBOT_EDGE_CLEARANCE_M = 0.18


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _placement(base_xyz: tuple[float, float, float], yaw_deg: float) -> Dict[str, str]:
    yaw_rad = math.radians(yaw_deg)
    return {
        "base_xyz": f"{base_xyz[0]} {base_xyz[1]} {base_xyz[2]}",
        "base_rpy": f"0 0 {yaw_rad}",
    }


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    moveit_share = get_package_share_directory("moveit_resources_panda_moveit_config")
    config = os.path.join(pkg_share, "config", "digital_twin.yaml")
    simulate_isaac_completion = LaunchConfiguration("simulate_isaac_completion")
    moveit_enabled = LaunchConfiguration("moveit")
    perception_enabled = LaunchConfiguration("perception")
    xacro_file = os.path.join(pkg_share, "description", "manipulator.urdf.xacro")
    runtime_config = _load_yaml(config)["/**"]["ros__parameters"]
    board_origin = runtime_config["board_origin"]
    square_size_m = float(runtime_config["square_size_m"])
    center_x = float(board_origin[0]) + 3.5 * square_size_m
    center_y = float(board_origin[1]) + 3.5 * square_size_m
    offset_y = 4.0 * square_size_m + ROBOT_EDGE_CLEARANCE_M
    white_base = (center_x, center_y - offset_y, 0.0)
    black_base = (center_x, center_y + offset_y, 0.0)

    white_robot_description = xacro.process_file(
        xacro_file,
        mappings=_placement(white_base, 90.0),
    ).toxml()
    black_robot_description = xacro.process_file(
        xacro_file,
        mappings=_placement(black_base, -90.0),
    ).toxml()
    srdf_path = os.path.join(moveit_share, "config", "panda_arm.srdf.xacro")
    robot_description_semantic = xacro.process_file(srdf_path).toxml()

    moveit_common_parameters = {
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
        "use_sim_time": False,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    moveit_common_parameters.update(
        _load_yaml(os.path.join(moveit_share, "config", "moveit_controllers.yaml"))
    )
    white_moveit_parameters = {
        **moveit_common_parameters,
        "robot_description": white_robot_description,
    }
    black_moveit_parameters = {
        **moveit_common_parameters,
        "robot_description": black_robot_description,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "simulate_isaac_completion",
                default_value="false",
                description="Keep false for native Isaac Sim 4.2 execution. Enable only for ROS-side smoke tests.",
            ),
            DeclareLaunchArgument("moveit", default_value="true"),
            DeclareLaunchArgument("perception", default_value="true"),
            Node(
                package="chess_manipulator",
                executable="isaac_bridge",
                name="white_isaac_bridge",
                parameters=[
                    config,
                    {
                        "sim_backend": "isaac",
                        "simulate_isaac_completion": simulate_isaac_completion,
                        "planned_trajectory_topic": "/white/planned_trajectory",
                        "execution_feedback_topic": "/white/execution_feedback",
                        "joint_state_topic": "/joint_states",
                        "source_joint_state_topic": "/white/joint_states",
                        "command_topic": "/white/command/joint_trajectory",
                        "isaac_execution_result_topic": "/white/execution_result",
                    },
                ],
                output="screen",
            ),
            Node(
                package="chess_manipulator",
                executable="isaac_bridge",
                name="black_isaac_bridge",
                parameters=[
                    config,
                    {
                        "sim_backend": "isaac",
                        "simulate_isaac_completion": simulate_isaac_completion,
                        "planned_trajectory_topic": "/black/planned_trajectory",
                        "execution_feedback_topic": "/black/execution_feedback",
                        "joint_state_topic": "/black/joint_states",
                        "source_joint_state_topic": "/black/joint_states",
                        "command_topic": "/black/command/joint_trajectory",
                        "isaac_execution_result_topic": "/black/execution_result",
                    },
                ],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"use_sim_time": True, "robot_description": white_robot_description}],
            ),
            Node(
                package="chess_manipulator",
                executable="board_perception",
                output="screen",
                parameters=[
                    config,
                    {
                        "use_sim_time": False,
                        "segmentation_topic": "/perception/camera/segmentation",
                        "use_segmentation_mask": True,
                    },
                ],
                condition=IfCondition(perception_enabled),
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[white_moveit_parameters],
                arguments=["--ros-args", "--log-level", "info"],
                condition=IfCondition(moveit_enabled),
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace="black_moveit",
                output="screen",
                parameters=[black_moveit_parameters],
                remappings=[("/joint_states", "/black/joint_states")],
                arguments=["--ros-args", "--log-level", "info"],
                condition=IfCondition(moveit_enabled),
            ),
        ]
    )
