"""Isaac-first bringup for the ROS-side digital twin."""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    config = os.path.join(pkg_share, "config", "digital_twin.yaml")
    simulate_isaac_completion = LaunchConfiguration("simulate_isaac_completion")
    xacro_file = os.path.join(pkg_share, "description", "manipulator.urdf.xacro")
    robot_description = xacro.process_file(xacro_file).toxml()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "simulate_isaac_completion",
                default_value="false",
                description="Keep false for native Isaac Sim 4.2 execution. Enable only for ROS-side smoke tests.",
            ),
            Node(
                package="chess_manipulator",
                executable="isaac_bridge",
                parameters=[config, {"sim_backend": "isaac", "simulate_isaac_completion": simulate_isaac_completion}],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"use_sim_time": True, "robot_description": robot_description}],
            ),
        ]
    )
