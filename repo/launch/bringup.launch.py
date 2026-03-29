"""Dual-backend bringup for the chess manipulator digital twin."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    sim_backend = LaunchConfiguration("sim_backend")
    rviz = LaunchConfiguration("rviz")
    engine_backend = LaunchConfiguration("engine_backend")
    engine_executable = LaunchConfiguration("engine_executable")
    simulate_isaac_completion = LaunchConfiguration("simulate_isaac_completion")
    shared_parameters = [
        os.path.join(pkg_share, "config", "digital_twin.yaml"),
        {
            "sim_backend": sim_backend,
            "engine_backend": engine_backend,
            "engine_executable": engine_executable,
            "simulate_isaac_completion": simulate_isaac_completion,
        },
    ]

    manager = Node(
        package="chess_manipulator",
        executable="chess_manager",
        parameters=shared_parameters,
        output="screen",
    )
    white_executor = Node(
        package="chess_manipulator",
        executable="robot_executor",
        parameters=shared_parameters,
        output="screen",
    )
    relay = Node(
        package="chess_manipulator",
        executable="trajectory_relay",
        parameters=[
            *shared_parameters,
            {
                "planned_trajectory_topic": "/white/planned_trajectory",
                "execution_feedback_topic": "/white/execution_feedback",
            },
        ],
        output="screen",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(pkg_share, "rviz", "config.rviz")],
        condition=IfCondition(rviz),
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "gazebo_legacy.launch.py")),
        condition=IfCondition(PythonExpression(["'", sim_backend, "' == 'gazebo'"])),
    )
    ros_gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "ros_gz.launch.py")),
        launch_arguments={"rviz": rviz}.items(),
        condition=IfCondition(PythonExpression(["'", sim_backend, "' == 'ros_gz'"])),
    )
    isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "isaac_sim.launch.py")),
        condition=IfCondition(PythonExpression(["'", sim_backend, "' == 'isaac'"])),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_backend", default_value="isaac"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("engine_backend", default_value="stockfish"),
            DeclareLaunchArgument("engine_executable", default_value=""),
            DeclareLaunchArgument("simulate_isaac_completion", default_value="false"),
            manager,
            white_executor,
            relay,
            gazebo_launch,
            ros_gz_launch,
            isaac_launch,
            rviz_node,
        ]
    )
