"""Launch the native Isaac Sim app process using the workspace-local 4.2 install."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    run_script = os.path.join(pkg_share, "scripts", "run_isaac_demo.sh")
    workspace_root = os.path.abspath(os.path.join(pkg_share, os.pardir, os.pardir, os.pardir, os.pardir))
    source_repo_root = os.path.join(workspace_root, "repo")
    isaac_root = os.path.join(workspace_root, "third_party", "isaac-sim-4.2.0")
    headless = LaunchConfiguration("headless")
    white_command_topic = LaunchConfiguration("white_command_topic")
    white_joint_state_topic = LaunchConfiguration("white_joint_state_topic")
    white_status_topic = LaunchConfiguration("white_status_topic")
    white_execution_result_topic = LaunchConfiguration("white_execution_result_topic")
    black_command_topic = LaunchConfiguration("black_command_topic")
    black_joint_state_topic = LaunchConfiguration("black_joint_state_topic")
    black_status_topic = LaunchConfiguration("black_status_topic")
    black_execution_result_topic = LaunchConfiguration("black_execution_result_topic")
    app_cmd = [
        run_script,
        "--white-command-topic",
        white_command_topic,
        "--white-joint-state-topic",
        white_joint_state_topic,
        "--white-status-topic",
        white_status_topic,
        "--white-execution-result-topic",
        white_execution_result_topic,
        "--black-command-topic",
        black_command_topic,
        "--black-joint-state-topic",
        black_joint_state_topic,
        "--black-status-topic",
        black_status_topic,
        "--black-execution-result-topic",
        black_execution_result_topic,
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Launch the native Isaac app without a GUI window.",
            ),
            DeclareLaunchArgument(
                "white_command_topic",
                default_value="/white/command/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "white_joint_state_topic",
                default_value="/white/joint_states",
            ),
            DeclareLaunchArgument("white_status_topic", default_value="/white/status"),
            DeclareLaunchArgument(
                "white_execution_result_topic",
                default_value="/white/execution_result",
            ),
            DeclareLaunchArgument(
                "black_command_topic",
                default_value="/black/command/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "black_joint_state_topic",
                default_value="/black/joint_states",
            ),
            DeclareLaunchArgument("black_status_topic", default_value="/black/status"),
            DeclareLaunchArgument(
                "black_execution_result_topic",
                default_value="/black/execution_result",
            ),
            ExecuteProcess(
                cmd=app_cmd,
                additional_env={
                    "WORKSPACE_ROOT": workspace_root,
                    "SOURCE_REPO_ROOT": source_repo_root,
                    "ISAAC_ROOT": isaac_root,
                },
                output="screen",
                condition=UnlessCondition(headless),
            ),
            ExecuteProcess(
                cmd=app_cmd + ["--headless"],
                additional_env={
                    "WORKSPACE_ROOT": workspace_root,
                    "SOURCE_REPO_ROOT": source_repo_root,
                    "ISAAC_ROOT": isaac_root,
                },
                output="screen",
                condition=IfCondition(headless),
            ),
        ]
    )
