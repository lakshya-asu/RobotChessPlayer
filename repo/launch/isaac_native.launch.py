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
    headless = LaunchConfiguration("headless")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Launch the native Isaac app without a GUI window.",
            ),
            ExecuteProcess(
                cmd=[run_script],
                output="screen",
                condition=UnlessCondition(headless),
            ),
            ExecuteProcess(
                cmd=[run_script, "--headless"],
                output="screen",
                condition=IfCondition(headless),
            ),
        ]
    )
