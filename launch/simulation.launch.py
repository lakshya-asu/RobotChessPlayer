"""Backward-compatible launch entry point for the ros_gz demo."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("chess_manipulator")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_share, "launch", "bringup.launch.py")
                ),
                launch_arguments={"sim_backend": "ros_gz"}.items(),
            )
        ]
    )
