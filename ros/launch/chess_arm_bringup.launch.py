from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="franka_chess_arm",
                executable="ros_bridge.py",  # if you later install an entrypoint
                name="franka_chess_bridge",
                output="screen",
            )
        ]
    )
