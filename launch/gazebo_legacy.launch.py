"""Legacy Gazebo bringup for the original Panda chess simulation."""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "chess_manipulator"
    pkg_share = get_package_share_directory(pkg_name)
    xacro_file = os.path.join(pkg_share, "description", "manipulator.urdf.xacro")
    robot_description_content = xacro.process_file(xacro_file).toxml()

    models_path = os.path.join(pkg_share, "models")
    if "GAZEBO_MODEL_PATH" in os.environ:
        model_path = os.environ["GAZEBO_MODEL_PATH"] + ":" + models_path
    else:
        model_path = models_path

    world_path = os.path.join(pkg_share, "worlds", "chesset.world")

    return LaunchDescription(
        [
            SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description_content}],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
                ),
                launch_arguments={"world": world_path}.items(),
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=["-topic", "robot_description", "-entity", "panda_chess_bot"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_trajectory_controller"],
                output="screen",
            ),
        ]
    )
