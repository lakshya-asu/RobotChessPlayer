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
    white_backend = LaunchConfiguration("white_backend")
    black_backend = LaunchConfiguration("black_backend")
    white_engine_executable = LaunchConfiguration("white_engine_executable")
    black_engine_executable = LaunchConfiguration("black_engine_executable")
    white_dqn_checkpoint = LaunchConfiguration("white_dqn_checkpoint")
    black_dqn_checkpoint = LaunchConfiguration("black_dqn_checkpoint")
    white_controller_action_name = LaunchConfiguration("white_controller_action_name")
    black_controller_action_name = LaunchConfiguration("black_controller_action_name")
    coordinator_enabled = LaunchConfiguration("coordinator")
    coordinator_auto_start = LaunchConfiguration("coordinator_auto_start")
    game_log_dir = LaunchConfiguration("game_log_dir")
    simulate_isaac_completion = LaunchConfiguration("simulate_isaac_completion")
    launch_native_isaac_app = LaunchConfiguration("launch_native_isaac_app")
    isaac_headless = LaunchConfiguration("isaac_headless")
    moveit_enabled = LaunchConfiguration("moveit")
    perception_enabled = LaunchConfiguration("perception")
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
        condition=UnlessCondition(coordinator_enabled),
    )
    white_executor = Node(
        package="chess_manipulator",
        executable="robot_executor",
        name="white_robot_executor",
        parameters=[
            *shared_parameters,
            {
                "robot_side": "white",
                "execute_action_name": "/white/execute_move",
                "planned_trajectory_topic": "/white/planned_trajectory",
                "execution_feedback_topic": "/white/execution_feedback",
                "capture_slot_offset": 0,
                "moveit_joint_state_topic": "/joint_states",
                "moveit_compute_ik_service": "/compute_ik",
                "moveit_motion_plan_service": "/plan_kinematic_path",
                "moveit_apply_planning_scene_service": "/apply_planning_scene",
            },
        ],
        output="screen",
    )
    black_executor = Node(
        package="chess_manipulator",
        executable="robot_executor",
        name="black_robot_executor",
        parameters=[
            *shared_parameters,
            {
                "robot_side": "black",
                "execute_action_name": "/black/execute_move",
                "planned_trajectory_topic": "/black/planned_trajectory",
                "execution_feedback_topic": "/black/execution_feedback",
                "capture_slot_offset": 16,
                "moveit_joint_state_topic": "/black/joint_states",
                "moveit_compute_ik_service": "/black_moveit/compute_ik",
                "moveit_motion_plan_service": "/black_moveit/plan_kinematic_path",
                "moveit_apply_planning_scene_service": "/black_moveit/apply_planning_scene",
            },
        ],
        output="screen",
    )
    white_relay = Node(
        package="chess_manipulator",
        executable="trajectory_relay",
        name="white_trajectory_relay",
        parameters=[
            *shared_parameters,
            {
                "planned_trajectory_topic": "/white/planned_trajectory",
                "execution_feedback_topic": "/white/execution_feedback",
                "controller_action_name": white_controller_action_name,
            },
        ],
        output="screen",
    )
    black_relay = Node(
        package="chess_manipulator",
        executable="trajectory_relay",
        name="black_trajectory_relay",
        parameters=[
            *shared_parameters,
            {
                "planned_trajectory_topic": "/black/planned_trajectory",
                "execution_feedback_topic": "/black/execution_feedback",
                "controller_action_name": black_controller_action_name,
            },
        ],
        output="screen",
    )
    coordinator = Node(
        package="chess_manipulator",
        executable="game_coordinator_node",
        parameters=[
            *shared_parameters,
            {
                "white_backend": white_backend,
                "black_backend": black_backend,
                "white_engine_executable": white_engine_executable,
                "black_engine_executable": black_engine_executable,
                "white_dqn_checkpoint": white_dqn_checkpoint,
                "black_dqn_checkpoint": black_dqn_checkpoint,
                "auto_start": coordinator_auto_start,
                "game_log_dir": game_log_dir,
            },
        ],
        output="screen",
        condition=IfCondition(coordinator_enabled),
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
        launch_arguments={
            "simulate_isaac_completion": simulate_isaac_completion,
            "moveit": moveit_enabled,
            "perception": perception_enabled,
        }.items(),
        condition=IfCondition(PythonExpression(["'", sim_backend, "' == 'isaac'"])),
    )
    isaac_native_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "isaac_native.launch.py")),
        launch_arguments={"headless": isaac_headless}.items(),
        condition=IfCondition(
            PythonExpression(
                ["'", sim_backend, "' == 'isaac' and '", launch_native_isaac_app, "' == 'true'"]
            )
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_backend", default_value="isaac"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("engine_backend", default_value="stockfish"),
            DeclareLaunchArgument("engine_executable", default_value=""),
            DeclareLaunchArgument("white_backend", default_value="stockfish"),
            DeclareLaunchArgument("black_backend", default_value="dqn"),
            DeclareLaunchArgument("white_engine_executable", default_value=""),
            DeclareLaunchArgument("black_engine_executable", default_value=""),
            DeclareLaunchArgument("white_dqn_checkpoint", default_value=""),
            DeclareLaunchArgument("black_dqn_checkpoint", default_value=""),
            DeclareLaunchArgument(
                "white_controller_action_name",
                default_value="/white/panda_arm_controller/follow_joint_trajectory",
            ),
            DeclareLaunchArgument(
                "black_controller_action_name",
                default_value="/black/panda_arm_controller/follow_joint_trajectory",
            ),
            DeclareLaunchArgument("coordinator", default_value="false"),
            DeclareLaunchArgument("coordinator_auto_start", default_value="false"),
            DeclareLaunchArgument(
                "game_log_dir",
                default_value="/home/flux/Desktop/chessPlayer/repo/results/game_logs",
            ),
            DeclareLaunchArgument("simulate_isaac_completion", default_value="false"),
            DeclareLaunchArgument("launch_native_isaac_app", default_value="false"),
            DeclareLaunchArgument("isaac_headless", default_value="false"),
            DeclareLaunchArgument("moveit", default_value="true"),
            DeclareLaunchArgument("perception", default_value="true"),
            manager,
            white_executor,
            black_executor,
            white_relay,
            black_relay,
            coordinator,
            gazebo_launch,
            ros_gz_launch,
            isaac_launch,
            isaac_native_launch,
            rviz_node,
        ]
    )
