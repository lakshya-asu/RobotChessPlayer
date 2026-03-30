#!/usr/bin/env bash
set -euo pipefail

pkill -f 'ros2 launch chess_manipulator bringup.launch.py' || true
pkill -f chess_manager || true
pkill -f robot_executor || true
pkill -f game_coordinator || true
pkill -f game_coordinator_node || true
pkill -f demo_turn || true
pkill -f trajectory_relay || true
pkill -f ros_gz_trajectory_controller || true
pkill -f isaac_bridge || true
pkill -f board_perception || true
pkill -f robot_state_publisher || true
pkill -f gazebo || true
pkill -f gzserver || true
pkill -f gzclient || true
pkill -f 'ros2 launch ros_gz_sim gz_sim.launch.py' || true
pkill -f 'gz sim server' || true
pkill -f 'gz sim gui' || true
pkill -f 'gz sim' || true
pkill -f 'ruby .*gz.* sim' || true
pkill -f spawn_entity.py || true
pkill -f controller_manager || true
pkill -f 'python.sh -m isaac_app' || true
pkill -f 'isaac_app --sync-board-state' || true
pkill -f '/kit/python/bin/python3.10 -m isaac_app' || true
pkill -f 'install_sys/chess_manipulator/share/chess_manipulator/scripts/run_isaac_demo.sh' || true

echo "Stopped chess demo processes."
