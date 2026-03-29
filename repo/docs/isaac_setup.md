# Isaac Sim Setup

This project treats Isaac Sim as the primary showcase backend. The recommended flow is to launch the standalone Isaac GUI app in `repo/isaac_app` and then run the ROS bringup in a second terminal.

## Topic Contract

- Subscribe in Isaac Sim: `/isaac/command/joint_trajectory`
  - Type: `trajectory_msgs/msg/JointTrajectory`
  - Joint order: `panda_joint1` through `panda_joint7`
  - Timing source: each point’s `time_from_start`
- Publish from Isaac Sim: `/isaac/joint_states`
  - Type: `sensor_msgs/msg/JointState`
  - Joint order should match the trajectory joint order
- ROS-side execution payload: `/chess/planned_trajectory`
  - Type: `trajectory_msgs/msg/JointTrajectory`
- Mirrored ROS topics:
  - `/joint_states` is republished by `isaac_bridge`
  - `/chess/execution_feedback` is used to mark accepted and completed execution

## Isaac Scene Expectations

- The app loads a Franka Panda articulation with seven controllable arm joints
- The app procedurally creates a chessboard scene aligned with the project’s board origin parameters
- The default scene includes visible placeholder pieces in the starting position
- The app enables the ROS 2 bridge in Isaac Sim
- The app subscribes to `/isaac/command/joint_trajectory`
- The app publishes joint states to `/isaac/joint_states`
- The ROS bringup continues to publish `/chess/planned_trajectory` and `/chess/execution_feedback` exactly as before
- The board is intentionally camera-friendly and uses alternating colored squares so a demo recording reads clearly

## Suggested Board Assumptions

- Default board origin: `[0.235, -0.266, 0.31]`
- Default square size: `0.076 m`
- Hover height: `0.12 m`
- Pickup height: `0.015 m`

## Bringup Flow

1. Build and source the workspace.
2. Start the Isaac GUI app:

```bash
./scripts/run_isaac_demo.sh
```

This default GUI mode keeps the starting pieces visible but does not live-refresh them from FEN updates. That is currently the more stable mode for recording. If you want to experiment with live redraws later, pass `--sync-board-state`.

3. Start ROS bringup:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ROS_LOG_DIR=/tmp/ros_logs ros2 launch chess_manipulator bringup.launch.py sim_backend:=isaac
```

4. In another shell, execute one engine-backed move:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator demo_turn --think-time 0.25
```

## Notes

- The visible simulator side now lives in `repo/isaac_app`, while the ROS launch remains responsible for board state, planning, and execution orchestration.
- The current bridge still publishes completion acknowledgment on the ROS side after the final trajectory time. That keeps the demo working even when the app is running as a separate GUI process.
- Gazebo legacy completion still comes from `trajectory_relay`; Isaac mode uses `isaac_bridge` for the same execution payload.
- Gazebo remains available for smoke testing, but Isaac Sim is the target demo path.
