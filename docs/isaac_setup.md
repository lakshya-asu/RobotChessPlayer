# Isaac Sim 4.2 Setup

This branch is pivoting the simulator backend from `ros_gz` to Isaac Sim while keeping the ROS-side stack intact:

- MoveIt 2 stays on the ROS side
- perception stays on the ROS side
- the chess engine and DQN adapters stay on the ROS side
- Isaac Sim becomes the primary visual and physics backend

The target runtime is **Isaac Sim 4.2.0**, matching the Ekumen AR4 reference stack. The current robot stays **Franka Panda-first**, and the native Isaac scene now stages both white and black Panda instances for the dual-player demo path.

## Expected Install Layout

The native install path for this repo is:

```bash
${WORKSPACE_ROOT}/third_party/isaac-sim-4.2.0
```

The launcher validates that path by default and also accepts an override:

```bash
ISAAC_ROOT=/path/to/isaac-sim-4.2.0 ./repo/scripts/run_isaac_demo.sh
```

## Manual Install Step

1. Download Isaac Sim `4.2.0` from NVIDIA.
2. Extract it into the workspace-local path:

```bash
export WORKSPACE_ROOT=/home/flux/Desktop/chessPlayer
mkdir -p "${WORKSPACE_ROOT}/third_party"
tar -xf ~/Downloads/isaac-sim-4.2.0-linux-x86_64.tar.gz -C "${WORKSPACE_ROOT}/third_party"
mv "${WORKSPACE_ROOT}/third_party/isaac-sim" "${WORKSPACE_ROOT}/third_party/isaac-sim-4.2.0"
```

3. Verify the native launch script exists:

```bash
ls "${WORKSPACE_ROOT}/third_party/isaac-sim-4.2.0/python.sh"
```

4. Verify the repo sees the install:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/check_isaac_install.sh
```

## Why 4.2.0

The Isaac pivot is following the same baseline Ekumen used for AR4:

- native Isaac Sim app
- ROS-side MoveIt
- simulator-side ROS bridge
- a dedicated Isaac runner script instead of trying to embed everything inside a ROS launch file

This reduces simulator-version drift while we port the Panda stack.

## Current Isaac App Contract

The existing Isaac app in `repo/isaac_app` is now written to support both:

- Isaac Sim 4.2 style `omni.isaac.*` Python modules
- newer `isaacsim.*` Python modules

For the pivot branch, `4.2.0` is the intended target.

## Current ROS <-> Isaac Topic Contract

- White Panda subscribes: `/white/command/joint_trajectory`
  - `trajectory_msgs/msg/JointTrajectory`
- White Panda publishes: `/white/joint_states`
  - `sensor_msgs/msg/JointState`
- White Panda publishes: `/white/status`
  - `std_msgs/msg/String`
- White Panda publishes: `/white/execution_result`
  - `std_msgs/msg/String`
- Black Panda subscribes: `/black/command/joint_trajectory`
  - `trajectory_msgs/msg/JointTrajectory`
- Black Panda publishes: `/black/joint_states`
  - `sensor_msgs/msg/JointState`
- Black Panda publishes: `/black/status`
  - `std_msgs/msg/String`
- Black Panda publishes: `/black/execution_result`
  - `std_msgs/msg/String`
- Shared board state remains:
  - `/chess/board_state`

The Isaac app is now the authoritative source for both robot visuals and per-robot execution results. The ROS-side bridge remains transitional.

## Bringup Flow

Build the ROS workspace first:

```bash
cd /home/flux/Desktop/chessPlayer
source /opt/ros/humble/setup.bash
env PATH=/usr/bin:/bin:$PATH colcon --log-base log_sys build \
  --build-base build_sys \
  --install-base install_sys \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

Start the native Isaac GUI app:

```bash
cd /home/flux/Desktop/chessPlayer/repo
./scripts/run_isaac_demo.sh
```

The launcher forwards the default white/black namespaces above, so the app starts with both Pandas active unless you override the topics explicitly.

Start the ROS-side Isaac bringup in another terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 launch chess_manipulator bringup.launch.py sim_backend:=isaac
```

If you want ROS launch to start the native Isaac app process as well, use:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 launch chess_manipulator bringup.launch.py \
  sim_backend:=isaac \
  launch_native_isaac_app:=true
```

Run one ROS-driven move in a third terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator demo_turn --think-time 0.25
```

## What Still Requires Manual Work

## Current Milestones

Working now:

- `demo_turn` succeeds end to end against native Isaac 4.2
- white and black Pandas both launch in the same Isaac scene
- white and black use separate execution topics
- white and black use separate MoveIt planning services
- a short alternating white-vs-black match can run through the coordinator on Isaac

Still in progress:

- make perception authoritative instead of fallback-backed
- promote the DQN side from heuristic-backed runtime to a trained checkpoint
- add attach / detach and capture handling in the Isaac scene
- extend the short match flow into a polished engine-vs-DQN demo

## Dual-Match Launcher

The fastest way to run the current two-robot branch is:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/run_engine_vs_learned_demo.sh
```

For a headless smoke run:

```bash
cd /home/flux/Desktop/chessPlayer
ISAAC_HEADLESS=true ./repo/scripts/run_engine_vs_learned_demo.sh
```

The learned-player launcher expects a promoted checkpoint artifact by default. Promote one with:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/promote_learned_checkpoint.sh \
  /home/flux/Desktop/chessPlayer/repo/results/training/black_dqn.pt
```

If you need to reset the current scene and ROS processes before relaunching:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/reset_demo_state.sh
```

## Notes

- `run_isaac_demo.sh` strips Conda state before launching Isaac. Keep using it instead of launching `python.sh` manually.
- The launcher now defaults only to the workspace-local Isaac 4.2.0 path unless you override `ISAAC_ROOT`.
- This doc is the native install path for the pivot branch. The older ROS GZ instructions no longer describe the intended simulator target.
