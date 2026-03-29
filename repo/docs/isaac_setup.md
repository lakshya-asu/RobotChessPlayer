# Isaac Sim 4.2 Setup

This branch is pivoting the simulator backend from `ros_gz` to Isaac Sim while keeping the ROS-side stack intact:

- MoveIt 2 stays on the ROS side
- perception stays on the ROS side
- the chess engine and DQN adapters stay on the ROS side
- Isaac Sim becomes the primary visual and physics backend

The target runtime is **Isaac Sim 4.2.0**, matching the Ekumen AR4 reference stack. The current robot stays **Franka Panda-first**. We only switch robots if Panda-on-Isaac becomes the blocker.

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

- Isaac subscribes: `/isaac/command/joint_trajectory`
  - `trajectory_msgs/msg/JointTrajectory`
- Isaac publishes: `/isaac/joint_states`
  - `sensor_msgs/msg/JointState`
- Isaac publishes: `/isaac/status`
  - `std_msgs/msg/String`
- Isaac publishes: `/isaac/execution_result`
  - `std_msgs/msg/String`
- ROS republishes and coordinates:
  - `/joint_states`
  - `/chess/planned_trajectory`
  - `/chess/execution_feedback`
  - `/chess/board_state`

This is still the transitional contract. The next implementation step is replacing the partial Isaac bridge path with a fuller Panda-in-Isaac control flow patterned after Ekumen's simulator package.

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

The simulator pivot is not complete yet. The remaining manual / implementation work is:

- install Isaac Sim 4.2.0 locally in the project tree
- replace the procedural single-Panda Isaac scene with the full Panda chess scene
- wire the native Panda control path so execution comes from Isaac, not a transitional ROS-side shim
- mirror the single-robot path into dual executors
- keep perception and MoveIt unchanged on the ROS side while changing only the simulator backend

## Notes

- `run_isaac_demo.sh` strips Conda state before launching Isaac. Keep using it instead of launching `python.sh` manually.
- The launcher now defaults only to the workspace-local Isaac 4.2.0 path unless you override `ISAAC_ROOT`.
- This doc is the native install path for the pivot branch. The older ROS GZ instructions no longer describe the intended simulator target.
