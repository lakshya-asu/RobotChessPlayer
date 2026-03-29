# Robot Chess Player

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Isaac Sim 4.2](https://img.shields.io/badge/Simulator-Isaac%20Sim%204.2-4A90E2)](https://developer.nvidia.com/isaac/sim)
[![MoveIt 2](https://img.shields.io/badge/Motion-MoveIt%202-6C63FF)](https://moveit.ros.org/)
[![Chess Stack](https://img.shields.io/badge/Reasoning-Stockfish%20%2B%20DQN-2E8B57)](#current-scope)
[![Status](https://img.shields.io/badge/Status-Active%20Prototype-F5A623)](#current-scope)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./repo/package.xml)

Robot Chess Player is a ROS 2 Humble research prototype for autonomous chess manipulation with Franka Panda robots. The project is building toward a full end-to-end loop where vision reconstructs board state in FEN, high-level players choose legal moves, and MoveIt 2 plans physical pick-and-place actions in simulation.

This branch is the active **Isaac Sim pivot**. We are reusing the ROS, MoveIt, perception, and gameplay systems already built in the repo while replacing the simulator backend with **Isaac Sim 4.2.0**, following the same version family used in Ekumen's AR4 Isaac work.

The longer-term goal is a dual-robot match:

- white side driven by a traditional chess engine
- black side driven by a DQN-based reinforcement learning player
- both robots taking turns from the same perceived board state
- both robots executing moves through MoveIt and Isaac Sim

## Problem Statement

This project tackles a hard multi-layer robotics problem:

1. perceive a cluttered chessboard and convert it into machine-usable state
2. reason over that state using chess logic and learned policies
3. transform symbolic moves into collision-aware robot motion
4. manipulate physical pieces reliably in simulation
5. verify outcomes through perception before the next turn begins

That makes the system part manipulation stack, part vision stack, part planning stack, and part game-playing system.

## What Has Been Built

### Core ROS 2 workspace

The workspace is split into two ROS packages:

- `repo/`
  - main `chess_manipulator` package
  - motion planning, perception, simulator bridges, coordinator logic, scripts, models, docs, and tests
- `chess_manipulator_msgs/`
  - generated ROS 2 interfaces for chess move execution and engine queries

### Isaac pivot status

The repo is in the middle of replacing the earlier `ros_gz` backend with an Isaac-native path.

Built so far on this branch:

- project-local Isaac Sim 4.2 native install convention
- Isaac launcher script that prefers the workspace-local install
- compatibility layer for both `omni.isaac.*` and newer `isaacsim.*` Python namespaces
- standalone Isaac app scaffold under `repo/isaac_app`
- ROS-side Isaac bridge and Panda playback path
- Panda-first pivot plan and setup docs

### Motion planning and robot execution

The manipulation stack currently includes:

- MoveIt 2 integration for Panda planning
- staged chess move execution
  - home
  - approach
  - move
  - return
- ROS-side trajectory execution for Gazebo joint-position control
- action/service interfaces for move execution and engine queries

### Chess reasoning stack

The current high-level reasoning layer includes:

- chess board-state management
- FEN-based move handling
- Stockfish integration
- coordinator scaffold for multi-player turn logic
- DQN training and inference scaffold
- heuristic fallback path for DQN-side runtime testing

### Perception scaffold

The perception work is already started and includes:

- overhead camera bridge into ROS
- board image processing scaffold
- board-vision utilities
- FEN / board-state publishing topics
- debug image publication for inspection

### Benchmarking and validation

The repo also includes:

- engine benchmark tooling
- planner and system benchmark tooling
- unit tests for board logic, motion planning, RL action space, and perception utilities

## Current Scope

What works today:

- ROS 2 workspace builds cleanly
- custom message package builds cleanly
- ROS 2 workspace builds cleanly
- Isaac-side app scaffold is launchable through the repo script once Isaac 4.2 is installed locally
- MoveIt 2 and the chess stack remain intact on the ROS side
- coordinator scaffold can alternate reasoning backends
- benchmark and RL scaffold code runs

What is still in progress:

- full Panda-in-Isaac execution path
- stable dual-robot world execution
- full camera-to-FEN piece-type perception
- robust capture / attach / detach handling
- trained DQN checkpoint for meaningful play
- full engine-vs-DQN robot-vs-robot demo loop

## Workspace Layout

```text
RobotChessPlayer/
├── README.md
├── .gitignore
├── STL/
├── chess_manipulator_msgs/
└── repo/
    ├── chess_manipulator/
    ├── config/
    ├── description/
    ├── docs/
    ├── launch/
    ├── models/
    ├── results/
    ├── scripts/
    ├── test/
    └── worlds/
```

## Prerequisites

Recommended environment:

- Ubuntu 22.04
- ROS 2 Humble
- Isaac Sim 4.2.0
- MoveIt 2 for ROS 2 Humble
- `python3`, `pip`, `colcon`
- Stockfish on `PATH` or at `/usr/games/stockfish`

Python packages used by the workspace include:

- `numpy`
- `PyYAML`
- `python-chess`
- `opencv-python` or system OpenCV bindings
- `torch` for DQN training and inference

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
env PATH=/usr/bin:/bin:$PATH colcon --log-base log_sys build \
  --build-base build_sys \
  --install-base install_sys \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install_sys/setup.bash
```

## Run The Isaac Pivot Stack

Install Isaac Sim 4.2.0 locally first. The expected path is:

```bash
./third_party/isaac-sim-4.2.0
```

Then launch the Isaac app:

```bash
cd repo
./scripts/run_isaac_demo.sh
```

In a second terminal:

```bash
source /opt/ros/humble/setup.bash
source install_sys/setup.bash
ros2 launch chess_manipulator bringup.launch.py sim_backend:=isaac
```

In a third terminal:

```bash
source /opt/ros/humble/setup.bash
source install_sys/setup.bash
ros2 run chess_manipulator demo_turn --think-time 0.25
```

Detailed native setup notes live in [repo/docs/isaac_setup.md](./repo/docs/isaac_setup.md).

## Run The Coordinator

The coordinator can already alternate reasoning backends at the symbolic level:

```bash
source /opt/ros/humble/setup.bash
source install_sys/setup.bash
ros2 run chess_manipulator game_coordinator \
  --max-plies 4 \
  --white-backend stockfish \
  --black-backend dqn \
  --think-time 0.01
```

## Run Benchmarks

```bash
python3 ./repo/scripts/benchmark_engines.py \
  --suite ./repo/config/benchmark_suite.yaml \
  --json-out ./repo/results/benchmark_results.json \
  --csv-out ./repo/results/benchmark_results.csv
```

Optional engine overrides:

```bash
python3 ./repo/scripts/benchmark_engines.py \
  --suite ./repo/config/benchmark_suite.yaml \
  --stockfish /usr/games/stockfish \
  --sunfish /path/to/sunfish.py
```

## RL Scaffold

Inference:

```bash
source /opt/ros/humble/setup.bash
source install_sys/setup.bash
ros2 run chess_manipulator rl_infer \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
```

Training:

```bash
source /opt/ros/humble/setup.bash
source install_sys/setup.bash
ros2 run chess_manipulator rl_train \
  --checkpoint ./repo/results/rl_dqn.pt \
  --episodes 50 \
  --opponent heuristic \
  --device cpu
```

## Technical Highlights

- ROS 2 action / service control interfaces for chess move execution
- Isaac Sim pivot with Panda-first backend compatibility
- MoveIt 2 integration for Panda arm planning
- board-state and FEN utilities
- dual-player coordinator scaffold
- DQN training / inference scaffold
- STL-based chess-piece model library
- benchmark pipeline for engines and planner behavior

## Notes

- This repo is an active prototype, not a finished research artifact yet.
- `main` preserves the `ros_gz` baseline, while `pivot/isaac-sim-panda-first` retargets the demo to Isaac Sim 4.2.0.
- The simulator pivot keeps the Panda / MoveIt / perception / coordinator stack and replaces only the simulator backend first.
