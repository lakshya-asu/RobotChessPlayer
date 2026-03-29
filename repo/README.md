# Chess Manipulator Digital Twin

A ROS 2 portfolio project that is being migrated into a ROS GZ-first chess manipulation stack with:

- engine-backed move selection
- a dual-player coordinator scaffold for engine-vs-DQN matches
- MoveIt-backed staged pick-and-place planning
- ROS 2 action/service control interfaces
- an overhead camera and perception scaffold
- STL-based physical chess pieces in the active world
- reproducible engine and system benchmarking
- ROS GZ as the active simulator path

## Project Achievements

- Developed a digital twin of a Franka Panda robotic arm for chess manipulation with a simulator-first ROS 2 workflow.
- Implemented deterministic trajectory generation for collision-aware pick-and-place motion across standard chess moves, captures, castling, and promotion.
- Built a ROS-based control interface with engine move selection, execution actions, board-state publication, and simulator bridge topics.
- Added open-source chess engine comparison and robotics-stack benchmark tooling for portfolio reporting.

## Architecture

The workspace is intentionally split into two packages:

- `repo/`
  - `chess_manipulator` as the main `ament_python` package
  - motion planning, engine integration, ROS nodes, launch files, benchmarks, and docs
- `chess_manipulator_msgs/`
  - generated ROS 2 interfaces as a sibling `ament_cmake` package

Core modules inside `chess_manipulator`:

- `chess_manipulator/chess`
  board-state management, SAN/UCI helpers, and engine integration
- `chess_manipulator/coordinator`
  shared game loop scaffolding for alternating engine and DQN players
- `chess_manipulator/motion`
  square calibration, staged motion planning, MoveIt integration, joint-limit validation, and trajectory generation
- `chess_manipulator/rl`
  DQN training, inference, action-space, and board-encoding utilities
- `chess_manipulator/sim`
  backend topic contracts and execution-feedback helpers
- `chess_manipulator/nodes`
  chess manager, robot executor, trajectory relay, ros_gz controller, and perception nodes

## Public ROS 2 Interfaces

- Topic: `/joint_states`
- Topic: `/chess/board_state`
- Topic: `/chess/execution_status`
- Topic: `/white/planned_trajectory` carrying `chess_manipulator_msgs/msg/ExecutionCommand`
- Topic: `/white/execution_feedback`
- Topic: `/perception/camera/image_raw`
- Topic: `/perception/debug/annotated_image`
- Service: `/chess/get_best_move`
- Action: `/chess/execute_move`
- Action: `/white/execute_move`

`/chess/execute_move` is now a coordinator-facing action that forwards the move to the white-side robot executor. In `ros_gz`, Gazebo execution completion is driven by the ROS-side joint-position controller shim backing `/panda_arm_controller/follow_joint_trajectory`.

## Clean Build

Use system Python 3.10 for the workspace build:

```bash
source /opt/ros/humble/setup.bash
env PATH=/usr/bin:/bin:$PATH colcon --log-base log_sys build \
  --build-base build_sys \
  --install-base install_sys \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
```

This avoids Conda Python leaking into ROS interface generation.

## Launch

ROS GZ-first bringup:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ROS_LOG_DIR=/tmp/ros_logs ros2 launch chess_manipulator bringup.launch.py sim_backend:=ros_gz
```

Convenience wrapper:

```bash
/home/flux/Desktop/chessPlayer/repo/scripts/run_ros_gz_demo.sh
```

Backward-compatible simulation entry point:

```bash
ros2 launch chess_manipulator simulation.launch.py
```

Current `ros_gz` baseline:
- STL chess pieces spawn in the world
- the overhead camera is bridged into ROS
- MoveIt plans the arm trajectory
- a ROS-side `FollowJointTrajectory` shim executes those plans through Gazebo joint position controllers
- `demo_turn` succeeds end to end for a one-robot engine-driven move

## Demo One Turn

Request a move from the configured engine and execute it through the action server:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator demo_turn --think-time 0.25
```

This currently executes a full home -> move -> return-home cycle for the white-side Panda in `ros_gz`.

## Coordinator Demo

Run the shared turn coordinator without the robots to exercise the FEN-in/UCI-out player layer:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator game_coordinator --max-plies 4 --white-backend stockfish --black-backend dqn --think-time 0.01
```

Today, the DQN backend uses the new RL adapter and falls back to a deterministic heuristic if no trained checkpoint is supplied.

## Engine Configuration

Runtime parameters:

- `engine_backend`
- `engine_executable`
- `engine_think_time_sec`
- `sim_backend`
- `square_size_m`
- `board_origin`
- `hover_height_m`
- `pickup_height_m`
- `move_time_sec`
- `execution_timeout_sec`

Default engine path in `config/digital_twin.yaml` is `/usr/games/stockfish`. Override it if your binary lives elsewhere:

```bash
ros2 launch chess_manipulator bringup.launch.py \
  sim_backend:=isaac \
  engine_executable:=/tmp/Stockfish/src/stockfish
```

You can also set engine environment variables such as `CHESS_MANIPULATOR_STOCKFISH`.

## Benchmarks

The benchmark suite covers both engine and system metrics:

- engine legality across a fixed FEN suite
- mean and median move latency
- bestmove reproducibility on repeated fixed-FEN runs
- planner success rate
- invalid IK rejection count
- collision-envelope rejection count
- mean stages per move
- estimated end-to-end latency for a canonical opening sequence

Run the suite:

```bash
/usr/bin/python3 /home/flux/Desktop/chessPlayer/repo/scripts/benchmark_engines.py \
  --suite /home/flux/Desktop/chessPlayer/repo/config/benchmark_suite.yaml \
  --stockfish /tmp/Stockfish/src/stockfish \
  --sunfish /tmp/sunfish/sunfish.py \
  --json-out /home/flux/Desktop/chessPlayer/repo/results/benchmark_results.json \
  --csv-out /home/flux/Desktop/chessPlayer/repo/results/benchmark_results.csv
```

### Current Example Results

| Benchmark | Value |
|---|---:|
| Stockfish legal moves | 4 / 4 |
| Stockfish mean latency | 50 ms |
| Sunfish legal moves | 4 / 4 |
| Sunfish mean latency | 39 ms |

Interpretation:
Sunfish responds slightly faster in the small fixed-time microbenchmark, but Stockfish remains the stronger baseline and default engine for the project.

## RL Scaffold

The repo now includes a bounded DQN scaffold for separate training and runtime inference.

Inference:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator rl_infer --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
```

Training:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator rl_train --checkpoint /home/flux/Desktop/chessPlayer/repo/results/rl_dqn.pt --episodes 50 --opponent heuristic --device cpu
```

PyTorch is intentionally treated as an optional training dependency for now.

## Simulator Notes

The active simulator migration target is ROS GZ. The detailed roadmap lives in [docs/ros_gz_dual_player_demo_plan.md](/home/flux/Desktop/chessPlayer/repo/docs/ros_gz_dual_player_demo_plan.md).

The older Isaac-specific docs remain in [docs/isaac_setup.md](/home/flux/Desktop/chessPlayer/repo/docs/isaac_setup.md) as reference material while the repo transitions away from the old simulator path.

## Assets

Existing media lives in `media/`. For final portfolio capture, add:

- one short ROS GZ demo clip
- one ROS GZ simulator screenshot
- one terminal or RViz screenshot showing ROS execution

## Scope Notes

- ROS GZ is the primary showcase and development path.
- The current execution confirmation is driven by Gazebo joint-state convergence, not piece-contact verification.
- Perception is currently a camera + occupancy/debug scaffold, not yet full piece-type recognition.
- Real-robot deployment, advanced grasp physics, and Franka safety certification remain out of scope for this portfolio version.
