# Franka Panda Chess Arm Digital Twin (Isaac Sim)

Digital twin of a **Franka Emika Panda** robotic arm playing chess in **NVIDIA Isaac Sim**.  
The simulation exposes a ROS-based interface and a chess-engine (FEN) interface and is structured to be deployable to a future real robot arm.

---

## Features

- **Isaac Sim Digital Twin**
  - Franka Panda robot modeled via Isaac Sim (`omni.isaac.franka` / URDF import).
  - Joint limits, kinematics, and gripper actuation exposed through a Python API.
  - Configurable physics and simulation parameters (`config/sim_params.yaml`).

- **Trajectory Generation & Motion Planning**
  - Python control stack for joint-space trajectory generation.
  - Simple motion planner in joint space (straight-line with collision hooks).
  - Smooth, time-parameterized joint trajectories for pick-and-place of chess pieces.

- **Chess Engine Interface (FEN-based)**
  - Uses `python-chess` for FEN parsing and move handling.
  - Maps engine moves (UCI / algebraic) to end-effector goals on a calibrated board.
  - Board calibration and camera-to-world transforms stored in `config/board_config.yaml`.

- **Simulated Perception**
  - Virtual camera configuration in `config/camera_config.yaml`.
  - Synthetic piece pose estimation from FEN.
  - Simple calibration refinement that estimates a translation offset for the board.

- **ROS2 Interface**
  - ROS2 bridge for:
    - Publishing Franka joint states.
    - Accepting joint trajectory commands.
  - Treats the Isaac Sim Franka as a software-compatible target for a future physical Panda.

---

## Repository Structure

```text
chess_arm/
  sim/
    digital_twin.py        # FrankaChessDigitalTwin wrapper around Isaac Sim Franka
    board_env.py           # Board + pieces environment and asset loading
  control/
    ik.py                  # Lula-based IK wrapper for end-effector poses
    trajectory_generator.py# JointTrajectory dataclass + time-scaling
    motion_planner.py      # Simple joint-space interpolation + collision hook
    controllers.py         # JointTrajectoryExecutor for Isaac Sim articulation actions
  chess/
    engine_interface.py    # FEN + UCI engine wrapper (python-chess)
    move_mapping.py        # Move → board waypoints and square name helpers
  perception/
    vision_sim.py          # Simulated perception + calibration refinement
    board_tracker.py       # Tracks FEN and latest perceived piece poses
  ros/
    ros_bridge.py          # ROS2 node: joint states + JointTrajectory control
  utils/
    transforms.py          # BoardCalibration + square <-> world transforms
    logging_utils.py       # Logging setup

config/
  sim_params.yaml          # Physics and prim paths
  board_config.yaml        # Board origin/axes/square size
  camera_config.yaml       # Virtual camera pose and intrinsics
  ros_topics.yaml          # Joint names and ROS topic names

assets/
  urdf/                    # URDF description(s) for Franka
  usd/                     # USD assets for Franka, board and pieces

scripts/
  launch_isaac_chess_env.py    # Minimal smoke test for sim + digital twin
  spawn_franka_with_board.py   # Spawn Franka + board + pieces only
  run_single_move_demo.py      # Execute one UCI move end-to-end in sim
  run_chess_match_demo.py      # Skeleton for engine-driven play
  calibrate_board_from_sim.py  # Board calibration refinement using simulated perception
  ros_bridge_main.py           # Isaac Sim + ROS2 integrated loop

ros/
  package.xml                  # ROS2 package metadata
  CMakeLists.txt               # ROS2 build file
  launch/                      # ROS2 launch files for the bridge

tests/
  test_transforms.py
  test_move_mapping.py
  test_trajectory_generator.py
```

---

## Requirements

- Python **3.10+**
- NVIDIA Isaac Sim installed and working
- ROS2 (e.g. Humble / Foxy) for the ROS bridge (optional, but expected)
- A UCI chess engine binary (e.g. Stockfish) for engine-based play

Python packages (also in `pyproject.toml` / `requirements.txt`):

- `numpy`
- `scipy`
- `pyyaml`
- `transforms3d`
- `python-chess`

For local testing:

- `pytest` (not listed in `requirements.txt`; install as a dev dependency).

---

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest                  # optional, for tests
```

### 2. Launch minimal Isaac Sim environment

Run this script inside Isaac Sim’s Python environment (for example using `python.sh`):

```bash
./python.sh scripts/launch_isaac_chess_env.py
```

This starts Isaac Sim, creates a world, spawns a board environment and the Franka digital twin, and steps the simulation.

### 3. Run a single chess move

```bash
./python.sh scripts/run_single_move_demo.py --move e2e4
```

The script:

- Initializes the digital twin and board.
- Parses the UCI move (default `e2e4`).
- Maps the move to board-space waypoints.
- Uses IK (when available) to compute joint-space waypoints.
- Generates a time-parameterized joint trajectory and executes it.

### 4. ROS2 bridge

Run the integrated Isaac Sim + ROS2 bridge:

```bash
./python.sh scripts/ros_bridge_main.py
```

The bridge:

- Publishes joint states on `/franka_chess/joint_states`.
- Subscribes to joint trajectories on `/franka_chess/joint_trajectory`.
- Converts incoming `trajectory_msgs/JointTrajectory` into the internal `JointTrajectory` and executes it on the digital twin.

The topic names and joint names are configured in `config/ros_topics.yaml`.

---

## Perception and Board Calibration

The perception stack is designed for simulated sensing and calibration:

- `SimulatedChessPerception`:
  - Builds a camera from `config/camera_config.yaml`.
  - Generates synthetic `PiecePose` objects for a given FEN (standard initial position by default).
  - Estimates a translation-only correction to the board origin.

- `BoardTracker`:
  - Stores a FEN string and a map of latest perceived piece poses.
  - Includes placeholders for future FEN reconstruction from perception.

`calibrate_board_from_sim.py` ties these together:

- Runs the sim for a short period.
- Calls `estimate_piece_poses`.
- Computes a translation offset for `origin_world`.
- Writes a `board_config_calibrated.yaml` file.

---

## Architecture Overview

High-level data flow for a single move:

1. **Chess engine / FEN**
   - `ChessEngineInterface` maintains the current board state and connects to a UCI engine.
   - Engine produces a move in UCI format (e.g. `e2e4`).

2. **Move → board waypoints**
   - `move_mapping` converts the move into `(from_square, to_square)` algebraic names.
   - `make_move_waypoints` uses `BoardCalibration` to compute approach, grasp, lift, place and retreat positions in world coordinates.

3. **Board waypoints → joint trajectory**
   - `FrankaChessDigitalTwin` uses `IKSolver` (Lula-based) to map each board-space waypoint to a joint configuration.
   - `trajectory_generator` builds a time-parameterized `JointTrajectory`.

4. **Execution**
   - `JointTrajectoryExecutor` samples the trajectory each sim step and produces `ArticulationAction` commands for the Panda in Isaac Sim.
   - `FrankaChessDigitalTwin.step` applies these actions.

5. **ROS2 integration (optional)**
   - `FrankaRosBridge` publishes joint states and accepts `JointTrajectory` messages.
   - External planners or controllers can treat the Isaac Sim twin as a ROS-controlled robot.

---

## From Simulation to a Real Franka Panda

The project is structured to make transition to a physical Panda straightforward:

- **Shared URDF / semantics**
  - `assets/urdf/franka_panda.urdf` can be shared between Isaac Sim and the real robot’s control stack.
  - Joint ordering and naming are kept consistent with ROS2 configuration in `config/ros_topics.yaml`.

- **ROS interface compatibility**
  - The ROS2 bridge exposes standard `sensor_msgs/JointState` and `trajectory_msgs/JointTrajectory`.
  - A controller running on the real robot can subscribe to the same topics or a compatible namespace.

- **Calibration**
  - `BoardCalibration` and the camera configuration mirror real-world calibration concepts:
    - Board origin in world coordinates.
    - Board axes and square size.
    - Camera pose and intrinsics.
  - The same structures can be populated from offline calibration routines in a lab.

The final mapping to a real robot primarily requires:

- A real ROS2 driver for the Panda that accepts `JointTrajectory`.
- A calibration procedure that populates the board and camera configs from measurements rather than simulation.
- Safety layers and collision checking suitable for hardware operation.

---

## Tests

Unit tests focus on pure-python logic:

- `tests/test_transforms.py`
  - `square_to_indices` / `indices_to_square` round-trips.
  - `BoardCalibration` orthonormal axes.
  - `square_center_world` on a simple board.

- `tests/test_move_mapping.py`
  - Move name conversion (`move_to_square_names`, `uci_to_square_names`).
  - Waypoints differ between source and destination squares.

- `tests/test_trajectory_generator.py`
  - Monotonic trajectory time stamps.
  - Endpoint positions match input waypoints.
  - Sampling behavior at boundaries and midpoints.

Run tests with:

```bash
pytest
```

---

## License

MIT – see [LICENSE](LICENSE).
