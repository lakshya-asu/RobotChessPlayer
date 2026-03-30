# ROS GZ Dual-Player Chess Demo Plan

## Purpose

This document is the completion plan for turning the current `chess_manipulator` workspace into a demo-ready `ros_gz` chess system with:

- a physical chess scene in `ros_gz`
- camera-based board perception
- MoveIt 2 IK and motion planning
- two robots on opposite sides of the board
- one classical chess-engine player
- one DQN-based RL player trained separately and deployed only for inference
- a final demo flow of `chess engine vs reinforcement learning agent`

This plan assumes ROS 2 Humble, `ros_gz` as the supported simulator path, and the STL chess piece assets in `/home/flux/Desktop/chessPlayer/STL`.

## Current Baseline

The current repo already provides useful building blocks, but it is not yet aligned with the final target:

- the main ROS package is in `repo/`
- custom interfaces are in `chess_manipulator_msgs/`
- the current simulator path is still centered on Isaac and Gazebo Classic shims
- the current board state is primarily logical and `python-chess` driven
- the current IK is the approximate solver in `chess_manipulator/motion/kinematics.py`
- there is no real camera perception pipeline
- there is no MoveIt 2 stack
- there is no second robot or RL inference path

## Final Demo Definition

The project is complete only when the following end-to-end demo is reliable:

1. Launch a `ros_gz` world containing:
   - two Franka Panda arms on opposite sides of the board
   - one chessboard
   - 32 physical chess pieces built from the provided STL meshes
   - one overhead camera for board perception
2. Start ROS bringup, perception, MoveIt 2, and the game coordinator.
3. Perception observes the real board image and publishes the board state as FEN.
4. The white-side player uses a chess engine to choose a move.
5. The white robot plans and executes the move through MoveIt 2.
6. Perception verifies the resulting board state.
7. The black-side player uses a DQN-based RL model to choose a move.
8. The black robot plans and executes the move through MoveIt 2.
9. Perception verifies the resulting board state again.
10. The system can run a short engine-vs-RL sequence and be recorded as a clean demo video.

## Architectural End State

### Simulator

- `ros_gz_sim` is the primary and supported simulator.
- Gazebo Classic support is retired from the main demo path.
- The world contains:
  - environment geometry
  - both robots
  - physical piece models
  - camera sensor

### Board State and Game Logic

- one `GameCoordinator` owns the single canonical board state
- perception proposes observed board state
- legality is enforced centrally against `python-chess`
- board state changes are accepted only after:
  - a legal move decision
  - successful execution
  - perception confirmation

### Motion and IK

- MoveIt 2 replaces the current approximate IK path
- motion is generated through staged manipulation goals:
  - pre-grasp
  - descend
  - grasp
  - lift
  - transit
  - place
  - retreat
- collision checking uses:
  - table
  - board
  - non-target pieces
  - robot geometry

### High-Level Reasoners

- white side:
  - Stockfish or compatible engine through the existing engine adapter pattern
- black side:
  - DQN inference adapter
  - no online training in the runtime workspace
  - training is separate and offline

## Multi-Agent Execution Plan

This plan is intentionally designed for parallel work. The recommended agent topology is below.

### Agent 1: ROS GZ Platform Migration

Ownership:

- `package.xml`
- launch files under `launch/`
- simulator dependency model
- `ros_gz` world startup and bridges

Responsibilities:

- remove the main dependency on classic `gazebo_ros`
- add the supported `ros_gz_sim` launch path
- launch the world, clock bridge, and robot-related bridges
- make `ros_gz` the default simulator path in docs and scripts

Primary outputs:

- new `ros_gz` launch file
- updated bringup path
- updated dependency declarations

### Agent 2: MoveIt 2 and Robot Execution

Ownership:

- Panda MoveIt config package
- control mapping
- MoveIt execution integration
- replacement of the approximate IK path

Responsibilities:

- add MoveIt 2 configuration for the Panda and gripper
- align controllers with MoveIt execution
- replace the current direct trajectory backend with MoveIt planning and execution
- expose execution success/failure back to the coordinator

Primary outputs:

- MoveIt config package
- MoveIt-enabled launch path
- execution adapter replacing simulator-specific fake completion

### Agent 3: Physical Chess World and STL Piece Models

Ownership:

- world files under `worlds/`
- piece models generated from `/home/flux/Desktop/chessPlayer/STL`
- physical material and inertial settings

Responsibilities:

- package STL assets as reusable sim models
- create white and black variants
- use mesh visuals with simplified collision shapes for stable physics
- add capture zones / graveyard positions
- ensure consistent model naming so perception and execution can reason about pieces

Primary outputs:

- model directories for pieces
- updated `ros_gz` world
- stable piece spawn strategy

### Agent 4: Camera Perception Pipeline

Ownership:

- new perception package
- camera launch and image bridging
- board-state inference

Responsibilities:

- add an overhead camera in the world
- bridge image and camera info into ROS
- rectify the board view
- infer square-level occupancy and piece identity
- publish FEN and perception confidence
- validate perceived state against legal game transitions

Primary outputs:

- `camera_preprocessor`
- `board_detector`
- `square_extractor`
- `piece_classifier`
- `fen_builder`
- `fen_validator`

### Agent 5: Dual-Player Game Coordinator

Ownership:

- current `chess_manager` responsibilities
- canonical board ownership
- turn orchestration
- player namespacing

Responsibilities:

- split the single-manager flow into:
  - one shared game coordinator
  - two player reasoner interfaces
  - two robot execution namespaces
- keep one canonical FEN authority
- dispatch turns correctly to white and black
- update board state only after execution plus perception verification

Primary outputs:

- new coordinator node
- namespaced execution and status interfaces
- clean engine-vs-RL game loop

### Agent 6: RL Training and Inference Integration

Ownership:

- DQN model interface
- training/inference separation
- RL move selection adapter

Responsibilities:

- define the offline training contract
- define a move-space encoder/decoder
- load a trained checkpoint for inference only
- mask illegal moves at inference time
- expose the DQN player through the same move-selection contract as the engine path

Primary outputs:

- RL inference adapter in the runtime workspace
- documented offline training workflow
- engine-vs-RL match configuration

### Agent 7: Demo Hardening and Reviewer Experience

Ownership:

- run scripts
- benchmarks
- docs
- demo recording flow

Responsibilities:

- add one-command startup helpers
- add acceptance and smoke tests
- define metrics for perception, planning, execution, and match flow
- update README and docs for reviewer reproducibility
- produce a final demo checklist and expected artifacts

Primary outputs:

- operator scripts
- benchmark/report outputs
- final documentation
- demo checklist

## Phase Plan

### Phase 0: Environment Stabilization

Goal:

- make the workspace consistently build and run against the installed `ros_gz` stack

Tasks:

- remove classic Gazebo assumptions from the active launch path
- make sure runtime scripts source ROS 2 and the local workspace cleanly
- ensure all new dependencies are documented and installable
- normalize package naming and supported simulator documentation

Acceptance:

- clean workspace build from a documented command
- `ros_gz_sim` packages resolved at runtime

### Phase 1: ROS GZ Migration

Goal:

- make `ros_gz` the primary simulator path

Tasks:

- add a new `ros_gz` launch file
- update `bringup.launch.py` to target `ros_gz` as the default simulator mode
- deprecate `gazebo_legacy.launch.py` from the main demo path
- bridge `/clock` and any required sensor/control topics

Acceptance:

- `ros2 launch ...` starts `ros_gz` successfully
- sim time is valid in ROS
- the world opens without falling back to classic Gazebo

### Phase 2: Physical World and STL Piece Integration

Goal:

- replace placeholder block pieces with physical piece models built from the STL set

Tasks:

- create sim-ready model packages for:
  - pawn
  - rook
  - knight
  - bishop
  - queen
  - king
- define visual mesh scale
- define simplified collision geometry
- define inertial, friction, and contact settings
- spawn all 32 pieces in standard starting squares

Acceptance:

- all 32 pieces appear correctly
- pieces rest stably on the board
- pieces do not jitter or explode under normal startup physics

### Phase 3: Camera and Perception

Goal:

- perceive the real board from camera images instead of relying only on internal state

Tasks:

- add a fixed overhead camera in the world
- bridge image topics through `ros_gz_bridge`
- rectify the board into a top-down crop
- split the board into 64 square regions
- classify each square
- build full FEN output
- validate inferred board state against legal transitions

Acceptance:

- starting FEN is recognized correctly
- simple moves update the perceived FEN correctly
- perception outputs confidence and validation status

### Phase 4: MoveIt 2 for a Single Arm

Goal:

- establish the real IK/planning path before scaling to two players

Tasks:

- add Panda MoveIt config
- align controllers and execution interfaces
- replace the approximate solver with MoveIt planning
- implement staged pick-and-place planning through MoveIt
- populate the planning scene with board and piece obstacles

Acceptance:

- one Panda can plan and execute a simple move
- MoveIt provides the active IK path
- board state changes only after successful execution

### Phase 5: Dual-Arm World

Goal:

- add the second robot and make both robots symmetry-aware

Tasks:

- spawn a second Panda opposite the first
- define mirrored base transforms
- namespace each robot’s:
  - joint states
  - controllers
  - MoveIt group
  - execution topics
- define per-side safe approach directions
- define per-side capture/drop zones

Acceptance:

- both robots exist in one world
- both robots can plan to board-relative targets
- neither robot interferes with the other’s namespace or planning scene updates

### Phase 6: Coordinator and Turn Logic

Goal:

- replace the current single-player orchestration with a true dual-player game coordinator

Tasks:

- split the current `chess_manager` role into:
  - `GameCoordinator`
  - white-side reasoner client
  - black-side reasoner client
  - white-side execution client
  - black-side execution client
- keep one canonical board authority
- add turn management and move history
- validate moves before dispatch

Acceptance:

- the coordinator alternates turns correctly
- white and black moves are dispatched to the proper robot
- a stale or duplicated execution result cannot corrupt game state

### Phase 7: RL Player Integration

Goal:

- add the second high-level player as a DQN inference path

Tasks:

- define the offline RL training contract
- define the runtime model loading path
- implement legal-move masking
- expose the RL player through the same move-selection interface as the engine player
- support engine-vs-RL play mode

Acceptance:

- the RL player can produce legal UCI moves from FEN
- the coordinator can alternate between engine and RL policies
- training remains outside the runtime workspace

### Phase 8: Perception-Verified Manipulation Loop

Goal:

- close the loop so camera input confirms every move

Tasks:

- before each turn, use perceived FEN as the primary observed board state
- after each move, require perception confirmation of the new board
- if perception disagrees, flag execution failure and stop the game

Acceptance:

- at least one engine move and one RL move are executed and verified through perception
- the board state never advances on unverified execution

### Phase 9: Demo Hardening

Goal:

- make the system reviewer-ready

Tasks:

- add startup scripts for:
  - sim-only
  - full stack
  - one-turn demo
  - short match demo
- add benchmarks for:
  - perception accuracy
  - FEN exact-match rate
  - MoveIt planning success rate
  - execution success rate
  - end-to-end turn latency
- update docs
- define recording flow and demo checklist

Acceptance:

- a new reviewer can build and run the demo from docs
- the system can produce a clean short demo clip
- the demo can run engine-vs-RL for multiple plies without desynchronization

## Interface Strategy

### Shared Game Interfaces

- `/chess/board_state`
- `/chess/current_turn`
- `/chess/game_status`
- `/chess/move_history`

### Perception Outputs

- `/perception/board_state_fen`
- `/perception/board_confidence`
- `/perception/validation_status`
- optional square-level debug topics

### Player Reasoner Interfaces

- `/players/white/get_move`
- `/players/black/get_move`

### Robot Execution Interfaces

- `/players/white/execute_move`
- `/players/black/execute_move`
- `/players/white/execution_feedback`
- `/players/black/execution_feedback`

### Internal Rule

- only the `GameCoordinator` owns canonical board truth
- perception is the observation layer
- reasoners propose moves
- executors move hardware/sim
- no second node is allowed to own independent game truth

## Demo-Ready Deliverables

The project is not complete until all of these exist:

- a working `ros_gz` simulator launch
- a dual-arm world
- physical STL-based chess pieces
- a camera-based perception stack
- MoveIt 2 planning and execution
- engine-vs-RL orchestration
- startup scripts for reviewers
- a final README and supporting docs
- benchmark outputs
- a short engine-vs-RL demo video

## Risks and Guardrails

### Risk: Perception brittleness

Guardrail:

- start with a fixed overhead camera and controlled lighting
- use board rectification and temporal smoothing

### Risk: Physics instability on piece grasping

Guardrail:

- use mesh visuals with simplified collision
- use contact-aware attach/detach for demo reliability if full grasp physics is unstable

### Risk: MoveIt integration complexity

Guardrail:

- validate one-arm MoveIt execution before adding the second robot

### Risk: RL player weakness

Guardrail:

- keep the DQN agent framed as an RL reasoner, not as a stronger chess engine
- separate training from runtime and require legal-move masking at inference

### Risk: Competing board authorities

Guardrail:

- one coordinator only
- one canonical FEN only
- perception and reasoners remain subordinate to the coordinator

## Completion Criteria

This plan is complete only when all of the following are true:

- `ros_gz` is the active simulator path
- the board and pieces are physical objects in sim
- the camera perception stack produces correct board state
- MoveIt 2 provides the active IK/planning path
- both robots can manipulate pieces from opposite sides of the board
- white uses a chess engine
- black uses a DQN inference model trained separately
- the coordinator runs an engine-vs-RL match safely
- perception confirms post-move state
- the system is stable enough for a recorded demo
