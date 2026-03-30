# Isaac Sim Pivot Plan

## Locked Decisions

- Robot first: Franka Panda
- Isaac version: 4.2.0
- Install style: native local install in the workspace
- Default local path: `third_party/isaac-sim-4.2.0`
- AR4 switch: only if Panda-on-Isaac becomes the real blocker

## Isaac Version And Install Contract

This branch targets **Isaac Sim 4.2.0** to match the Ekumen AR4 reference stack.

Recommended native install location:

```text
${WORKSPACE_ROOT}/third_party/isaac-sim-4.2.0
```

Rationale:

- minimizes API drift against the reference integration
- keeps the simulator install local to this workspace instead of hidden in a home-directory path
- makes the branch reproducible for other users

The repo launch scripts now validate that install path and version by default.

## Goal

Pivot the active demo path from `ros_gz` / Gazebo Harmonic to NVIDIA Isaac Sim while preserving as much of the existing stack as possible:

- perception pipeline
- FEN-based game state
- Stockfish and DQN high-level reasoners
- MoveIt 2 planning
- robot executors and coordinator contracts

The end goal remains the same:

- two robots on opposite sides of a chessboard
- perception-driven board state in FEN
- white driven by a chess engine
- black driven by a DQN-based RL player
- alternating MoveIt-executed turns
- perception confirmation after each move

## External References Used

- Ekumen: Taking the AR4 further, from Gazebo to Isaac Sim
  - https://ekumenlabs.com/blog/posts/taking-the-ar4-further-from-gazebo-to-isaac-sim/
- Ekumen: AR4 Robot vs. the Chessboard, a Simulation in Isaac Sim
  - https://ekumenlabs.com/blog/posts/ar4-robot-vs-the-chessboard-a-simulation-in-isaac-sim/
- Ekumen AR4 repository
  - https://github.com/Ekumen-OS/ar4

## Main Technical Takeaway From Ekumen

The most important pattern from the AR4 work is not the specific robot model. It is the backend split:

- keep MoveIt and ROS control contracts on the ROS side
- introduce a dedicated Isaac simulation package
- use Isaac Sim to publish `/joint_states` and receive joint commands
- connect the simulated articulation through OmniGraph / Isaac ROS bridge
- keep higher-level planning and execution clients unchanged

In the AR4 repository this shows up as:

- a dedicated `ar4_isaac` package
- a simulation launcher that starts Isaac via `python.sh`
- a USD / USDA asset for the robot scene
- `topic_based_ros2_control/TopicBasedSystem` to preserve a ROS control-facing interface
- separate MoveIt config that remains ROS-native

That architecture maps well onto our current system.

## Decision

### Primary path: keep Panda, switch simulator

We should keep the Franka Panda as the primary robot for the pivot branch.

Why:

- our current MoveIt integration, joint naming, executor logic, and planner assumptions are already Panda-oriented
- the chess manipulation stack is already coupled to Panda joint-space behavior
- replacing the robot and the simulator at the same time would raise risk significantly
- the user explicitly wants the same perception, MoveIt, and high-level systems to stay intact

### Fallback path: switch to AR4 only if Panda Isaac import blocks us

We will keep AR4 as a fallback option if:

- Panda import into Isaac proves unstable
- ROS control bridging for Panda in Isaac becomes a hard blocker
- dual-robot staging is substantially easier with AR4 assets than with Panda assets

If that happens, we can preserve:

- FEN / perception stack
- coordinator
- engine adapter
- DQN adapter
- most of the chess move semantics

but we would need to re-target:

- MoveIt config
- robot executor joint naming
- grasp poses and board-relative kinematics

So AR4 is the fallback, not the default.

## What We Reuse Unchanged

These components should stay conceptually unchanged in the Isaac pivot:

- `repo/chess_manipulator/chess`
- `repo/chess_manipulator/players`
- `repo/chess_manipulator/rl`
- most of `repo/chess_manipulator/coordinator`
- perception contracts:
  - `/perception/observed_fen`
  - `/perception/fen_confidence`
  - `/perception/debug/annotated_image`
- execution contracts:
  - `ExecuteChessMove.action`
  - `GetBestMove.srv`

These components will need backend adaptation, not redesign:

- `repo/chess_manipulator/nodes/robot_executor.py`
- `repo/chess_manipulator/motion/moveit_planner.py`
- `repo/chess_manipulator/nodes/chess_manager.py`
- `repo/chess_manipulator/coordinator/game_coordinator.py`

These components become legacy after the pivot:

- `repo/launch/ros_gz.launch.py`
- `repo/chess_manipulator/nodes/ros_gz_trajectory_controller.py`
- `repo/worlds/chesset_ros_gz.sdf`

## Target Isaac Architecture

Add a new Isaac-focused simulation package and scene layer inside the current repo:

- `repo/isaac_sim/`
  - launchers
  - simulation bootstrap
  - stage assets
  - robot scene authoring utilities
  - OmniGraph / ROS bridge config

Target runtime structure:

1. Isaac Sim launches a saved stage with:
   - one chessboard
   - two Panda robots
   - one shared overhead camera
   - all chess pieces
   - graveyard / capture zones
2. Isaac publishes:
   - `/joint_states`
   - camera image / camera info
   - optional segmentation streams
   - execution completion / contact state topics
3. ROS side keeps:
   - MoveIt 2
   - robot executors
   - perception nodes
   - game coordinator
   - Stockfish adapter
   - DQN adapter

## Sprint Plan

### Sprint 0: Create the pivot branch and freeze the Gazebo baseline

Deliverables:

- create `pivot/isaac-sim-panda-first`
- preserve current `ros_gz` state as reference
- write this migration plan into the branch
- mark `ros_gz` as legacy path for this branch

Definition of done:

- branch exists remotely
- baseline state is documented

### Sprint 1: Single-robot Panda in Isaac Sim

Deliverables:

- add a new Isaac simulation package for Panda
- create a first Panda Isaac launch similar in spirit to Ekumen’s `ar4_isaac`
- load a Panda stage in Isaac via `python.sh`
- enable ROS 2 bridge
- publish `/joint_states`
- receive joint commands from ROS

Implementation strategy:

- use Ekumen’s `run_sim.py` pattern as the template
- keep MoveIt side ROS-native
- decide whether to bridge:
  - raw joint commands, or
  - a `topic_based_ros2_control`-style control interface

Definition of done:

- Isaac GUI launches a Panda scene
- the Panda is visible
- ROS sees live `/joint_states`

### Sprint 2: Replace the current sim backend with Isaac for one robot

Deliverables:

- add `sim_backend:=isaac` as the primary backend in this branch
- replace the current ad hoc Isaac bridge with a real articulation control path
- make `demo_turn` work through Isaac instead of `ros_gz`
- keep the action/result contract identical from the coordinator perspective

Definition of done:

- `demo_turn` succeeds with Panda in Isaac Sim
- visible motion occurs in the GUI
- MoveIt plans and executes against Isaac-fed joint states

### Sprint 3: Chess scene in Isaac

Deliverables:

- add the chessboard and STL-based piece set to the Isaac scene
- create piece prim naming that matches stable chess-square identifiers
- add a shared overhead camera
- add graveyard capture zones

Manual work expected:

- stage authoring and visual inspection inside Isaac GUI
- possibly one-time USD scene save / refinement

Definition of done:

- board, pieces, and Panda are visible in Isaac
- camera view contains the whole board

### Sprint 4: Perception migration without redesign

Deliverables:

- point current perception nodes at Isaac camera topics
- preserve FEN output contracts
- add segmentation-assisted inputs if Isaac makes them easy
- upgrade from occupancy-only inference toward piece-type recognition

Planned upgrade path:

1. keep the current occupancy/legal-transition pipeline working
2. add per-piece visual identity inference
3. build full piece-type-to-FEN reconstruction

Definition of done:

- starting position produces correct perceived FEN
- a single executed move updates the perceived FEN correctly in Isaac

### Sprint 5: Dual-robot Isaac scene

Deliverables:

- mirror a second Panda on the opposite side of the board
- namespace both executors:
  - `/white/execute_move`
  - `/black/execute_move`
- create dual robot home poses and collision-safe staging

Definition of done:

- both robots exist in the same Isaac scene
- each can independently execute a test move

### Sprint 6: Coordinator integration for real alternating play

Deliverables:

- connect the coordinator to both robot executors
- make the coordinator use perception-confirmed FEN only
- alternate between:
  - white engine adapter
  - black DQN adapter
- add execution verification and turn switching

Definition of done:

- one short alternating match can run through the coordinator

### Sprint 7: Grasp, attach/detach, captures, and promotions

Deliverables:

- physical pick-up and placement in Isaac
- attach / detach logic tied to gripper state
- capture handling to graveyard zones
- promotion handling
- castling handling

Definition of done:

- non-trivial chess moves work in scene, not only quiet pawn pushes

### Sprint 8: DQN maturity and demo hardening

Deliverables:

- train a meaningful DQN checkpoint outside the simulator loop
- remove heuristic fallback for demo runs
- add operator scripts for:
  - launching Isaac demo
  - launching full ROS stack
  - resetting scene
  - running a short engine-vs-DQN match
- add metrics and logging for demo verification

Definition of done:

- 6 to 8 plies complete reliably in Isaac
- engine vs DQN demo is recordable without manual recovery

## Missing Pieces To Fill

These are the currently known gaps and how they map to the new Isaac plan.

### Two physical robots are not wired yet

Plan:

- Sprint 5 introduces mirrored dual Panda staging in Isaac
- create per-robot namespaces, home poses, planning groups, and executors

### DQN is scaffolded but not meaningfully trained

Plan:

- Sprint 8 adds actual training output and checkpoint selection
- the demo branch should not rely on heuristic fallback once the demo is declared complete

### Perception is not full piece-type vision yet

Plan:

- keep current occupancy/legal-move inference as stage one
- add piece-type recognition using Isaac camera imagery and optional segmentation
- output only perceived FEN to the coordinator

### Coordinator is not yet connected to dual robot execution

Plan:

- Sprint 6 wires the coordinator into both namespaced executors
- the board is advanced only after perception confirmation

### Grasp / attach / detach / captures are still missing

Plan:

- Sprint 7 makes physical manipulation first-class in Isaac
- hybrid attach/detach is acceptable for demo reliability

## Manual User Inputs We Will Need

These are the points where manual help from the user will likely be required.

1. Isaac Sim environment confirmation
   - confirm installed Isaac Sim version
   - confirm preferred install path
   - confirm whether GUI use is local or containerized
2. One-time stage authoring and validation
   - if Panda USD import requires Isaac GUI adjustment
   - if chessboard scene placement needs hand-tuning
3. Visual review checkpoints
   - approve final camera angle
   - approve dual-robot placement
   - approve board/piece appearance for demo recording
4. DQN checkpoint acceptance
   - select the trained model checkpoint intended for the demo

## Immediate Next Implementation Steps

1. Add a new Isaac simulation package for Panda in this repo.
2. Reuse the Ekumen launch pattern:
   - launch Isaac with `python.sh`
   - load a prepared stage / USD
   - enable ROS 2 bridge
3. Keep MoveIt as-is on the ROS side and replace only the backend execution path.
4. Make `demo_turn` succeed in Isaac before touching dual-robot logic.
5. Once single-robot Isaac works, move to the dual-robot scene and coordinator wiring.

## Success Criteria

This pivot is complete when:

- Isaac Sim replaces `ros_gz` as the active demo backend
- Panda remains the primary robot unless blocked
- perception publishes board state as FEN from Isaac camera input
- MoveIt executes real robot motion through Isaac
- the coordinator alternates engine and DQN turns
- both robots play 6 to 8 plies on one shared board
- the demo is visually recordable in Isaac GUI
