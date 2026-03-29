# Demo Walkthrough

## Build

```bash
source /opt/ros/humble/setup.bash
env PATH=/usr/bin:/bin:$PATH colcon --log-base log_sys build \
  --build-base build_sys \
  --install-base install_sys \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## Launch Isaac-First Bringup

Start the visible Isaac GUI app in one terminal:

```bash
./scripts/run_isaac_demo.sh
```

Then start ROS bringup in another terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ROS_LOG_DIR=/tmp/ros_logs ros2 launch chess_manipulator bringup.launch.py sim_backend:=isaac
```

## Request And Execute One Move

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator demo_turn --think-time 0.25
```

## Run Benchmarks

```bash
/usr/bin/python3 /home/flux/Desktop/chessPlayer/repo/scripts/benchmark_engines.py \
  --suite /home/flux/Desktop/chessPlayer/repo/config/benchmark_suite.yaml \
  --stockfish /tmp/Stockfish/src/stockfish \
  --sunfish /tmp/sunfish/sunfish.py \
  --json-out /home/flux/Desktop/chessPlayer/repo/results/benchmark_results.json \
  --csv-out /home/flux/Desktop/chessPlayer/repo/results/benchmark_results.csv
```

## Expected Reviewer Outcome

- The workspace builds from a clean shell with system Python
- The custom ROS interfaces are discoverable
- The Isaac GUI app opens a Franka Panda with a procedural chessboard
- The bringup launches the manager, the simulator bridge for the selected backend, and robot state publisher
- A single engine-selected move completes and updates the published FEN
- The benchmark script emits engine and system metrics in machine-readable files
