# Learned-Player Demo Workflow

This is the operator-facing workflow for the final short or full `engine vs learned player` demo.

## What Counts As The Learned Player Artifact

The runtime demo expects an offline-trained checkpoint produced by:

```bash
source /opt/ros/humble/setup.bash
source /home/flux/Desktop/chessPlayer/install_sys/setup.bash
ros2 run chess_manipulator rl_train \
  --checkpoint /home/flux/Desktop/chessPlayer/repo/results/training/black_dqn.pt \
  --episodes 50 \
  --opponent heuristic \
  --device cpu
```

That training run writes:

- a checkpoint, for example `black_dqn.pt`
- a metadata sidecar beside it, for example `black_dqn.pt.metadata.json`

The demo does not point directly at arbitrary training output paths. Instead, promote the selected artifact into the stable demo location.

If you already have accumulated match logs under `repo/results/game_logs`, the fastest operator loop is the bundled offline-learning cycle:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/run_offline_learning_cycle.sh
```

Useful overrides:

```bash
GAME_LOG_DIR=./repo/results/game_logs \
OFFLINE_OPTIMIZATION_STEPS=500 \
SELF_PLAY_EPISODES=10 \
./repo/scripts/run_offline_learning_cycle.sh
```

## Baseline And Campaign Flow

Train and promote a baseline checkpoint:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/train_baseline_learned_player.sh
```

Run the long campaign with milestone videos at games `1, 50, 100, ...` and offline updates every `50` games:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/run_learning_campaign.sh
```

Useful overrides:

```bash
MATCH_FORMAT=short TOTAL_GAMES=200 START_GAME=1 BATCH_SIZE=50 THINK_TIME_SEC=0.1 ./repo/scripts/run_learning_campaign.sh
```

```bash
CAPTURE_MILESTONES=1 VIDEO_DISPLAY=:0.0 VIDEO_SIZE=1920x1080 VIDEO_FPS=30 ./repo/scripts/run_learning_campaign.sh
```

Full-game batches:

```bash
MATCH_FORMAT=full TOTAL_GAMES=50 START_GAME=1 BATCH_SIZE=10 THINK_TIME_SEC=0.1 ./repo/scripts/run_learning_campaign.sh
```

Long-run 2000-game full-game campaign:

```bash
CAMPAIGN_ID=learned_2000_full \
MATCH_FORMAT=full \
TOTAL_GAMES=2000 \
BATCH_SIZE=50 \
THINK_TIME_SEC=0.1 \
CAPTURE_MILESTONES=1 \
./repo/scripts/run_learning_campaign.sh
```

## Promote The Demo Artifact

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/promote_learned_checkpoint.sh \
  /home/flux/Desktop/chessPlayer/repo/results/training/black_dqn.pt
```

This copies the artifact into:

- checkpoint: `/home/flux/Desktop/chessPlayer/repo/results/demo/learned_player/black_dqn.pt`
- metadata: `/home/flux/Desktop/chessPlayer/repo/results/demo/learned_player/black_dqn.pt.metadata.json` if available
- manifest: `/home/flux/Desktop/chessPlayer/repo/results/demo/learned_player/PROMOTED_FROM.txt`

That promoted checkpoint is the default runtime input for the learned-player demo launcher.

## Run The Match

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/run_engine_vs_learned_demo.sh
```

Defaults:

- simulator: Isaac Sim
- white: `stockfish`
- black: `dqn`
- checkpoint: promoted artifact under `results/demo/learned_player/`
- match format: `short`
- short-format plies: `8`

Useful overrides:

```bash
LEARNED_CHECKPOINT=/path/to/other_black_dqn.pt ./repo/scripts/run_engine_vs_learned_demo.sh
```

```bash
ISAAC_HEADLESS=true ./repo/scripts/run_engine_vs_learned_demo.sh
```

```bash
WHITE_ENGINE_EXECUTABLE=/usr/games/stockfish ./repo/scripts/run_engine_vs_learned_demo.sh
```

Run one full game to completion:

```bash
MATCH_FORMAT=full ./repo/scripts/run_engine_vs_learned_demo.sh
```

## Reset State

There is no in-place scene reset service documented for the final operator path yet. The supported reset is a full process and scene restart:

```bash
cd /home/flux/Desktop/chessPlayer
./repo/scripts/reset_demo_state.sh
```

That script:

- stops bringup, coordinator, perception, and simulator-side launcher processes
- clears temporary ROS log/home state under `/tmp`
- prints the relaunch command for the learned-player demo

## Intentional Safety Guard

`run_engine_vs_learned_demo.sh` refuses to start a DQN-backed demo if no promoted checkpoint is present. This is intentional: it prevents the final demo workflow from silently falling back to the heuristic path.

If you explicitly want the heuristic fallback for development only:

```bash
ALLOW_HEURISTIC_FALLBACK=1 ./repo/scripts/run_engine_vs_learned_demo.sh
```
