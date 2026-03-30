#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

EPISODES="${EPISODES:-200}"
OPPONENT="${OPPONENT:-heuristic}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-7}"
TRAIN_DIR="${TRAIN_DIR:-${REPO_ROOT}/results/training/baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_DIR}/black_dqn.pt}"

mkdir -p "${TRAIN_DIR}"

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

printf 'Training baseline learned-player checkpoint.\n'
printf '  episodes:   %s\n' "${EPISODES}"
printf '  opponent:   %s\n' "${OPPONENT}"
printf '  device:     %s\n' "${DEVICE}"
printf '  checkpoint: %s\n' "${CHECKPOINT_PATH}"

ros2 run chess_manipulator rl_train \
  --checkpoint "${CHECKPOINT_PATH}" \
  --episodes "${EPISODES}" \
  --opponent "${OPPONENT}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

"${REPO_ROOT}/scripts/promote_learned_checkpoint.sh" "${CHECKPOINT_PATH}"

printf 'Baseline checkpoint is now promoted for demo use.\n'
