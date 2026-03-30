#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

GAME_LOG_DIR="${GAME_LOG_DIR:-${REPO_ROOT}/results/game_logs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/offline_learning/${RUN_ID}}"
SIDE_FILTER="${SIDE_FILTER:-black}"
OFFLINE_OPTIMIZATION_STEPS="${OFFLINE_OPTIMIZATION_STEPS:-250}"
SELF_PLAY_EPISODES="${SELF_PLAY_EPISODES:-0}"
POST_SELF_PLAY_OPTIMIZATION_STEPS="${POST_SELF_PLAY_OPTIMIZATION_STEPS:-50}"
MIN_LEGAL_RATE="${MIN_LEGAL_RATE:-1.0}"
MIN_AGREEMENT_RATE="${MIN_AGREEMENT_RATE:-0.0}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-${OUTPUT_DIR}/black_dqn_candidate.pt}"
PROMOTED_CHECKPOINT="${PROMOTED_CHECKPOINT:-${REPO_ROOT}/results/demo/learned_player/black_dqn.pt}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"

if [ ! -d "${GAME_LOG_DIR}" ]; then
  echo "Game log directory not found: ${GAME_LOG_DIR}" >&2
  exit 1
fi

if ! find "${GAME_LOG_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) | grep -q .; then
  echo "No game logs found in ${GAME_LOG_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

TRAIN_ARGS=(
  train
  --checkpoint "${CANDIDATE_CHECKPOINT}"
  --offline-log "${GAME_LOG_DIR}"
  --side-filter "${SIDE_FILTER}"
  --offline-optimization-steps "${OFFLINE_OPTIMIZATION_STEPS}"
  --self-play-episodes "${SELF_PLAY_EPISODES}"
  --post-self-play-optimization-steps "${POST_SELF_PLAY_OPTIMIZATION_STEPS}"
  --output-dir "${OUTPUT_DIR}/self_play_logs"
)

if [ -n "${INITIAL_CHECKPOINT}" ]; then
  TRAIN_ARGS+=(--initial-checkpoint "${INITIAL_CHECKPOINT}")
elif [ -f "${PROMOTED_CHECKPOINT}" ]; then
  TRAIN_ARGS+=(--initial-checkpoint "${PROMOTED_CHECKPOINT}")
fi

EVAL_REPORT="${OUTPUT_DIR}/evaluation.json"
PROMOTION_REPORT="${OUTPUT_DIR}/promotion.json"

printf 'Running offline learning cycle.\n'
printf '  game_logs:          %s\n' "${GAME_LOG_DIR}"
printf '  candidate:          %s\n' "${CANDIDATE_CHECKPOINT}"
printf '  promoted_target:    %s\n' "${PROMOTED_CHECKPOINT}"
printf '  side_filter:        %s\n' "${SIDE_FILTER}"
printf '  offline_updates:    %s\n' "${OFFLINE_OPTIMIZATION_STEPS}"
printf '  self_play_episodes: %s\n' "${SELF_PLAY_EPISODES}"

python3 "${REPO_ROOT}/scripts/rl_continual.py" "${TRAIN_ARGS[@]}"
python3 "${REPO_ROOT}/scripts/rl_continual.py" \
  evaluate \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --log "${GAME_LOG_DIR}" \
  --side-filter "${SIDE_FILTER}" \
  --report "${EVAL_REPORT}"
python3 "${REPO_ROOT}/scripts/rl_continual.py" \
  promote \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --report "${EVAL_REPORT}" \
  --destination "${PROMOTED_CHECKPOINT}" \
  --min-legal-rate "${MIN_LEGAL_RATE}" \
  --min-agreement-rate "${MIN_AGREEMENT_RATE}" \
  --overwrite \
  --output "${PROMOTION_REPORT}"

printf 'Offline learning cycle complete.\n'
printf '  evaluation_report: %s\n' "${EVAL_REPORT}"
printf '  promotion_report:  %s\n' "${PROMOTION_REPORT}"
