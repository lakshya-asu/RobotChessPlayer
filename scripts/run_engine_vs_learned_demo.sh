#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

LEARNED_CHECKPOINT_DEFAULT="${REPO_ROOT}/results/demo/learned_player/black_dqn.pt"
LEARNED_CHECKPOINT="${LEARNED_CHECKPOINT:-${LEARNED_CHECKPOINT_DEFAULT}}"
WHITE_BACKEND="${WHITE_BACKEND:-stockfish}"
BLACK_BACKEND="${BLACK_BACKEND:-dqn}"
ISAAC_HEADLESS="${ISAAC_HEADLESS:-false}"
WHITE_ENGINE_EXECUTABLE="${WHITE_ENGINE_EXECUTABLE:-}"
BLACK_ENGINE_EXECUTABLE="${BLACK_ENGINE_EXECUTABLE:-}"

if [ "${BLACK_BACKEND}" = "dqn" ] && [ ! -f "${LEARNED_CHECKPOINT}" ] && [ "${ALLOW_HEURISTIC_FALLBACK:-0}" != "1" ]; then
  cat >&2 <<EOF
No promoted learned-player checkpoint was found.
Expected: ${LEARNED_CHECKPOINT}

Promote one first:
  ./repo/scripts/promote_learned_checkpoint.sh /path/to/trained_checkpoint.pt

If you intentionally want the current heuristic fallback path, set:
  ALLOW_HEURISTIC_FALLBACK=1
EOF
  exit 1
fi

if [ "${BLACK_BACKEND}" = "dqn" ] && [ ! -f "${LEARNED_CHECKPOINT}" ] && [ "${ALLOW_HEURISTIC_FALLBACK:-0}" = "1" ]; then
  LEARNED_CHECKPOINT=""
fi

if [ "${SIM_BACKEND:-isaac}" = "isaac" ]; then
  "${REPO_ROOT}/scripts/check_isaac_install.sh"
fi

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

env \
  SIM_BACKEND="${SIM_BACKEND:-isaac}" \
  WHITE_BACKEND="${WHITE_BACKEND}" \
  BLACK_BACKEND="${BLACK_BACKEND}" \
  LEARNED_CHECKPOINT="${LEARNED_CHECKPOINT}" \
  WHITE_ENGINE_EXECUTABLE="${WHITE_ENGINE_EXECUTABLE}" \
  BLACK_ENGINE_EXECUTABLE="${BLACK_ENGINE_EXECUTABLE}" \
  ALLOW_HEURISTIC_FALLBACK="${ALLOW_HEURISTIC_FALLBACK:-0}" \
  "${REPO_ROOT}/scripts/check_learned_demo_ready.sh"

printf 'Starting engine-vs-learned-player demo.\n'
printf '  sim_backend: %s\n' "${SIM_BACKEND:-isaac}"
printf '  white_backend: %s\n' "${WHITE_BACKEND}"
printf '  black_backend: %s\n' "${BLACK_BACKEND}"
if [ -f "${LEARNED_CHECKPOINT}" ]; then
  printf '  learned_checkpoint: %s\n' "${LEARNED_CHECKPOINT}"
elif [ "${BLACK_BACKEND}" = "dqn" ]; then
  printf '  learned_checkpoint: not provided, heuristic fallback enabled\n'
fi
printf '  plies: config default (currently 8 unless config changed)\n'

launch_args=(
  coordinator:=true
  coordinator_auto_start:=true
  launch_native_isaac_app:=true
  "isaac_headless:=${ISAAC_HEADLESS}"
  "white_backend:=${WHITE_BACKEND}"
  "black_backend:=${BLACK_BACKEND}"
)

if [ -n "${WHITE_ENGINE_EXECUTABLE}" ]; then
  launch_args+=("white_engine_executable:=${WHITE_ENGINE_EXECUTABLE}")
fi
if [ -n "${BLACK_ENGINE_EXECUTABLE}" ]; then
  launch_args+=("black_engine_executable:=${BLACK_ENGINE_EXECUTABLE}")
fi
if [ -n "${LEARNED_CHECKPOINT}" ]; then
  launch_args+=("black_dqn_checkpoint:=${LEARNED_CHECKPOINT}")
fi

exec env ROS_LOG_DIR=/tmp/ros_logs SIM_BACKEND="${SIM_BACKEND:-isaac}" \
  "${REPO_ROOT}/scripts/run_ros_demo.sh" \
  "${launch_args[@]}" \
  "${@}"
