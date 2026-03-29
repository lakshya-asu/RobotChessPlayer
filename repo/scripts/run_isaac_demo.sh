#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-${HOME}/snydrone_ws/isaacsim}"

if [ ! -d "${ISAAC_ROOT}" ]; then
  echo "Isaac Sim not found at ${ISAAC_ROOT}. Set ISAAC_ROOT to your install path." >&2
  exit 1
fi

sanitize_path() {
  local path_value="$1"
  local sanitized=""
  local segment=""
  local old_ifs="${IFS}"
  IFS=':'
  for segment in ${path_value}; do
    case "${segment}" in
      *miniconda*|*anaconda*)
        continue
        ;;
    esac
    if [[ -z "${sanitized}" ]]; then
      sanitized="${segment}"
    else
      sanitized="${sanitized}:${segment}"
    fi
  done
  IFS="${old_ifs}"
  printf '%s' "${sanitized}"
}

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
unset PYTHONHOME PYTHONSTARTUP PYTHONPATH

export PATH="$(sanitize_path "${PATH:-}")"
export PYTHONPATH="${ISAAC_ROOT}/exts/isaacsim.ros2.bridge/humble/rclpy:${REPO_ROOT}"
export LD_LIBRARY_PATH="${ISAAC_ROOT}/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH:-}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"

exec "${ISAAC_ROOT}/python.sh" -m isaac_app --sync-board-state "$@"
