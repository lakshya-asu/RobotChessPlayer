#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PROJECT_LOCAL_ISAAC_ROOT="${WORKSPACE_ROOT}/third_party/isaac-sim-4.2.0"
LEGACY_ISAAC_ROOT="${HOME}/snydrone_ws/isaacsim"
ISAAC_ROOT="${ISAAC_ROOT:-${PROJECT_LOCAL_ISAAC_ROOT}}"

if [ ! -d "${ISAAC_ROOT}" ] && [ "${ISAAC_ROOT}" = "${PROJECT_LOCAL_ISAAC_ROOT}" ] && [ -d "${LEGACY_ISAAC_ROOT}" ]; then
  ISAAC_ROOT="${LEGACY_ISAAC_ROOT}"
fi

if [ ! -d "${ISAAC_ROOT}" ]; then
  cat >&2 <<EOF
Isaac Sim not found.
Expected path: ${PROJECT_LOCAL_ISAAC_ROOT}
Override with: ISAAC_ROOT=/path/to/isaac-sim-4.2.0 ./scripts/run_isaac_demo.sh
EOF
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

resolve_bridge_root() {
  local isaac_root="$1"
  local candidate=""
  for candidate in \
    "${isaac_root}/exts/isaacsim.ros2.bridge/humble" \
    "${isaac_root}/exts/omni.isaac.ros2_bridge/humble"
  do
    if [ -d "${candidate}" ]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  return 1
}

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
unset PYTHONHOME PYTHONSTARTUP PYTHONPATH

BRIDGE_ROOT="$(resolve_bridge_root "${ISAAC_ROOT}")" || {
  echo "Could not locate the Isaac ROS 2 bridge extension under ${ISAAC_ROOT}/exts." >&2
  exit 1
}

export PATH="$(sanitize_path "${PATH:-}")"
export PYTHONPATH="${BRIDGE_ROOT}/rclpy:${REPO_ROOT}"
export LD_LIBRARY_PATH="${BRIDGE_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"
export ISAAC_ROOT

exec "${ISAAC_ROOT}/python.sh" -m isaac_app --sync-board-state "$@"
