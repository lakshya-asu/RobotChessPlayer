#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repo/scripts/isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

ISAAC_ROOT_RESOLVED="$(resolve_isaac_root)"
validate_isaac_install "${ISAAC_ROOT_RESOLVED}"
mapfile -t ISAAC_BRIDGE_PATHS < <(build_isaac_ros_bridge_paths "${ISAAC_ROOT_RESOLVED}")
ISAAC_BRIDGE_PYTHONPATH="${ISAAC_BRIDGE_PATHS[0]}"
ISAAC_BRIDGE_LIBRARYPATH="${ISAAC_BRIDGE_PATHS[1]}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
unset PYTHONHOME PYTHONSTARTUP PYTHONPATH

export PATH="$(sanitize_path "${PATH:-}")"
if [ -n "${ISAAC_BRIDGE_PYTHONPATH}" ]; then
  export PYTHONPATH="${ISAAC_BRIDGE_PYTHONPATH// /:}:${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
if [ -n "${ISAAC_BRIDGE_LIBRARYPATH}" ]; then
  export LD_LIBRARY_PATH="${ISAAC_BRIDGE_LIBRARYPATH// /:}:${LD_LIBRARY_PATH:-}"
fi
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
export ROS_HOME="${ROS_HOME:-/tmp/ros_home}"
export ISAAC_ROOT="${ISAAC_ROOT_RESOLVED}"

exec "${ISAAC_ROOT_RESOLVED}/python.sh" -m isaac_app --sync-board-state "$@"
