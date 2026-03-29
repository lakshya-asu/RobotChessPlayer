#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
SIM_BACKEND="${SIM_BACKEND:-ros_gz}"

if [ "${SIM_BACKEND}" = "ros_gz" ]; then
  if [ "${ROS_GZ_FORCE_SOFTWARE_RENDERING:-0}" = "1" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
  fi
fi

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

exec env ROS_LOG_DIR=/tmp/ros_logs ros2 launch chess_manipulator bringup.launch.py \
  "sim_backend:=${SIM_BACKEND}" \
  "${@}"
