#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SIM_BACKEND=ros_gz
exec "${SCRIPT_DIR}/run_ros_demo.sh" "$@"
