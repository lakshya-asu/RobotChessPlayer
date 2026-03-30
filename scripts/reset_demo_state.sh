#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/stop_demo.sh"

rm -rf /tmp/ros_logs/* 2>/dev/null || true
rm -rf /tmp/ros_home 2>/dev/null || true

printf 'Reset completed.\n'
printf 'Current reset strategy is process + scene restart.\n'
printf 'Relaunch with:\n'
printf '  %s\n' "./repo/scripts/run_engine_vs_learned_demo.sh"
