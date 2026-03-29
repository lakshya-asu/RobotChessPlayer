#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repo/scripts/isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

ISAAC_ROOT_RESOLVED="$(resolve_isaac_root)"
validate_isaac_install "${ISAAC_ROOT_RESOLVED}"
ISAAC_VERSION_FOUND="$(read_isaac_version "${ISAAC_ROOT_RESOLVED}")"

printf 'Isaac Sim install looks usable.\n'
printf '  path: %s\n' "${ISAAC_ROOT_RESOLVED}"
if [ -n "${ISAAC_VERSION_FOUND}" ]; then
  printf '  version: %s\n' "${ISAAC_VERSION_FOUND}"
fi
printf '  expected series: %s.x\n' "${ISAAC_VERSION_EXPECTED}"
printf '  python entrypoint: %s\n' "${ISAAC_ROOT_RESOLVED}/python.sh"
