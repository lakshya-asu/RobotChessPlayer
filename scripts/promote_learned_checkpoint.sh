#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  cat >&2 <<'EOF'
Usage:
  ./repo/scripts/promote_learned_checkpoint.sh /path/to/checkpoint.pt [/path/to/checkpoint.json]

This copies an offline-trained learned-player artifact into the stable demo location:
  repo/results/demo/learned_player/
EOF
  exit 1
fi

SOURCE_CHECKPOINT="$1"
SOURCE_METADATA="${2:-}"

if [ ! -f "${SOURCE_CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

if [ -z "${SOURCE_METADATA}" ]; then
  AUTO_METADATA="${SOURCE_CHECKPOINT}.metadata.json"
  if [ -f "${AUTO_METADATA}" ]; then
    SOURCE_METADATA="${AUTO_METADATA}"
  else
    LEGACY_METADATA="${SOURCE_CHECKPOINT%.*}.json"
    if [ -f "${LEGACY_METADATA}" ]; then
      SOURCE_METADATA="${LEGACY_METADATA}"
    fi
  fi
fi

if [ -n "${SOURCE_METADATA}" ] && [ ! -f "${SOURCE_METADATA}" ]; then
  echo "Metadata file not found: ${SOURCE_METADATA}" >&2
  exit 1
fi

TARGET_DIR="${REPO_ROOT}/results/demo/learned_player"
TARGET_CHECKPOINT="${TARGET_DIR}/black_dqn.pt"
TARGET_METADATA="${TARGET_DIR}/black_dqn.pt.metadata.json"
LEGACY_TARGET_METADATA="${TARGET_DIR}/black_dqn.json"
TARGET_MANIFEST="${TARGET_DIR}/PROMOTED_FROM.txt"

mkdir -p "${TARGET_DIR}"
cp "${SOURCE_CHECKPOINT}" "${TARGET_CHECKPOINT}"
if [ -n "${SOURCE_METADATA}" ]; then
  cp "${SOURCE_METADATA}" "${TARGET_METADATA}"
  rm -f "${LEGACY_TARGET_METADATA}"
else
  rm -f "${TARGET_METADATA}" "${LEGACY_TARGET_METADATA}"
fi

{
  printf 'source_checkpoint=%s\n' "${SOURCE_CHECKPOINT}"
  if [ -n "${SOURCE_METADATA}" ]; then
    printf 'source_metadata=%s\n' "${SOURCE_METADATA}"
  fi
  printf 'promoted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${TARGET_MANIFEST}"

printf 'Promoted learned-player artifact.\n'
printf '  checkpoint: %s\n' "${TARGET_CHECKPOINT}"
if [ -n "${SOURCE_METADATA}" ]; then
  printf '  metadata:   %s\n' "${TARGET_METADATA}"
fi
printf '  manifest:   %s\n' "${TARGET_MANIFEST}"
