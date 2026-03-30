#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

SIM_BACKEND="${SIM_BACKEND:-isaac}"
WHITE_BACKEND="${WHITE_BACKEND:-stockfish}"
BLACK_BACKEND="${BLACK_BACKEND:-dqn}"
LEARNED_CHECKPOINT="${LEARNED_CHECKPOINT:-}"
WHITE_ENGINE_EXECUTABLE="${WHITE_ENGINE_EXECUTABLE:-}"
BLACK_ENGINE_EXECUTABLE="${BLACK_ENGINE_EXECUTABLE:-}"
ALLOW_HEURISTIC_FALLBACK="${ALLOW_HEURISTIC_FALLBACK:-0}"

if [ "${SIM_BACKEND}" = "isaac" ]; then
  "${REPO_ROOT}/scripts/check_isaac_install.sh"
fi

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

resolve_engine() {
  local backend="$1"
  local configured="$2"
  env PATH=/usr/bin:/bin:$PATH REPO_ROOT="${REPO_ROOT}" /usr/bin/python3 - <<'PY' "${backend}" "${configured}"
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from chess_manipulator.chess.engine import ChessEngineAdapter, EngineConfig

backend = sys.argv[1]
configured = sys.argv[2] or None
adapter = ChessEngineAdapter(EngineConfig(backend=backend, executable=configured))
print(adapter.resolve_executable())
PY
}

if [ "${WHITE_BACKEND}" = "stockfish" ]; then
  WHITE_ENGINE_EXECUTABLE="$(resolve_engine "stockfish" "${WHITE_ENGINE_EXECUTABLE}")"
fi
if [ "${BLACK_BACKEND}" = "stockfish" ]; then
  BLACK_ENGINE_EXECUTABLE="$(resolve_engine "stockfish" "${BLACK_ENGINE_EXECUTABLE}")"
fi

if [ "${BLACK_BACKEND}" = "dqn" ] && [ -n "${LEARNED_CHECKPOINT}" ]; then
  if [ ! -f "${LEARNED_CHECKPOINT}" ]; then
    echo "Learned checkpoint not found: ${LEARNED_CHECKPOINT}" >&2
    exit 1
  fi
  env PATH=/usr/bin:/bin:$PATH REPO_ROOT="${REPO_ROOT}" /usr/bin/python3 - <<'PY' "${LEARNED_CHECKPOINT}"
import sys
from pathlib import Path
import os

checkpoint = Path(sys.argv[1])
if not checkpoint.exists():
    raise SystemExit(f"Checkpoint not found: {checkpoint}")

repo_root = Path(os.environ["REPO_ROOT"])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - shell preflight
    raise SystemExit(f"PyTorch is required for learned-player runtime: {exc}") from exc

from chess_manipulator.rl.adapter import RLMoveAdapter

adapter = RLMoveAdapter(checkpoint_path=str(checkpoint), device="cpu")
result = adapter.select_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
if not result.uci_move:
    raise SystemExit("Learned-player checkpoint loaded but did not produce a move.")
PY
elif [ "${BLACK_BACKEND}" = "dqn" ] && [ "${ALLOW_HEURISTIC_FALLBACK}" != "1" ]; then
  echo "DQN backend selected but no checkpoint provided." >&2
  exit 1
fi

printf 'Learned-player demo preflight passed.\n'
printf '  sim_backend: %s\n' "${SIM_BACKEND}"
printf '  white_backend: %s\n' "${WHITE_BACKEND}"
printf '  black_backend: %s\n' "${BLACK_BACKEND}"
if [ "${WHITE_BACKEND}" = "stockfish" ]; then
  printf '  white_engine: %s\n' "${WHITE_ENGINE_EXECUTABLE}"
fi
if [ "${BLACK_BACKEND}" = "stockfish" ]; then
  printf '  black_engine: %s\n' "${BLACK_ENGINE_EXECUTABLE}"
fi
if [ "${BLACK_BACKEND}" = "dqn" ] && [ -n "${LEARNED_CHECKPOINT}" ]; then
  printf '  learned_checkpoint: %s\n' "${LEARNED_CHECKPOINT}"
fi
