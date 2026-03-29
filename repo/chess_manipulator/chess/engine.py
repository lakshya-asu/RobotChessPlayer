"""Pluggable chess-engine adapter."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional
import shutil
import subprocess

try:
    import chess
    import chess.engine
except ImportError:  # pragma: no cover - runtime dependency
    chess = None


class EngineError(RuntimeError):
    """Raised when an engine cannot be configured or queried."""


@dataclass(frozen=True)
class EngineConfig:
    """Runtime configuration for UCI-compatible engines."""

    backend: str = "stockfish"
    executable: Optional[str] = None
    think_time_sec: float = 0.25


class ChessEngineAdapter:
    """Adapter for Stockfish, lc0, Fairy-Stockfish, or Sunfish."""

    EXECUTABLE_NAMES = {
        "stockfish": ("stockfish",),
        "lc0": ("lc0",),
        "fairy_stockfish": ("fairy-stockfish", "fairy_stockfish"),
        "sunfish": ("sunfish.py", "sunfish"),
    }
    EXECUTABLE_PATHS = {
        "stockfish": ("/tmp/Stockfish/src/stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish"),
        "lc0": ("/usr/local/bin/lc0",),
        "fairy_stockfish": ("/usr/local/bin/fairy-stockfish",),
        "sunfish": (),
    }
    EXECUTABLE_ENV_VARS = {
        "stockfish": "CHESS_MANIPULATOR_STOCKFISH",
        "lc0": "CHESS_MANIPULATOR_LC0",
        "fairy_stockfish": "CHESS_MANIPULATOR_FAIRY_STOCKFISH",
        "sunfish": "CHESS_MANIPULATOR_SUNFISH",
    }

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    @property
    def backend(self) -> str:
        return self._config.backend

    def resolve_executable(self) -> str:
        if self._config.executable:
            if os.path.exists(self._config.executable):
                return self._config.executable
        env_var = self.EXECUTABLE_ENV_VARS.get(self._config.backend)
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]
        for candidate in self.EXECUTABLE_PATHS.get(self._config.backend, ()):
            if os.path.exists(candidate):
                return candidate
        for candidate in self.EXECUTABLE_NAMES.get(self._config.backend, ()):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise EngineError(
            f"No executable configured for backend '{self._config.backend}'. "
            "Set the 'engine_executable' ROS parameter or the matching CHESS_MANIPULATOR_* env var."
        )

    def get_best_move(self, fen: str, think_time_sec: Optional[float] = None) -> str:
        if chess is None:
            raise EngineError("python-chess is required for engine integration.")

        board = chess.Board(fen)
        executable = self.resolve_executable()
        limit = chess.engine.Limit(time=think_time_sec or self._config.think_time_sec)

        if self._config.backend == "sunfish" and executable.endswith(".py"):
            command = ["python3", executable]
        else:
            command = [executable]

        try:
            with chess.engine.SimpleEngine.popen_uci(command) as engine:
                result = engine.play(board, limit)
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise EngineError(f"Failed to query engine '{executable}': {exc}") from exc

        if result.move is None:
            raise EngineError("Engine returned no move.")
        return result.move.uci()
