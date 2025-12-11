from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine


@dataclass
class EngineConfig:
    engine_path: Path
    thinking_time_s: float = 0.5


class ChessEngineInterface:
    """Wraps python-chess Board and UCI engine interaction."""

    def __init__(self, engine_config: Optional[EngineConfig] = None) -> None:
        self.board = chess.Board()
        self.engine_config = engine_config
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def set_fen(self, fen: str) -> None:
        self.board = chess.Board(fen)

    def get_fen(self) -> str:
        return self.board.fen()

    def connect_engine(self) -> None:
        if self.engine_config is None:
            raise RuntimeError("No engine_config provided")
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                [str(self.engine_config.engine_path)]
            )

    def best_move(self) -> chess.Move:
        """Query engine for best move from current FEN."""
        if self._engine is None:
            self.connect_engine()
        assert self.engine_config is not None
        result = self._engine.play(
            self.board,
            chess.engine.Limit(time=self.engine_config.thinking_time_s),
        )
        return result.move

    def push_move(self, move: chess.Move) -> None:
        self.board.push(move)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
