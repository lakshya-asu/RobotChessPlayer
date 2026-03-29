"""Engine-backed player adapter."""

from __future__ import annotations

from dataclasses import dataclass

from chess_manipulator.chess.engine import ChessEngineAdapter, EngineConfig, EngineError

from .base import PlayerError, PlayerIdentity


@dataclass
class EnginePlayerAdapter:
    """Expose the existing UCI engine adapter through the player protocol."""

    engine_config: EngineConfig
    display_name: str = "engine"

    def __post_init__(self) -> None:
        self._engine = ChessEngineAdapter(self.engine_config)

    @property
    def identity(self) -> PlayerIdentity:
        return PlayerIdentity(name=self.display_name, backend=self.engine_config.backend)

    def select_move(self, fen: str) -> str:
        try:
            return self._engine.get_best_move(
                fen=fen,
                think_time_sec=self.engine_config.think_time_sec,
            )
        except EngineError as exc:
            raise PlayerError(str(exc)) from exc
