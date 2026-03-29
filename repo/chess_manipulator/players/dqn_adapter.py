"""DQN-backed player adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import PlayerError, PlayerIdentity
from chess_manipulator.rl.adapter import RLMoveAdapter

@dataclass
class DQNPlayerAdapter:
    """Adapter for a learned FEN-to-UCI policy with heuristic fallback."""

    policy_name: str = "dqn"
    checkpoint_path: str = ""
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._adapter = RLMoveAdapter(
            checkpoint_path=self.checkpoint_path or None,
            device=self.device,
        )

    @property
    def identity(self) -> PlayerIdentity:
        return PlayerIdentity(name=self.policy_name, backend="dqn")

    def select_move(self, fen: str) -> str:
        try:
            result = self._adapter.select_move(fen)
        except Exception as exc:  # pragma: no cover - adapter/runtime guard
            raise PlayerError(str(exc)) from exc
        return result.uci_move
