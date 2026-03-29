"""Shared game coordinator scaffolding."""

from .game_coordinator import GameCoordinator, GameCoordinatorError, GameResult, TurnRecord

__all__ = [
    "GameCoordinator",
    "GameCoordinatorError",
    "GameResult",
    "TurnRecord",
]
