"""Shared game coordinator scaffolding."""

from .game_coordinator import (
    GameCoordinator,
    GameCoordinatorError,
    GameResult,
    TurnRecord,
    build_player_adapter,
    expected_fen_after_move,
    perception_confirmation_valid,
    side_to_move_from_fen,
)
from .game_logging import GameLog, PlyLog, legal_moves_for_fen, write_game_log

__all__ = [
    "GameCoordinator",
    "GameCoordinatorError",
    "GameLog",
    "GameResult",
    "PlyLog",
    "TurnRecord",
    "build_player_adapter",
    "expected_fen_after_move",
    "legal_moves_for_fen",
    "perception_confirmation_valid",
    "side_to_move_from_fen",
    "write_game_log",
]
