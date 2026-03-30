from dataclasses import dataclass

import pytest

from chess_manipulator.coordinator import (
    GameCoordinator,
    GameCoordinatorError,
    expected_fen_after_move,
    perception_confirmation_valid,
    side_to_move_from_fen,
)
from chess_manipulator.players.base import PlayerIdentity


@dataclass
class StaticPlayer:
    identity: PlayerIdentity
    move: str

    def select_move(self, fen: str) -> str:
        del fen
        return self.move


def test_side_to_move_from_fen():
    assert side_to_move_from_fen("8/8/8/8/8/8/8/8 w - - 0 1") == "white"
    assert side_to_move_from_fen("8/8/8/8/8/8/8/8 b - - 0 1") == "black"


def test_expected_fen_after_move():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert expected_fen_after_move(fen, "e2e4") == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_game_coordinator_alternates_white_and_black():
    white = StaticPlayer(PlayerIdentity(name="white_test", backend="stub"), "e2e4")
    black = StaticPlayer(PlayerIdentity(name="black_test", backend="stub"), "e7e5")
    coordinator = GameCoordinator(white_player=white, black_player=black, max_plies=2)

    result = coordinator.play()

    assert len(result.turns) == 2
    assert result.turns[0].side_to_move == "white"
    assert result.turns[0].uci_move == "e2e4"
    assert result.turns[1].side_to_move == "black"
    assert result.turns[1].uci_move == "e7e5"


def test_expected_fen_after_move_rejects_illegal_move():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(GameCoordinatorError):
        expected_fen_after_move(fen, "e7e5")


def test_perception_confirmation_valid_requires_fresh_pose_and_fen():
    assert perception_confirmation_valid(
        expected_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        observed_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        observed_confidence=0.9,
        confidence_threshold=0.65,
        observed_update_time=10.0,
        board_pose_update_time=10.0,
        min_update_time=9.0,
    )
    assert not perception_confirmation_valid(
        expected_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        observed_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        observed_confidence=0.9,
        confidence_threshold=0.65,
        observed_update_time=10.0,
        board_pose_update_time=8.5,
        min_update_time=9.0,
    )
