import pytest

from chess_manipulator.chess.board import BoardError, ChessBoardManager


def _board_available():
    try:
        ChessBoardManager()
    except BoardError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _board_available(), reason="python-chess is not installed")


def test_push_move_updates_fen():
    board = ChessBoardManager()
    board.push_uci("e2e4")

    assert board.fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")


def test_illegal_move_raises():
    board = ChessBoardManager()

    with pytest.raises(BoardError):
        board.push_uci("e2e5")


def test_castling_and_promotion_descriptions():
    board = ChessBoardManager("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    description = board.describe_move("e1g1")
    assert description.is_castling is True

    board = ChessBoardManager("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    description = board.describe_move("a7a8q")
    assert description.promotion == "q"
