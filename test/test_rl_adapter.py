import chess

from chess_manipulator.rl.adapter import RLMoveAdapter


def test_adapter_returns_legal_move_without_checkpoint():
    adapter = RLMoveAdapter()
    board = chess.Board()

    result = adapter.select_move(board.fen())

    assert chess.Move.from_uci(result.uci_move) in board.legal_moves
    assert result.source == "heuristic_fallback"
