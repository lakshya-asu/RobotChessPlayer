"""Small deterministic heuristics used for fallback move selection and rewards."""

from __future__ import annotations

from typing import Iterable

import chess

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}


def move_priority(board: chess.Board, move: chess.Move) -> float:
    score = 0.0
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured is not None:
            score += 5.0 + PIECE_VALUES[captured.piece_type]
    if board.gives_check(move):
        score += 2.0
    if move.to_square in CENTER_SQUARES:
        score += 0.5
    if move.promotion:
        score += 8.0 + PIECE_VALUES.get(move.promotion, 0.0)
    if board.is_castling(move):
        score += 0.75
    return score


def best_heuristic_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("Board has no legal moves.")
    return max(legal_moves, key=lambda move: (move_priority(board, move), move.uci()))


def material_balance(board: chess.Board, color: chess.Color) -> float:
    own = 0.0
    opp = 0.0
    for piece_type, value in PIECE_VALUES.items():
        own += len(board.pieces(piece_type, color)) * value
        opp += len(board.pieces(piece_type, not color)) * value
    return own - opp
