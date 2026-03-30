"""Board encoding helpers for RL models."""

from __future__ import annotations

from typing import List

import chess

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
FEATURE_PLANES = 19


def _square_offset(square: chess.Square) -> int:
    rank = chess.square_rank(square)
    file_index = chess.square_file(square)
    return rank * 8 + file_index


def feature_size() -> int:
    return FEATURE_PLANES * 64


def encode_board(board: chess.Board) -> List[float]:
    """Encode a board into flattened float features."""
    features = [0.0] * feature_size()

    for color_offset, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_offset, piece_type in enumerate(PIECE_TYPES):
            plane = color_offset * len(PIECE_TYPES) + piece_offset
            for square in board.pieces(piece_type, color):
                features[(plane * 64) + _square_offset(square)] = 1.0

    side_plane = 12
    if board.turn == chess.WHITE:
        for square in chess.SQUARES:
            features[(side_plane * 64) + _square_offset(square)] = 1.0

    castling_planes = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for plane_offset, enabled in enumerate(castling_planes, start=13):
        if enabled:
            for square in chess.SQUARES:
                features[(plane_offset * 64) + _square_offset(square)] = 1.0

    if board.ep_square is not None:
        features[(17 * 64) + _square_offset(board.ep_square)] = 1.0

    normalized_halfmove_clock = min(board.halfmove_clock, 100) / 100.0
    normalized_fullmove_number = min(board.fullmove_number, 200) / 200.0
    for square in chess.SQUARES:
        features[(18 * 64) + _square_offset(square)] = normalized_halfmove_clock
        features[(18 * 64) + _square_offset(square)] += normalized_fullmove_number * 0.5

    return features
