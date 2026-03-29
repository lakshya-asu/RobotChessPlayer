from __future__ import annotations

import cv2
import numpy as np
import pytest

try:
    import chess
except ImportError:  # pragma: no cover - environment guard
    chess = None

from chess_manipulator.perception.board_vision import (
    BoardCrop,
    annotate_board,
    crop_board,
    infer_observed_board,
    occupancy_ascii,
    occupancy_grid,
)


pytestmark = pytest.mark.skipif(chess is None, reason="python-chess is not installed")


def _synthetic_board() -> np.ndarray:
    image = np.zeros((960, 960, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    board = image[216:744, 216:744]
    square = board.shape[0] // 8
    for row in range(8):
        for col in range(8):
            color = (220, 220, 220) if (row + col) % 2 == 0 else (60, 90, 140)
            y0 = row * square
            x0 = col * square
            board[y0 : y0 + square, x0 : x0 + square] = color
    # Add one black piece-like disk on e2 and one on e7.
    for center in ((4, 6), (4, 1)):
        col, row = center
        cv_x = col * square + (square // 2)
        cv_y = row * square + (square // 2)
        rr, cc = np.ogrid[:board.shape[0], :board.shape[1]]
        mask = (rr - cv_y) ** 2 + (cc - cv_x) ** 2 <= (square // 3) ** 2
        board[mask] = (20, 20, 20)
    return image


def _synthetic_board_from_fen(fen: str) -> np.ndarray:
    image = _synthetic_board()
    board = image[216:744, 216:744]
    square = board.shape[0] // 8
    for row in range(8):
        for col in range(8):
            color = (220, 220, 220) if (row + col) % 2 == 0 else (60, 90, 140)
            y0 = row * square
            x0 = col * square
            board[y0 : y0 + square, x0 : x0 + square] = color

    board_model = chess.Board(fen)
    for square_index, piece in board_model.piece_map().items():
        file_index = chess.square_file(square_index)
        rank_index = 7 - chess.square_rank(square_index)
        center_x = file_index * square + (square // 2)
        center_y = rank_index * square + (square // 2)
        color = (15, 15, 15) if piece.color == chess.BLACK else (245, 245, 245)
        cv2.circle(board, (center_x, center_y), square // 3, color, -1)
    return image


def test_board_crop_and_ascii_map() -> None:
    image = _synthetic_board()
    board = crop_board(image, BoardCrop())
    scores = occupancy_grid(board, threshold=0.05)
    ascii_map = occupancy_ascii(scores)
    assert ascii_map.count("1") >= 2
    assert len(ascii_map.split("/")) == 8


def test_annotation_keeps_board_shape() -> None:
    image = _synthetic_board()
    board = crop_board(image, BoardCrop())
    scores = occupancy_grid(board, threshold=0.05)
    annotated = annotate_board(board, scores)
    assert annotated.shape == board.shape


def test_infer_observed_board_tracks_a_simple_legal_move() -> None:
    reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moved_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    image = _synthetic_board_from_fen(moved_fen)
    board = crop_board(image, BoardCrop())
    observation = infer_observed_board(reference_fen, board, threshold=0.05)

    assert observation.matched_move == "e2e4"
    assert observation.fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert observation.confidence > 0.5
