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
    PIECE_TO_LABEL,
    annotate_board,
    board_bounds_from_aruco,
    board_origin_from_bounds,
    board_signal_score,
    build_piece_templates,
    crop_board,
    fuse_square_scores,
    infer_observed_board_from_labels,
    infer_observed_board,
    infer_observed_board_from_templates,
    occupancy_ascii,
    occupancy_grid,
    resize_board_image,
    segmentation_probability_grid,
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
        glyph_color = (245, 245, 245) if piece.color == chess.BLACK else (15, 15, 15)
        cv2.putText(
            board,
            piece.symbol().upper(),
            (center_x - square // 6, center_y + square // 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            glyph_color,
            2,
            cv2.LINE_AA,
        )
    return image


def _synthetic_label_board_from_fen(fen: str) -> np.ndarray:
    image = np.zeros((960, 960), dtype=np.uint8)
    board = image[216:744, 216:744]
    square = board.shape[0] // 8
    board_model = chess.Board(fen)
    for square_index, piece in board_model.piece_map().items():
        file_index = chess.square_file(square_index)
        rank_index = 7 - chess.square_rank(square_index)
        center_x = file_index * square + (square // 2)
        center_y = rank_index * square + (square // 2)
        rr, cc = np.ogrid[:board.shape[0], :board.shape[1]]
        mask = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= (square // 3) ** 2
        board[mask] = PIECE_TO_LABEL[piece.symbol()]
    return image


def _draw_aruco_anchor(image: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco is not available")
    dict_id = getattr(cv2.aruco, "DICT_4X4_50", None)
    if dict_id is None:
        pytest.skip("DICT_4X4_50 is not available")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board_left = int(image.shape[1] * 0.225)
    board_top = int(image.shape[0] * 0.225)
    board_width = int(image.shape[1] * (0.775 - 0.225))
    square = board_width // 8
    marker = cv2.aruco.generateImageMarker(dictionary, 23, square)
    gap = max(int(square * 0.35), 12)
    marker_left = int(board_left + (board_width / 2) - (square / 2))
    marker_top = board_top - gap - square
    image[marker_top : marker_top + square, marker_left : marker_left + square] = cv2.cvtColor(
        marker,
        cv2.COLOR_GRAY2BGR,
    )
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


def test_resize_board_image_aligns_to_eight_square_grid() -> None:
    board = np.zeros((517, 517, 3), dtype=np.uint8)

    resized = resize_board_image(board, size_px=517)

    assert resized.shape == (512, 512, 3)


def test_segmentation_scores_can_be_fused_with_image_scores() -> None:
    image = _synthetic_board()
    board = crop_board(image, BoardCrop())
    image_scores = occupancy_grid(board, threshold=0.0)
    mask = np.zeros((board.shape[0], board.shape[1]), dtype=np.uint8)
    square = board.shape[0] // 8
    cv2.rectangle(mask, (4 * square, 6 * square), (5 * square, 7 * square), 255, -1)

    segmentation_scores = segmentation_probability_grid(mask)
    fused_scores = fuse_square_scores(image_scores, segmentation_scores, segmentation_weight=0.9)

    fused_lookup = dict(fused_scores)
    assert fused_lookup["e2"] > fused_lookup["a3"]


def test_blank_board_has_low_signal_and_does_not_override_reference() -> None:
    blank_board = np.full((512, 512, 3), 255, dtype=np.uint8)

    observation = infer_observed_board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        blank_board,
        threshold=0.05,
        min_signal_score=0.05,
    )

    assert observation.fen.startswith("rnbqkbnr/pppppppp")
    assert observation.confidence == 0.0
    assert observation.signal_score < 0.05


def test_infer_observed_board_tracks_a_simple_legal_move() -> None:
    reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moved_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    image = _synthetic_board_from_fen(moved_fen)
    board = crop_board(image, BoardCrop())
    observation = infer_observed_board(reference_fen, board, threshold=0.05)

    assert observation.matched_move == "e2e4"
    assert observation.fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert observation.confidence > 0.5
    assert observation.signal_score > 0.0


def test_infer_observed_board_from_labels_recovers_piece_identity() -> None:
    reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moved_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

    label_image = _synthetic_label_board_from_fen(moved_fen)
    board = crop_board(label_image, BoardCrop())
    observation = infer_observed_board_from_labels(reference_fen, board)

    assert observation.fen.startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3")
    assert observation.confidence == 1.0
    assert observation.signal_score == 1.0


def test_piece_templates_track_simple_move_from_rgb_and_segmentation() -> None:
    reference_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moved_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    reference_board = crop_board(_synthetic_board_from_fen(reference_fen), BoardCrop())
    reference_labels = crop_board(_synthetic_label_board_from_fen(reference_fen), BoardCrop())
    templates = build_piece_templates(reference_fen, reference_board, reference_labels)

    moved_board = crop_board(_synthetic_board_from_fen(moved_fen), BoardCrop())
    moved_labels = crop_board(_synthetic_label_board_from_fen(moved_fen), BoardCrop())
    observation = infer_observed_board_from_templates(
        reference_fen,
        moved_board,
        moved_labels,
        templates,
        threshold=0.05,
    )

    assert observation.matched_move == "e2e4"
    assert observation.fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert observation.confidence > 0.5


def test_board_bounds_from_aruco_anchor_tracks_board_roi() -> None:
    image = _draw_aruco_anchor(_synthetic_board())

    x0, y0, x1, y1 = board_bounds_from_aruco(image, BoardCrop(), marker_id=23)

    assert abs(x0 - 216) <= 10
    assert abs(y0 - 216) <= 10
    assert abs(x1 - 744) <= 10
    assert abs(y1 - 744) <= 10


def test_board_origin_from_bounds_maps_pixel_shift_into_world_shift() -> None:
    nominal_bounds = (216, 216, 744, 744)
    observed_bounds = (226, 206, 754, 734)

    origin = board_origin_from_bounds(
        observed_bounds,
        nominal_bounds,
        nominal_origin=(0.235, -0.266, 0.31),
        square_size_m=0.076,
    )

    meters_per_px = (0.076 * 8.0) / (744 - 216)
    assert origin[0] == pytest.approx(0.235 + (10 * meters_per_px))
    assert origin[1] == pytest.approx(-0.266 - (-10 * meters_per_px))
    assert origin[2] == pytest.approx(0.31)
