"""Image-based board extraction helpers for the fixed overhead ROS GZ camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - exercised indirectly in environments with python-chess
    import chess
except ImportError:  # pragma: no cover - runtime fallback
    chess = None  # type: ignore[assignment]


SquareSample = Tuple[str, float]


@dataclass(frozen=True)
class BoardCrop:
    """Normalized board crop inside the full camera image."""

    left: float = 0.225
    top: float = 0.225
    right: float = 0.775
    bottom: float = 0.775


@dataclass(frozen=True)
class BoardObservation:
    """Best-effort observed board estimate from a single camera frame."""

    reference_fen: str
    fen: str
    confidence: float
    matched_move: str
    occupancy_ascii: str
    square_scores: Tuple[SquareSample, ...]


def crop_board(image_bgr: np.ndarray, crop: BoardCrop) -> np.ndarray:
    """Extract the chessboard ROI from a fixed overhead image."""

    height, width = image_bgr.shape[:2]
    x0 = int(width * crop.left)
    y0 = int(height * crop.top)
    x1 = int(width * crop.right)
    y1 = int(height * crop.bottom)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, width)
    y1 = min(y1, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Board crop is empty. Check normalized crop parameters.")
    return image_bgr[y0:y1, x0:x1]


def square_name(file_index: int, rank_index: int) -> str:
    return f"{chr(ord('a') + file_index)}{8 - rank_index}"


def slice_board(board_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Split a top-down board image into 64 square crops."""

    board_size = min(board_bgr.shape[:2])
    square_px = board_size // 8
    squares: List[Tuple[str, np.ndarray]] = []
    for rank_index in range(8):
        for file_index in range(8):
            x0 = file_index * square_px
            y0 = rank_index * square_px
            crop = board_bgr[y0 : y0 + square_px, x0 : x0 + square_px]
            squares.append((square_name(file_index, rank_index), crop))
    return squares


def square_probability_grid(board_bgr: np.ndarray) -> List[SquareSample]:
    """Return raw occupancy scores for each square without thresholding."""

    scores: List[SquareSample] = []
    for name, square in slice_board(board_bgr):
        scores.append((name, round(occupancy_score(square), 4)))
    return scores


def occupancy_score(square_bgr: np.ndarray) -> float:
    """Heuristic piece-presence score based on edges and texture."""

    if square_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    local_std = float(np.std(gray)) / 255.0
    return min((edge_density * 4.0) + (local_std * 1.5), 1.0)


def occupancy_grid(
    board_bgr: np.ndarray,
    threshold: float = 0.18,
) -> List[SquareSample]:
    """Return one occupancy score per square."""

    scores: List[SquareSample] = []
    for name, score in square_probability_grid(board_bgr):
        if score < threshold:
            score = 0.0
        scores.append((name, round(score, 4)))
    return scores


def occupancy_ascii(scores: Sequence[SquareSample]) -> str:
    """Compact 8x8 ASCII occupancy map for logs and debugging."""

    rows: List[str] = []
    for row_index in range(8):
        row_scores = scores[row_index * 8 : (row_index + 1) * 8]
        rows.append("".join("1" if score > 0.0 else "." for _, score in row_scores))
    return "/".join(rows)


def annotate_board(board_bgr: np.ndarray, scores: Sequence[SquareSample]) -> np.ndarray:
    """Overlay square labels and occupancy scores on a board image."""

    annotated = board_bgr.copy()
    board_size = min(board_bgr.shape[:2])
    square_px = board_size // 8
    for index, (name, score) in enumerate(scores):
        row = index // 8
        col = index % 8
        x0 = col * square_px
        y0 = row * square_px
        x1 = x0 + square_px
        y1 = y0 + square_px
        color = (0, 220, 0) if score > 0.0 else (120, 120, 120)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 1)
        cv2.putText(
            annotated,
            f"{name}:{score:.2f}",
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def _square_is_occupied(board: "chess.Board", square_name: str) -> bool:
    square = chess.parse_square(square_name)
    return board.piece_at(square) is not None


def _score_candidate(
    candidate_board: "chess.Board",
    scores: Sequence[SquareSample],
) -> float:
    total = 0.0
    for square_name, score in scores:
        probability = min(max(float(score), 0.0), 1.0)
        if _square_is_occupied(candidate_board, square_name):
            total += probability
        else:
            total += 1.0 - probability
    return total


def infer_observed_board(
    reference_fen: str,
    board_bgr: np.ndarray,
    threshold: float = 0.18,
) -> BoardObservation:
    """Infer the most likely legal board state from occupancy and a reference FEN."""

    raw_scores = tuple(square_probability_grid(board_bgr))
    occupancy_scores = occupancy_grid(board_bgr, threshold=threshold)
    occupancy_text = occupancy_ascii(occupancy_scores)

    if chess is None:
        return BoardObservation(
            reference_fen=reference_fen,
            fen=reference_fen,
            confidence=0.0,
            matched_move="",
            occupancy_ascii=occupancy_text,
            square_scores=raw_scores,
        )

    try:
        reference_board = chess.Board(reference_fen) if reference_fen else chess.Board()
    except Exception:  # pragma: no cover - defensive fallback for malformed FENs
        reference_board = chess.Board()

    candidates: List[Tuple[Optional[str], float, "chess.Board"]] = []
    base_board = reference_board.copy(stack=False)
    candidates.append((None, _score_candidate(base_board, raw_scores), base_board))
    for move in reference_board.legal_moves:
        candidate = reference_board.copy(stack=False)
        candidate.push(move)
        candidates.append((move.uci(), _score_candidate(candidate, raw_scores), candidate))

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_move, best_score, best_board = candidates[0]
    total_squares = float(len(raw_scores) or 64)
    confidence = max(0.0, min(1.0, best_score / total_squares))

    return BoardObservation(
        reference_fen=reference_fen,
        fen=best_board.fen(),
        confidence=round(confidence, 4),
        matched_move=best_move or "",
        occupancy_ascii=occupancy_text,
        square_scores=raw_scores,
    )
