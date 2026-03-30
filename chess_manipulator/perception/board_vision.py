"""Image-based board extraction helpers for the fixed overhead simulator camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - exercised indirectly in environments with python-chess
    import chess
except ImportError:  # pragma: no cover - runtime fallback
    chess = None  # type: ignore[assignment]


SquareSample = Tuple[str, float]
PieceTemplates = Dict[str, np.ndarray]
LABEL_TO_PIECE: Dict[int, str] = {
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    7: "p",
    8: "n",
    9: "b",
    10: "r",
    11: "q",
    12: "k",
}
PIECE_TO_LABEL: Dict[str, int] = {value: key for key, value in LABEL_TO_PIECE.items()}
ARUCO_FALLBACK_ID = 23
ARUCO_FALLBACK_DICT = "DICT_4X4_50"


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
    signal_score: float
    matched_move: str
    occupancy_ascii: str
    square_scores: Tuple[SquareSample, ...]


def _board_map_from_fen(fen: str) -> Dict[str, str]:
    board_part = (fen or "").split()[0]
    ranks = board_part.split("/")
    if len(ranks) != 8:
        return {}
    pieces: Dict[str, str] = {}
    for fen_rank_index, rank_data in enumerate(ranks):
        file_index = 0
        board_rank = 8 - fen_rank_index
        for char in rank_data:
            if char.isdigit():
                file_index += int(char)
                continue
            if file_index < 8:
                pieces[f"{chr(ord('a') + file_index)}{board_rank}"] = char
            file_index += 1
    return pieces


def board_map_to_fen(board_map: Dict[str, str], reference_fen: str = "") -> str:
    """Serialize a square->piece map into a FEN, preserving side-to-move metadata."""

    rows: List[str] = []
    for rank in range(8, 0, -1):
        row: List[str] = []
        empty_count = 0
        for file_char in "abcdefgh":
            square = f"{file_char}{rank}"
            piece = board_map.get(square, "")
            if piece:
                if empty_count:
                    row.append(str(empty_count))
                    empty_count = 0
                row.append(piece)
            else:
                empty_count += 1
        if empty_count:
            row.append(str(empty_count))
        rows.append("".join(row))

    board_part = "/".join(rows)
    if not reference_fen:
        return f"{board_part} w - - 0 1"

    fields = reference_fen.split()
    if len(fields) < 6:
        return f"{board_part} w - - 0 1"
    fields[0] = board_part
    return " ".join(fields[:6])


def board_pixel_bounds(image_shape: Sequence[int], crop: BoardCrop) -> Tuple[int, int, int, int]:
    """Return the pixel bounds of the configured board crop."""

    height, width = image_shape[:2]
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
    return (x0, y0, x1, y1)


def crop_with_bounds(image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image using precomputed pixel bounds."""

    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1]


def crop_board(image_bgr: np.ndarray, crop: BoardCrop) -> np.ndarray:
    """Extract the chessboard ROI from a fixed overhead image."""
    return crop_with_bounds(image_bgr, board_pixel_bounds(image_bgr.shape, crop))


def _aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        return None
    dict_id = getattr(cv2.aruco, dictionary_name, None)
    if dict_id is None:
        return None
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_aruco_marker(
    image_bgr: np.ndarray,
    marker_id: int = ARUCO_FALLBACK_ID,
    dictionary_name: str = ARUCO_FALLBACK_DICT,
) -> Optional[np.ndarray]:
    """Detect a known ArUco marker and return its 4x2 corner array."""

    dictionary = _aruco_dictionary(dictionary_name)
    if dictionary is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    for detected_corners, detected_id in zip(corners, ids.flatten()):
        if int(detected_id) == int(marker_id):
            return detected_corners.reshape(4, 2).astype(np.float32)
    return None


def board_bounds_from_aruco(
    image_bgr: np.ndarray,
    fallback_crop: BoardCrop,
    marker_id: int = ARUCO_FALLBACK_ID,
    dictionary_name: str = ARUCO_FALLBACK_DICT,
) -> Tuple[int, int, int, int]:
    """Infer board pixel bounds from a top-edge ArUco marker or fall back to the fixed crop."""

    marker_corners = detect_aruco_marker(
        image_bgr,
        marker_id=marker_id,
        dictionary_name=dictionary_name,
    )
    if marker_corners is None:
        return board_pixel_bounds(image_bgr.shape, fallback_crop)

    marker_center_x = float(np.mean(marker_corners[:, 0]))
    marker_bottom_y = float(np.max(marker_corners[:, 1]))
    marker_size_px = float(np.linalg.norm(marker_corners[0] - marker_corners[1]))
    if marker_size_px <= 1.0:
        return board_pixel_bounds(image_bgr.shape, fallback_crop)

    board_size_px = marker_size_px * 8.0
    gap_px = marker_size_px * 0.35
    x0 = int(round(marker_center_x - (board_size_px / 2.0)))
    y0 = int(round(marker_bottom_y + gap_px))
    x1 = int(round(x0 + board_size_px))
    y1 = int(round(y0 + board_size_px))

    height, width = image_bgr.shape[:2]
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, width)
    y1 = min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return board_pixel_bounds(image_bgr.shape, fallback_crop)
    return (x0, y0, x1, y1)


def board_origin_from_bounds(
    observed_bounds: Tuple[int, int, int, int],
    nominal_bounds: Tuple[int, int, int, int],
    nominal_origin: Tuple[float, float, float],
    square_size_m: float,
) -> Tuple[float, float, float]:
    """Estimate board origin in world coordinates from observed image bounds."""

    nominal_width_px = max(float(nominal_bounds[2] - nominal_bounds[0]), 1.0)
    meters_per_px = (square_size_m * 8.0) / nominal_width_px
    delta_x_px = float(observed_bounds[0] - nominal_bounds[0])
    delta_y_px = float(observed_bounds[1] - nominal_bounds[1])
    return (
        nominal_origin[0] + (delta_x_px * meters_per_px),
        nominal_origin[1] - (delta_y_px * meters_per_px),
        nominal_origin[2],
    )


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


def resize_board_image(board_image: np.ndarray, size_px: int = 800) -> np.ndarray:
    """Resize a square board crop to a square size divisible by eight."""

    clamped_size = max(8, int(size_px))
    aligned_size = max(8, (clamped_size // 8) * 8)
    if aligned_size == board_image.shape[0] == board_image.shape[1]:
        return board_image
    return cv2.resize(board_image, (aligned_size, aligned_size), interpolation=cv2.INTER_AREA)


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


def threshold_scores(
    scores: Sequence[SquareSample],
    threshold: float = 0.18,
) -> List[SquareSample]:
    """Apply an occupancy threshold to per-square probabilities."""

    thresholded: List[SquareSample] = []
    for name, score in scores:
        thresholded.append((name, round(score if score >= threshold else 0.0, 4)))
    return thresholded


def segmentation_probability_grid(mask_image: np.ndarray) -> List[SquareSample]:
    """Estimate square occupancy probabilities from a segmentation-like mask."""

    if mask_image.ndim == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image.copy()

    if mask_gray.dtype != np.uint8:
        normalized = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)
        mask_gray = normalized.astype(np.uint8)

    scores: List[SquareSample] = []
    for name, square in slice_board(mask_gray):
        occupied_ratio = float(np.count_nonzero(square > 0)) / float(square.size or 1)
        scores.append((name, round(min(max(occupied_ratio, 0.0), 1.0), 4)))
    return scores


def fuse_square_scores(
    image_scores: Sequence[SquareSample],
    segmentation_scores: Sequence[SquareSample],
    segmentation_weight: float = 0.7,
) -> List[SquareSample]:
    """Fuse image texture scores with segmentation occupancy probabilities."""

    seg_lookup: Dict[str, float] = {name: float(score) for name, score in segmentation_scores}
    weight = min(max(float(segmentation_weight), 0.0), 1.0)
    fused_scores: List[SquareSample] = []
    for name, image_score in image_scores:
        seg_score = seg_lookup.get(name, float(image_score))
        fused = ((1.0 - weight) * float(image_score)) + (weight * float(seg_score))
        fused_scores.append((name, round(min(max(fused, 0.0), 1.0), 4)))
    return fused_scores


def occupancy_ascii(scores: Sequence[SquareSample]) -> str:
    """Compact 8x8 ASCII occupancy map for logs and debugging."""

    rows: List[str] = []
    for row_index in range(8):
        row_scores = scores[row_index * 8 : (row_index + 1) * 8]
        rows.append("".join("1" if score > 0.0 else "." for _, score in row_scores))
    return "/".join(rows)


def board_signal_score(board_bgr: np.ndarray) -> float:
    """Estimate whether the cropped image contains board texture or pieces."""

    if board_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    local_std = float(np.std(gray)) / 255.0
    edges = cv2.Canny(gray, 30, 90)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
    return round(min((local_std * 1.8) + (edge_density * 2.5), 1.0), 4)


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


def decode_piece_labels(label_image: np.ndarray) -> Dict[str, str]:
    """Decode square-wise piece identities from a mono8 semantic label image."""

    if label_image.ndim != 2:
        raise ValueError("Piece labels must be provided as a mono8 image.")

    piece_map: Dict[str, str] = {}
    for square_name_, square in slice_board(label_image):
        if square.size == 0:
            continue
        center_y = square.shape[0] // 2
        center_x = square.shape[1] // 2
        label_value = int(square[center_y, center_x])
        piece_symbol = LABEL_TO_PIECE.get(label_value)
        if piece_symbol:
            piece_map[square_name_] = piece_symbol
    return piece_map


def extract_piece_feature(square_bgr: np.ndarray, square_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract a compact per-square feature for piece classification."""

    if square_bgr.ndim == 2:
        square_bgr = cv2.cvtColor(square_bgr, cv2.COLOR_GRAY2BGR)
    if square_mask is None:
        mask = np.ones(square_bgr.shape[:2], dtype=np.uint8)
    else:
        mask = (square_mask > 0).astype(np.uint8)

    occupancy_ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
    if occupancy_ratio <= 1e-6:
        return np.zeros(12, dtype=np.float32)

    masked_pixels = square_bgr[mask > 0]
    mean_bgr = masked_pixels.mean(axis=0) / 255.0

    ys, xs = np.nonzero(mask)
    width = float(xs.max() - xs.min() + 1) / float(mask.shape[1])
    height = float(ys.max() - ys.min() + 1) / float(mask.shape[0])
    aspect = width / max(height, 1e-6)

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-6)
    hu = hu[:5]

    return np.concatenate(
        [
            mean_bgr.astype(np.float32),
            np.array([occupancy_ratio, width, height, aspect], dtype=np.float32),
            hu.astype(np.float32),
        ]
    )


def build_piece_templates(
    reference_fen: str,
    board_bgr: np.ndarray,
    label_image: np.ndarray,
) -> PieceTemplates:
    """Bootstrap piece-type templates from a labeled initial board image."""

    piece_map = _board_map_from_fen(reference_fen)
    rgb_squares = dict(slice_board(board_bgr))
    label_squares = dict(slice_board(label_image))
    collected: Dict[str, List[np.ndarray]] = {}

    for square_name_, piece_symbol in piece_map.items():
        square_rgb = rgb_squares.get(square_name_)
        square_label = label_squares.get(square_name_)
        if square_rgb is None or square_label is None:
            continue
        mask = (square_label > 0).astype(np.uint8)
        feature = extract_piece_feature(square_rgb, mask)
        if not np.any(feature):
            continue
        collected.setdefault(piece_symbol, []).append(feature)

    templates: PieceTemplates = {}
    for piece_symbol, features in collected.items():
        templates[piece_symbol] = np.mean(np.stack(features, axis=0), axis=0)
    return templates


def _template_similarity(feature: np.ndarray, template: np.ndarray) -> float:
    distance = float(np.linalg.norm(feature - template))
    return 1.0 / (1.0 + distance)


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


def _matched_move_for_observed_fen(reference_fen: str, observed_fen: str) -> str:
    if chess is None:
        return ""
    try:
        reference_board = chess.Board(reference_fen) if reference_fen else chess.Board()
        observed_board = chess.Board(observed_fen)
    except Exception:  # pragma: no cover - defensive fallback for malformed FENs
        return ""

    if reference_board.board_fen() == observed_board.board_fen():
        return ""

    for move in reference_board.legal_moves:
        candidate = reference_board.copy(stack=False)
        candidate.push(move)
        if candidate.board_fen() == observed_board.board_fen():
            return move.uci()
    return ""


def infer_observed_board_from_labels(
    reference_fen: str,
    label_image: np.ndarray,
) -> BoardObservation:
    """Infer an observed board directly from mono8 semantic labels."""

    piece_map = decode_piece_labels(label_image)
    observed_fen = board_map_to_fen(piece_map, reference_fen=reference_fen)
    raw_scores = tuple(
        (square_name_, 1.0 if square_name_ in piece_map else 0.0)
        for square_name_, _ in slice_board(label_image)
    )
    return BoardObservation(
        reference_fen=reference_fen,
        fen=observed_fen,
        confidence=1.0,
        signal_score=1.0,
        matched_move=_matched_move_for_observed_fen(reference_fen, observed_fen),
        occupancy_ascii=occupancy_ascii(raw_scores),
        square_scores=raw_scores,
    )


def infer_observed_board_from_templates(
    reference_fen: str,
    board_bgr: np.ndarray,
    segmentation_mask: np.ndarray,
    templates: PieceTemplates,
    threshold: float = 0.18,
    min_signal_score: float = 0.02,
) -> BoardObservation:
    """Infer a legal next board state using template-matched occupied squares."""

    signal_score = board_signal_score(board_bgr)
    raw_scores: List[SquareSample] = segmentation_probability_grid(segmentation_mask)
    occupancy_scores = threshold_scores(raw_scores, threshold=threshold)
    occupancy_text = occupancy_ascii(occupancy_scores)

    if signal_score < min_signal_score or chess is None or not templates:
        return BoardObservation(
            reference_fen=reference_fen,
            fen=reference_fen,
            confidence=0.0,
            signal_score=signal_score,
            matched_move="",
            occupancy_ascii=occupancy_text,
            square_scores=tuple(raw_scores),
        )

    rgb_squares = dict(slice_board(board_bgr))
    mask_squares = dict(slice_board(segmentation_mask))
    observed_features: Dict[str, np.ndarray] = {}
    for square_name_, occ_score in occupancy_scores:
        if occ_score <= 0.0:
            continue
        observed_features[square_name_] = extract_piece_feature(
            rgb_squares[square_name_],
            mask_squares[square_name_],
        )

    try:
        reference_board = chess.Board(reference_fen) if reference_fen else chess.Board()
    except Exception:
        reference_board = chess.Board()

    candidates: List[Tuple[Optional[str], float, "chess.Board"]] = []
    base_board = reference_board.copy(stack=False)
    candidates.append((None, _score_candidate_with_templates(base_board, observed_features, occupancy_scores, templates), base_board))
    for move in reference_board.legal_moves:
        candidate = reference_board.copy(stack=False)
        candidate.push(move)
        candidates.append(
            (
                move.uci(),
                _score_candidate_with_templates(candidate, observed_features, occupancy_scores, templates),
                candidate,
            )
        )

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_move, best_score, best_board = candidates[0]
    confidence = max(0.0, min(1.0, best_score / float(len(raw_scores) or 64)))
    return BoardObservation(
        reference_fen=reference_fen,
        fen=best_board.fen(),
        confidence=round(confidence, 4),
        signal_score=signal_score,
        matched_move=best_move or "",
        occupancy_ascii=occupancy_text,
        square_scores=tuple(raw_scores),
    )


def infer_observed_board(
    reference_fen: str,
    board_bgr: np.ndarray,
    threshold: float = 0.18,
    segmentation_mask: Optional[np.ndarray] = None,
    segmentation_weight: float = 0.7,
    min_signal_score: float = 0.02,
) -> BoardObservation:
    """Infer the most likely legal board state from occupancy and a reference FEN."""

    signal_score = board_signal_score(board_bgr)
    raw_scores: List[SquareSample] = square_probability_grid(board_bgr)
    if segmentation_mask is not None:
        segmentation_scores = segmentation_probability_grid(segmentation_mask)
        raw_scores = fuse_square_scores(
            raw_scores,
            segmentation_scores,
            segmentation_weight=segmentation_weight,
        )
    occupancy_scores = threshold_scores(raw_scores, threshold=threshold)
    occupancy_text = occupancy_ascii(occupancy_scores)

    if signal_score < min_signal_score:
        return BoardObservation(
            reference_fen=reference_fen,
            fen=reference_fen,
            confidence=0.0,
            signal_score=signal_score,
            matched_move="",
            occupancy_ascii=occupancy_text,
            square_scores=tuple(raw_scores),
        )

    if chess is None:
        return BoardObservation(
            reference_fen=reference_fen,
            fen=reference_fen,
            confidence=0.0,
            signal_score=signal_score,
            matched_move="",
            occupancy_ascii=occupancy_text,
            square_scores=tuple(raw_scores),
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
        signal_score=signal_score,
        matched_move=best_move or "",
        occupancy_ascii=occupancy_text,
        square_scores=tuple(raw_scores),
    )


def _score_candidate_with_templates(
    candidate_board: "chess.Board",
    observed_features: Dict[str, np.ndarray],
    occupancy_scores: Sequence[SquareSample],
    templates: PieceTemplates,
) -> float:
    candidate_map = _board_map_from_fen(candidate_board.fen())
    total = 0.0
    for square_name_, occ_score in occupancy_scores:
        observed = observed_features.get(square_name_)
        piece_symbol = candidate_map.get(square_name_)
        occupancy = min(max(float(occ_score), 0.0), 1.0)

        if piece_symbol and observed is not None:
            template = templates.get(piece_symbol)
            similarity = _template_similarity(observed, template) if template is not None else 0.0
            total += 0.35 * occupancy + 0.65 * similarity
        elif piece_symbol and observed is None:
            total += 0.0
        elif piece_symbol is None and observed is None:
            total += 1.0
        else:
            total += max(0.0, 1.0 - occupancy)
    return total
