from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


@dataclass
class BoardCalibration:
    """Calibration of the chessboard in the world frame."""
    origin_world: np.ndarray      # [3]
    x_axis_world: np.ndarray      # [3] direction from a-file to h-file
    y_axis_world: np.ndarray      # [3] direction from rank 1 to rank 8
    square_size: float
    piece_height: float

    def __post_init__(self) -> None:
        self.x_axis_world = normalize(self.x_axis_world)
        self.y_axis_world = normalize(self.y_axis_world)
        # z-axis as cross product (right-hand rule)
        self.z_axis_world = normalize(np.cross(self.x_axis_world, self.y_axis_world))


def square_to_indices(square: str) -> Tuple[int, int]:
    """Convert algebraic square (e.g. 'e4') to (file_idx, rank_idx)."""
    if len(square) != 2:
        raise ValueError(f"Invalid square: {square}")
    file_char, rank_char = square[0], square[1]
    file_idx = ord(file_char.lower()) - ord("a")
    rank_idx = int(rank_char) - 1
    if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
        raise ValueError(f"Square out of range: {square}")
    return file_idx, rank_idx


def indices_to_square(file_idx: int, rank_idx: int) -> str:
    """Convert 0-based indices to algebraic square name."""
    if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
        raise ValueError("Indices out of range")
    return f"{chr(ord('a') + file_idx)}{rank_idx + 1}"


def square_center_world(square: str, calib: BoardCalibration) -> np.ndarray:
    """Return world position of the center of a given square."""
    file_idx, rank_idx = square_to_indices(square)
    u = (file_idx + 0.5) * calib.square_size
    v = (rank_idx + 0.5) * calib.square_size
    w = calib.piece_height
    return (
        calib.origin_world
        + u * calib.x_axis_world
        + v * calib.y_axis_world
        + w * calib.z_axis_world
    )
