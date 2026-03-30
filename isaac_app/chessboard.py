"""Procedural chessboard geometry and FEN helpers for the Isaac Sim demo scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np


FILES = "abcdefgh"
RANKS = "12345678"
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@dataclass(frozen=True)
class BoardGeometry:
    """Axis-aligned board geometry that matches the ROS-side calibration."""

    origin_x: float = 0.235
    origin_y: float = -0.266
    board_z_m: float = 0.31
    square_size_m: float = 0.076
    board_thickness_m: float = 0.012

    @property
    def board_length_m(self) -> float:
        return self.square_size_m * 8.0

    @property
    def center_x(self) -> float:
        return self.origin_x + 3.5 * self.square_size_m

    @property
    def center_y(self) -> float:
        return self.origin_y + 3.5 * self.square_size_m

    @property
    def top_z_m(self) -> float:
        return self.board_z_m + self.board_thickness_m / 2.0

    def square_center(self, square: str) -> Tuple[float, float, float]:
        if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
            raise ValueError(f"Invalid square name: {square}")
        file_index = FILES.index(square[0])
        rank_index = RANKS.index(square[1])
        # Files increase along +x and ranks increase along +y so the board is
        # oriented correctly between the mirrored white/black robots.
        x = self.origin_x + file_index * self.square_size_m
        y = self.origin_y + rank_index * self.square_size_m
        return (x, y, self.top_z_m)

    def iter_squares(self) -> Iterator[Tuple[str, Tuple[float, float, float]]]:
        for file_ in FILES:
            for rank in RANKS:
                square = f"{file_}{rank}"
                yield square, self.square_center(square)

    def square_map(self) -> Dict[str, Tuple[float, float, float]]:
        return dict(self.iter_squares())

    def graveyard_position(self, slot_index: int, lateral_offset: float = 0.0) -> Tuple[float, float, float]:
        row = slot_index // 8
        col = slot_index % 8
        x = self.origin_x - (1.7 + row * 0.55) * self.square_size_m
        y = self.origin_y - 2.8 * self.square_size_m + col * lateral_offset
        return (x, y, self.top_z_m)


def square_color(square: str) -> np.ndarray:
    """Return alternating board colors for a checker pattern."""
    file_index = FILES.index(square[0])
    rank_index = RANKS.index(square[1])
    is_light = (file_index + rank_index) % 2 == 0
    return np.array([0.86, 0.78, 0.64] if is_light else [0.33, 0.22, 0.16])


def piece_color(kind: str = "white") -> np.ndarray:
    """Small color helper for visual markers or future pieces."""
    return np.array([0.94, 0.92, 0.88] if kind == "white" else [0.18, 0.16, 0.14])


def board_map_from_fen(fen: str) -> Dict[str, str]:
    """Parse a FEN string into a mapping of algebraic square to piece symbol."""
    board_part = (fen or INITIAL_FEN).split()[0]
    ranks = board_part.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN board payload: {fen}")

    pieces: Dict[str, str] = {}
    for fen_rank_index, rank_data in enumerate(ranks):
        file_index = 0
        board_rank = 8 - fen_rank_index
        for char in rank_data:
            if char.isdigit():
                file_index += int(char)
                continue
            if file_index >= 8 or char.lower() not in {"p", "r", "n", "b", "q", "k"}:
                raise ValueError(f"Invalid FEN board payload: {fen}")
            square = f"{FILES[file_index]}{board_rank}"
            pieces[square] = char
            file_index += 1
        if file_index != 8:
            raise ValueError(f"Invalid FEN board payload: {fen}")
    return pieces


def fen_from_board_map(board_map: Dict[str, str], reference_fen: str = "") -> str:
    """Serialize a square->piece map into a FEN string."""

    rows = []
    for rank in range(8, 0, -1):
        empty_count = 0
        row_parts = []
        for file_ in FILES:
            piece = board_map.get(f"{file_}{rank}", "")
            if piece:
                if empty_count:
                    row_parts.append(str(empty_count))
                    empty_count = 0
                row_parts.append(piece)
            else:
                empty_count += 1
        if empty_count:
            row_parts.append(str(empty_count))
        rows.append("".join(row_parts))

    board_part = "/".join(rows)
    fields = reference_fen.split()
    if len(fields) >= 6:
        fields[0] = board_part
        return " ".join(fields[:6])
    return f"{board_part} w - - 0 1"
