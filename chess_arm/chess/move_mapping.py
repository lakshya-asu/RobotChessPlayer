from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import chess

from chess_arm.utils.transforms import BoardCalibration, square_center_world


@dataclass
class MoveWaypoints:
    """World-space waypoints for a pick-and-place corresponding to a chess move."""
    approach_from: np.ndarray
    grasp_from: np.ndarray
    lift_from: np.ndarray
    approach_to: np.ndarray
    place_to: np.ndarray
    retreat_to: np.ndarray


def make_move_waypoints(
    from_square: str,
    to_square: str,
    calib: BoardCalibration,
    approach_offset: float,
    retreat_offset: float,
) -> MoveWaypoints:
    """Create simple pick-and-place waypoints for a move."""
    z_axis = calib.z_axis_world

    from_center = square_center_world(from_square, calib)
    to_center = square_center_world(to_square, calib)

    approach_from = from_center + approach_offset * z_axis
    grasp_from = from_center
    lift_from = from_center + approach_offset * z_axis

    approach_to = to_center + approach_offset * z_axis
    place_to = to_center
    retreat_to = to_center + retreat_offset * z_axis

    return MoveWaypoints(
        approach_from=approach_from,
        grasp_from=grasp_from,
        lift_from=lift_from,
        approach_to=approach_to,
        place_to=place_to,
        retreat_to=retreat_to,
    )


def move_to_square_names(move: chess.Move) -> Tuple[str, str]:
    """
    Convert a python-chess Move into (from_square, to_square) algebraic names.
    """
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    return from_sq, to_sq


def uci_to_square_names(uci: str) -> Tuple[str, str]:
    """
    Convert a UCI string (e.g. 'e2e4') into (from_square, to_square) names.
    """
    move = chess.Move.from_uci(uci)
    return move_to_square_names(move)
