from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

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
