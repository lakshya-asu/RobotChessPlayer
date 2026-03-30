"""Pure layout helpers for the Isaac chess scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .chessboard import BoardGeometry

ROBOT_EDGE_CLEARANCE_M = 0.18


@dataclass(frozen=True)
class RobotPlacement:
    """Pose and prim-path metadata for one Panda in the Isaac stage."""

    role: str
    prim_path: str
    world_position: Tuple[float, float, float]
    yaw_deg: float


def default_dual_robot_placements(
    board_geometry: BoardGeometry | None = None,
) -> tuple[RobotPlacement, RobotPlacement]:
    """Place two Panda robots on the opposing sides of the board."""
    board = board_geometry or BoardGeometry()
    offset_y = board.board_length_m / 2.0 + ROBOT_EDGE_CLEARANCE_M
    center_x = board.center_x
    center_y = board.center_y
    return (
        RobotPlacement(
            role="white",
            prim_path="/World/WhiteFranka",
            world_position=(center_x, center_y - offset_y, 0.0),
            yaw_deg=90.0,
        ),
        RobotPlacement(
            role="black",
            prim_path="/World/BlackFranka",
            world_position=(center_x, center_y + offset_y, 0.0),
            yaw_deg=-90.0,
        ),
    )
