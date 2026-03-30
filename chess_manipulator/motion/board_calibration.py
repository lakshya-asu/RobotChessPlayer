"""Board geometry and square-to-world mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


FILES = "abcdefgh"
RANKS = "12345678"


@dataclass(frozen=True)
class BoardCalibration:
    """Simple axis-aligned calibration for a chessboard in world coordinates."""

    origin_x: float = 0.235
    origin_y: float = -0.266
    square_size_m: float = 0.076
    board_z_m: float = 0.31
    graveyard_x: float = 0.12
    graveyard_y: float = -0.48

    @classmethod
    def from_xyz(cls, board_origin: Tuple[float, float, float], square_size_m: float) -> "BoardCalibration":
        return cls(
            origin_x=board_origin[0],
            origin_y=board_origin[1],
            board_z_m=board_origin[2],
            square_size_m=square_size_m,
        )

    @property
    def center_x(self) -> float:
        return self.origin_x + 3.5 * self.square_size_m

    @property
    def center_y(self) -> float:
        return self.origin_y + 3.5 * self.square_size_m

    def square_to_position(
        self,
        square: str,
        z_offset: float = 0.0,
        robot_side: str = "white",
    ) -> Tuple[float, float, float]:
        if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
            raise ValueError(f"Invalid square name: {square}")
        file_index = FILES.index(square[0])
        rank_index = RANKS.index(square[1])
        # Align the board so files run left-to-right (x) and ranks run
        # bottom-to-top (y), which matches the overhead perception raster
        # and the opposing white/black robot layout.
        x = self.origin_x + file_index * self.square_size_m
        y = self.origin_y + rank_index * self.square_size_m
        position = (x, y, self.board_z_m + z_offset)
        return self._mirror_position(position) if robot_side == "black" else position

    def all_square_positions(
        self,
        z_offset: float = 0.0,
        robot_side: str = "white",
    ) -> Dict[str, Tuple[float, float, float]]:
        return {
            f"{file_}{rank}": self.square_to_position(
                f"{file_}{rank}",
                z_offset=z_offset,
                robot_side=robot_side,
            )
            for file_ in FILES
            for rank in RANKS
        }

    def captured_piece_dropoff(
        self,
        slot_index: int,
        z_offset: float = 0.0,
        robot_side: str = "white",
    ) -> Tuple[float, float, float]:
        row = slot_index // 8
        col = slot_index % 8
        position = (
            self.graveyard_x - row * self.square_size_m,
            self.graveyard_y + col * (self.square_size_m / 2.0),
            self.board_z_m + z_offset,
        )
        return self._mirror_position(position) if robot_side == "black" else position

    def _mirror_position(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = position
        return (2.0 * self.center_x - x, 2.0 * self.center_y - y, z)
