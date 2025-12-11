from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import chess

from chess_arm.utils.transforms import BoardCalibration, square_center_world

try:
    # Isaac Sim camera APIs
    from omni.isaac.sensor import Camera  # type: ignore[import]
except ImportError:  # pragma: no cover
    Camera = object  # type: ignore[assignment]


@dataclass
class PiecePose:
    square: str
    position_world: np.ndarray


class SimulatedChessPerception:
    """
    Simulated perception using a virtual camera and simple synthetic piece poses.

    The default implementation does not perform real image processing. Instead,
    it can generate idealized piece poses from a FEN string and estimate a
    translation-only correction to the board calibration.
    """

    def __init__(
        self,
        calib: BoardCalibration,
        camera_prim_path: str,
        camera_name: str = "chess_cam",
    ) -> None:
        self.calib = calib
        self.camera_prim_path = camera_prim_path
        self.camera_name = camera_name

        self._camera: Optional[Camera] = None  # type: ignore[assignment]
        self._camera_config: Optional[dict] = None
        self._last_frame = None

        if Camera is not object:
            self._camera = Camera(prim_path=self.camera_prim_path, name=self.camera_name)
            # todo: configure camera resolution, fov and pose using Isaac Sim APIs

    @classmethod
    def from_config(
        cls,
        calib: BoardCalibration,
        camera_cfg: dict,
    ) -> "SimulatedChessPerception":
        prim_path = camera_cfg["prim_path"]
        name = camera_cfg.get("name", "chess_cam")
        instance = cls(calib=calib, camera_prim_path=prim_path, camera_name=name)
        instance._camera_config = camera_cfg
        instance._apply_camera_config()
        return instance

    def _apply_camera_config(self) -> None:
        """
        Apply configuration from camera_config.yaml to the underlying Isaac camera.
        """
        if self._camera is None or self._camera_config is None:
            return
        # todo: set resolution, fov and pose based on self._camera_config

    # --------------------------------------------------------------------- #
    # Capture and pose estimation
    # --------------------------------------------------------------------- #

    def capture(self) -> None:
        """
        Capture a frame from the virtual camera.

        This method stores the last captured frame if the camera backend is available.
        """
        if self._camera is None:
            return
        # todo: store rendered RGB / depth / segmentation if required by perception stack

    def estimate_piece_poses(self) -> List[PiecePose]:
        """
        Estimate piece poses from the current frame.

        This implementation returns synthetic poses for the standard initial
        chess position using the current board calibration.
        """
        board = chess.Board()  # standard initial position
        piece_poses: List[PiecePose] = []

        for square_index in chess.SQUARES:
            piece = board.piece_at(square_index)
            if piece is None:
                continue
            square_name = chess.square_name(square_index)
            pos = square_center_world(square_name, self.calib)
            piece_poses.append(PiecePose(square=square_name, position_world=pos))

        return piece_poses

    def correct_board_calibration(
        self,
        observed_poses: List[PiecePose],
    ) -> BoardCalibration:
        """
        Estimate a simple translation offset for the board origin using observed poses.

        Assumes:
        - observed poses are labeled with correct square names
        - board rotation and scale remain fixed

        Returns:
            A new BoardCalibration with updated origin_world and unchanged axes/scale.
        """
        if not observed_poses:
            return self.calib

        deltas = []
        for p in observed_poses:
            expected = square_center_world(p.square, self.calib)
            deltas.append(p.position_world - expected)

        mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)

        new_origin = self.calib.origin_world + mean_delta
        return BoardCalibration(
            origin_world=new_origin,
            x_axis_world=self.calib.x_axis_world,
            y_axis_world=self.calib.y_axis_world,
            square_size=self.calib.square_size,
            piece_height=self.calib.piece_height,
        )
