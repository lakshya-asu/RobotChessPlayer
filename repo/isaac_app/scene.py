"""Isaac Sim scene assembly for the chess manipulator demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualCapsule, VisualCuboid, VisualCylinder, VisualSphere
from isaacsim.core.utils import rotations, viewports
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

from .chessboard import BoardGeometry, board_map_from_fen, piece_color, square_color


FRANKA_STAGE_PATH = "/World/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


@dataclass
class SceneConfig:
    """Visual configuration for the Isaac GUI scene."""

    board_geometry: BoardGeometry = field(default_factory=BoardGeometry)
    camera_eye: tuple[float, float, float] = (1.05, 0.95, 0.85)
    camera_target: tuple[float, float, float] = (0.31, -0.01, 0.34)
    show_graveyard: bool = True


class ChessIsaacScene:
    """Procedural chessboard, Panda loading, and camera framing."""

    def __init__(self, config: Optional[SceneConfig] = None) -> None:
        self._config = config or SceneConfig()
        self._world: Optional[World] = None
        self._franka: Optional[SingleManipulator] = None
        self._piece_pool: dict[str, list[str]] = {}

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("Scene has not been built yet.")
        return self._world

    @property
    def franka(self) -> SingleManipulator:
        if self._franka is None:
            raise RuntimeError("Franka has not been loaded yet.")
        return self._franka

    @property
    def board_geometry(self) -> BoardGeometry:
        return self._config.board_geometry

    def build(self) -> SingleManipulator:
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        board = self._config.board_geometry
        self._add_camera()
        self._add_board(board)
        self._add_franka()
        self._create_piece_pool()
        self.sync_board_state("")

        self._world.reset()
        return self.franka

    def _add_camera(self) -> None:
        viewports.set_camera_view(
            eye=np.array(self._config.camera_eye),
            target=np.array(self._config.camera_target),
        )

    def _add_board(self, board: BoardGeometry) -> None:
        base_height = board.board_thickness_m
        base = FixedCuboid(
            prim_path="/World/ChessBoard/Base",
            name="board_base",
            position=np.array([board.center_x, board.center_y, board.board_z_m - base_height / 2.0]),
            size=1.0,
            scale=np.array([board.board_length_m, board.board_length_m, base_height]),
            color=np.array([0.17, 0.12, 0.09]),
        )
        self._world.scene.add(base)

        for square, position in board.iter_squares():
            square_prim = VisualCuboid(
                prim_path=f"/World/ChessBoard/{square}",
                name=f"square_{square}",
                position=np.array(position),
                size=1.0,
                scale=np.array([board.square_size_m * 0.98, board.square_size_m * 0.98, 0.003]),
                color=square_color(square),
            )
            self._world.scene.add(square_prim)

        if self._config.show_graveyard:
            graveyard = FixedCuboid(
                prim_path="/World/ChessBoard/Graveyard",
                name="graveyard",
                position=np.array([
                    board.origin_x - 0.20,
                    board.origin_y - 0.12,
                    board.board_z_m - 0.004,
                ]),
                size=1.0,
                scale=np.array([0.18, 0.18, 0.008]),
                color=piece_color("black"),
            )
            self._world.scene.add(graveyard)

    def sync_board_state(self, fen: str) -> None:
        """Reposition simple piece markers from a FEN payload."""
        board_state = board_map_from_fen(fen)
        placements: dict[str, list[str]] = {}
        for square, piece_symbol in board_state.items():
            placements.setdefault(piece_symbol, []).append(square)

        for piece_symbol, prim_paths in self._piece_pool.items():
            squares = sorted(placements.get(piece_symbol, []))
            for index, prim_path in enumerate(prim_paths):
                if index < len(squares):
                    position = self._piece_position(squares[index], piece_symbol)
                else:
                    position = self._graveyard_position(piece_symbol, index)
                self._set_prim_translation(prim_path, position)

    def _create_piece_pool(self) -> None:
        counts: dict[str, int] = {}
        for piece_symbol in board_map_from_fen("").values():
            counts[piece_symbol] = counts.get(piece_symbol, 0) + 1

        for piece_symbol, count in counts.items():
            prim_paths: list[str] = []
            for index in range(count):
                prim_path = self._create_piece_visual(piece_symbol, index)
                prim_paths.append(prim_path)
            self._piece_pool[piece_symbol] = prim_paths

    def _create_piece_visual(self, piece_symbol: str, index: int) -> str:
        board = self._config.board_geometry
        color = piece_color("white" if piece_symbol.isupper() else "black")
        prim_path = f"/World/ChessPieces/{piece_symbol}_{index}"
        piece_type = piece_symbol.lower()
        position = self._graveyard_position(piece_symbol, index)
        if piece_type == "p":
            VisualSphere(
                prim_path=prim_path,
                radius=board.square_size_m * 0.18,
                position=np.array(position),
                color=color,
            )
        elif piece_type == "r":
            VisualCylinder(
                prim_path=prim_path,
                radius=board.square_size_m * 0.17,
                height=board.square_size_m * 0.42,
                position=np.array(position),
                color=color,
            )
        elif piece_type == "n":
            VisualCapsule(
                prim_path=prim_path,
                radius=board.square_size_m * 0.12,
                height=board.square_size_m * 0.42,
                position=np.array(position),
                orientation=np.array([0.9238795, 0.0, 0.3826834, 0.0]),
                color=color,
            )
        elif piece_type == "b":
            VisualCapsule(
                prim_path=prim_path,
                radius=board.square_size_m * 0.12,
                height=board.square_size_m * 0.58,
                position=np.array(position),
                color=color,
            )
        elif piece_type == "q":
            VisualCapsule(
                prim_path=prim_path,
                radius=board.square_size_m * 0.14,
                height=board.square_size_m * 0.72,
                position=np.array(position),
                color=color,
            )
        else:
            VisualCapsule(
                prim_path=prim_path,
                radius=board.square_size_m * 0.15,
                height=board.square_size_m * 0.82,
                position=np.array(position),
                color=color,
            )
        return prim_path

    def _piece_position(self, square: str, piece_symbol: str) -> tuple[float, float, float]:
        x, y, _ = self._config.board_geometry.square_center(square)
        height = self._piece_height(piece_symbol)
        return (x, y, self._config.board_geometry.top_z_m + height)

    def _graveyard_position(self, piece_symbol: str, slot_index: int) -> tuple[float, float, float]:
        color_offset = 0.06 if piece_symbol.isupper() else -0.06
        x, y, _ = self._config.board_geometry.graveyard_position(
            slot_index,
            lateral_offset=self._config.board_geometry.square_size_m * 0.28,
        )
        height = self._piece_height(piece_symbol)
        return (x, y + color_offset, self._config.board_geometry.top_z_m + height)

    def _piece_height(self, piece_symbol: str) -> float:
        square_size_m = self._config.board_geometry.square_size_m
        piece_type = piece_symbol.lower()
        if piece_type == "p":
            return square_size_m * 0.18
        if piece_type == "r":
            return square_size_m * 0.21
        if piece_type == "n":
            return square_size_m * 0.24
        if piece_type == "b":
            return square_size_m * 0.29
        if piece_type == "q":
            return square_size_m * 0.36
        return square_size_m * 0.41

    def _set_prim_translation(self, prim_path: str, position: tuple[float, float, float]) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*position))

    def _add_franka(self) -> None:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not locate the Isaac Sim assets root.")

        add_reference_to_stage(
            usd_path=assets_root_path + FRANKA_USD_PATH,
            prim_path=FRANKA_STAGE_PATH,
        )
        robot = get_prim_at_path(FRANKA_STAGE_PATH)
        if robot.IsValid():
            robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

        gripper = ParallelGripper(
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )
        self._franka = self._world.scene.add(
            SingleManipulator(
                prim_path=FRANKA_STAGE_PATH,
                name="franka_panda",
                end_effector_prim_path="/World/Franka/panda_rightfinger",
                gripper=gripper,
            )
        )
        self._franka.set_world_pose(
            position=np.array([0.0, -0.64, 0.0]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
        )
        self._franka.gripper.set_default_state(self._franka.gripper.joint_opened_positions)
