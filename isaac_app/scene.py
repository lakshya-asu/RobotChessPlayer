"""Isaac Sim scene assembly for the chess manipulator demo."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import os
from pathlib import Path
from typing import Optional

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from .chessboard import (
    BoardGeometry,
    board_map_from_fen,
    fen_from_board_map,
    piece_color,
    square_color,
)
from .layout import RobotPlacement, default_dual_robot_placements
from .compat import (
    FixedCuboid,
    ParallelGripper,
    SingleManipulator,
    VisualCapsule,
    VisualCuboid,
    VisualCylinder,
    VisualSphere,
    World,
    add_reference_to_stage,
    enable_extension,
    get_assets_root_path,
    get_prim_at_path,
    rotations,
    viewports,
)


FRANKA_STAGE_PATH = "/World/Franka"
FRANKA_LOCAL_URDF_CANDIDATES = (
    "extscache/omni.importer.urdf-1.14.1+106.0.0.lx64.r.cp310/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
    "extscache/omni.importer.urdf-1.14.1+106.0.0.lx64.r.cp310/data/urdf/robots/franka_description/robots/panda_arm.urdf",
    "exts/omni.isaac.ocs2/data/franka/urdf/panda.urdf",
)
FRANKA_USD_CANDIDATES = (
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
)
GENERATED_FRANKA_RELATIVE_PATH = "third_party/generated_assets/franka_panda.usd"
GENERATED_PIECE_ASSET_DIR = "third_party/generated_assets/chess_pieces"
PIECE_SOURCE_FILES = {
    "p": ("STL/pawn.stl",),
    "r": ("STL/rook.stl",),
    "n": ("STL/knight.stl",),
    "b": ("STL/bishop.stl",),
    "q": ("STL/queen.stl",),
    "k": ("STL/king-body.stl", "STL/king-cross.stl"),
}
PIECE_SCALE_BY_TYPE = {
    "p": 0.00042,
    "r": 0.00042,
    "n": 0.00042,
    "b": 0.00042,
    "q": 0.00042,
    "k": 0.00042,
}


@dataclass
class SceneConfig:
    """Visual configuration for the Isaac GUI scene."""

    board_geometry: BoardGeometry = field(default_factory=BoardGeometry)
    robot_placements: tuple[RobotPlacement, ...] | None = None
    camera_eye: tuple[float, float, float] | None = None
    camera_target: tuple[float, float, float] | None = None
    show_graveyard: bool = True


@dataclass
class PieceTransport:
    role: str
    move_id: str
    source_square: str
    target_square: str
    moving_prim: str
    moving_symbol: str
    final_symbol: str
    start_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    duration_sec: float
    capture_prim: str | None = None
    capture_square: str = ""
    capture_symbol: str = ""
    capture_start_position: tuple[float, float, float] | None = None
    capture_target_position: tuple[float, float, float] | None = None
    rook_prim: str | None = None
    rook_symbol: str = ""
    rook_start_position: tuple[float, float, float] | None = None
    rook_target_position: tuple[float, float, float] | None = None
    moving_attached: bool = False
    moving_released: bool = False
    release_position: tuple[float, float, float] | None = None
    failure_reason: str = ""


@dataclass(frozen=True)
class TransportResult:
    success: bool
    message: str = ""


class ChessIsaacScene:
    """Procedural chessboard, Panda loading, and camera framing."""

    def __init__(self, config: Optional[SceneConfig] = None) -> None:
        self._config = config or SceneConfig()
        self._world: Optional[World] = None
        self._robots: dict[str, SingleManipulator] = {}
        self._piece_pool: dict[str, list[str]] = {}
        self._piece_symbols_by_prim: dict[str, str] = {}
        self._placements_by_role: dict[str, RobotPlacement] = {}
        self._piece_mesh_cache: dict[str, Path] = {}
        self._active_piece_transports: dict[str, PieceTransport] = {}
        self._graveyard_counts = {"white": 0, "black": 0}
        self._robot_prim_paths: dict[str, str] = {}

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("Scene has not been built yet.")
        return self._world

    @property
    def franka(self) -> SingleManipulator:
        return self.robot("white")

    @property
    def robots(self) -> dict[str, SingleManipulator]:
        if not self._robots:
            raise RuntimeError("No robots have been loaded yet.")
        return self._robots

    def robot(self, role: str) -> SingleManipulator:
        robot = self._robots.get(role)
        if robot is None:
            raise RuntimeError(f"Robot {role!r} has not been loaded yet.")
        return robot

    @property
    def board_geometry(self) -> BoardGeometry:
        return self._config.board_geometry

    def build(self) -> dict[str, SingleManipulator]:
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        board = self._config.board_geometry
        placements = self._config.robot_placements or default_dual_robot_placements(board)
        self._add_camera(board)
        self._add_board(board)
        self._add_robot_side_pads(placements)
        for placement in placements:
            self._placements_by_role[placement.role] = placement
            self._robots[placement.role] = self._add_franka(placement)
        self._create_piece_pool()
        self.sync_board_state("")

        self._world.reset()
        for role, placement in self._placements_by_role.items():
            self._set_robot_pose(self._robots[role], placement)
        return self.robots

    def _add_camera(self, board: BoardGeometry) -> None:
        eye = np.array(
            self._config.camera_eye
            or (board.center_x + 0.90, board.center_y + 0.85, board.top_z_m + 0.80)
        )
        target = np.array(
            self._config.camera_target or (board.center_x, board.center_y, board.top_z_m + 0.02)
        )
        viewports.set_camera_view(
            eye=eye,
            target=target,
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
            self._add_graveyard_box(board)

    def _add_robot_side_pads(self, placements: tuple[RobotPlacement, ...]) -> None:
        for placement in placements:
            is_black = placement.role == "black"
            color = np.array([0.10, 0.10, 0.10]) if is_black else np.array([0.82, 0.82, 0.82])
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/{placement.role.capitalize()}SidePad",
                    name=f"{placement.role}_side_pad",
                    position=np.array(
                        [
                            placement.world_position[0],
                            placement.world_position[1],
                            -0.001,
                        ]
                    ),
                    size=1.0,
                    scale=np.array([0.38, 0.32, 0.002]),
                    color=color,
                )
            )

    def sync_board_state(self, fen: str) -> None:
        """Reposition simple piece markers from a FEN payload."""
        board_state = board_map_from_fen(fen)
        placements: dict[str, list[str]] = {}
        graveyard_counts = {"white": 0, "black": 0}
        for square, piece_symbol in board_state.items():
            placements.setdefault(piece_symbol, []).append(square)

        for piece_symbol, prim_paths in self._piece_pool.items():
            squares = sorted(placements.get(piece_symbol, []))
            for index, prim_path in enumerate(prim_paths):
                if index < len(squares):
                    position = self._piece_position(squares[index], piece_symbol)
                else:
                    side = "white" if piece_symbol.isupper() else "black"
                    slot_index = graveyard_counts[side]
                    graveyard_counts[side] += 1
                    position = self._graveyard_position(piece_symbol, slot_index)
                self._set_prim_translation(prim_path, position)
        self._graveyard_counts = graveyard_counts

    def _create_piece_pool(self) -> None:
        counts: dict[str, int] = {}
        for piece_symbol in board_map_from_fen("").values():
            counts[piece_symbol] = counts.get(piece_symbol, 0) + 1

        for piece_symbol, count in counts.items():
            prim_paths: list[str] = []
            for index in range(count):
                prim_path = self._create_piece_visual(piece_symbol, index)
                prim_paths.append(prim_path)
                self._piece_symbols_by_prim[prim_path] = piece_symbol
            self._piece_pool[piece_symbol] = prim_paths

    def observed_board_map(self) -> dict[str, str]:
        """Infer a board map from the current visible piece locations."""

        square_map = self._config.board_geometry.square_map()
        assignment_candidates: dict[str, tuple[str, float]] = {}
        max_distance = self._config.board_geometry.square_size_m * 0.45

        for prim_path, piece_symbol in self._piece_symbols_by_prim.items():
            position = self._get_prim_translation(prim_path)
            if position is None:
                continue
            square_name = None
            best_distance = None
            for candidate_square, square_position in square_map.items():
                distance = float(
                    np.linalg.norm(
                        np.array(position[:2], dtype=float)
                        - np.array(square_position[:2], dtype=float)
                    )
                )
                if best_distance is None or distance < best_distance:
                    square_name = candidate_square
                    best_distance = distance
            if square_name is None or best_distance is None or best_distance > max_distance:
                continue

            incumbent = assignment_candidates.get(square_name)
            if incumbent is None or best_distance < incumbent[1]:
                assignment_candidates[square_name] = (piece_symbol, best_distance)

        return {
            square_name: piece_symbol
            for square_name, (piece_symbol, _distance) in assignment_candidates.items()
        }

    def observed_fen(self, reference_fen: str = "") -> str:
        """Return a FEN derived from the currently visible scene layout."""

        return fen_from_board_map(self.observed_board_map(), reference_fen=reference_fen)

    def _create_piece_visual(self, piece_symbol: str, index: int) -> str:
        prim_path = f"/World/ChessPieces/{piece_symbol}_{index}"
        position = self._graveyard_position(piece_symbol, index)
        self._create_piece_visual_at_path(piece_symbol, prim_path, position)
        return prim_path

    def _create_piece_visual_at_path(
        self,
        piece_symbol: str,
        prim_path: str,
        position: tuple[float, float, float],
    ) -> None:
        board = self._config.board_geometry
        color = piece_color("white" if piece_symbol.isupper() else "black")
        piece_type = piece_symbol.lower()
        if self._try_create_piece_mesh(piece_symbol, prim_path, position, color):
            return
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
                radius=board.square_size_m * 0.11,
                height=board.square_size_m * 0.48,
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

    def _add_graveyard_box(self, board: BoardGeometry) -> None:
        center = np.array([board.origin_x - 0.20, board.origin_y - 0.12, board.board_z_m - 0.004])
        outer_length = 0.18
        outer_width = 0.18
        wall_height = 0.028
        wall_thickness = 0.008
        base_thickness = 0.008
        dark = piece_color("black")

        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/ChessBoard/Graveyard/Base",
                name="graveyard_base",
                position=center,
                size=1.0,
                scale=np.array([outer_length, outer_width, base_thickness]),
                color=dark,
            )
        )

        wall_z = center[2] + (base_thickness / 2.0) + (wall_height / 2.0)
        half_length = outer_length / 2.0
        half_width = outer_width / 2.0

        wall_specs = (
            ("North", np.array([center[0], center[1] + half_width - wall_thickness / 2.0, wall_z]), np.array([outer_length, wall_thickness, wall_height])),
            ("South", np.array([center[0], center[1] - half_width + wall_thickness / 2.0, wall_z]), np.array([outer_length, wall_thickness, wall_height])),
            ("East", np.array([center[0] + half_length - wall_thickness / 2.0, center[1], wall_z]), np.array([wall_thickness, outer_width, wall_height])),
            ("West", np.array([center[0] - half_length + wall_thickness / 2.0, center[1], wall_z]), np.array([wall_thickness, outer_width, wall_height])),
        )

        for name, position, scale in wall_specs:
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/ChessBoard/Graveyard/{name}",
                    name=f"graveyard_{name.lower()}",
                    position=position,
                    size=1.0,
                    scale=scale,
                    color=dark,
                )
            )

    def _try_create_piece_mesh(
        self,
        piece_symbol: str,
        prim_path: str,
        position: tuple[float, float, float],
        color: np.ndarray,
    ) -> bool:
        piece_type = piece_symbol.lower()
        usd_paths = self._ensure_piece_mesh_assets(piece_type)
        if not usd_paths:
            return False

        root_xform = UsdGeom.Xform.Define(self.world.stage, prim_path)
        root_prim = root_xform.GetPrim()
        translate_op = root_xform.AddTranslateOp()
        scale_op = root_xform.AddScaleOp()
        rotate_op = root_xform.AddRotateXYZOp()
        translate_op.Set(Gf.Vec3d(*position))
        scale = PIECE_SCALE_BY_TYPE.get(piece_type, 0.00042)
        scale_op.Set(Gf.Vec3f(scale, scale, scale))
        rotate_op.Set(Gf.Vec3f(90.0, 0.0, 180.0 if piece_symbol.islower() else 0.0))

        for mesh_index, usd_path in enumerate(usd_paths):
            child_path = f"{prim_path}/mesh_{mesh_index}"
            child_prim = self.world.stage.DefinePrim(child_path, "Xform")
            child_prim.GetReferences().AddReference(str(usd_path))
            if piece_type == "k" and mesh_index == 1:
                child_xform = UsdGeom.Xformable(child_prim)
                child_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 42.0))
            self._tint_piece_mesh(child_prim, color)
        return root_prim.IsValid()

    def _tint_piece_mesh(self, root_prim, color: np.ndarray) -> None:
        display_color = [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr(display_color)

    def _ensure_piece_mesh_assets(self, piece_type: str) -> list[Path]:
        source_rel_paths = PIECE_SOURCE_FILES.get(piece_type)
        if not source_rel_paths:
            return []
        generated_paths: list[Path] = []
        for source_rel_path in source_rel_paths:
            source_path = self._workspace_root() / source_rel_path
            output_path = (
                self._workspace_root()
                / GENERATED_PIECE_ASSET_DIR
                / (Path(source_rel_path).stem + ".usd")
            )
            if not output_path.is_file():
                try:
                    self._convert_piece_asset(source_path, output_path)
                except Exception as exc:
                    print(f"[IsaacPieceAssets] Failed to convert {source_path.name}: {exc}")
                    return []
            if output_path.is_file():
                generated_paths.append(output_path)
        return generated_paths

    def _convert_piece_asset(self, source_path: Path, output_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing piece asset: {source_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enable_extension("omni.kit.asset_converter")
        import omni.kit.asset_converter as asset_converter

        async def _convert() -> bool:
            converter_manager = asset_converter.get_instance()
            context = asset_converter.AssetConverterContext()
            context.ignore_materials = True
            context.use_meter_as_world_unit = True
            context.create_world_as_default_root_prim = True
            task = converter_manager.create_converter_task(
                str(source_path),
                str(output_path),
                None,
                context,
            )
            return await task.wait_until_finished()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("Isaac asset conversion requires a non-running asyncio loop.")
        success = loop.run_until_complete(_convert())
        if not success or not output_path.is_file():
            raise RuntimeError(f"Could not convert {source_path.name} to USD")

    def _piece_position(self, square: str, piece_symbol: str) -> tuple[float, float, float]:
        x, y, _ = self._config.board_geometry.square_center(square)
        height = self._piece_height(piece_symbol)
        return (x, y, self._config.board_geometry.top_z_m + height)

    def _parse_move_id(self, move_id: str) -> tuple[str, str, str]:
        source_square = move_id[:2]
        target_square = move_id[2:4]
        promotion = move_id[4].lower() if len(move_id) >= 5 and move_id[4].isalpha() else ""
        return source_square, target_square, promotion

    def _promotion_symbol(self, moving_symbol: str, promotion: str) -> str:
        if not promotion:
            return moving_symbol
        return promotion.upper() if moving_symbol.isupper() else promotion.lower()

    def _en_passant_capture_square(
        self,
        moving_symbol: str,
        source_square: str,
        target_square: str,
        board_map: dict[str, str],
    ) -> str:
        if moving_symbol.lower() != "p":
            return ""
        if source_square[0] == target_square[0]:
            return ""
        if target_square in board_map:
            return ""
        capture_square = f"{target_square[0]}{source_square[1]}"
        captured_symbol = board_map.get(capture_square, "")
        if not captured_symbol or captured_symbol.isupper() == moving_symbol.isupper():
            return ""
        return capture_square

    def begin_piece_transport(self, role: str, move_id: str, duration_sec: float) -> bool:
        if len(move_id) < 4 or move_id.startswith("__"):
            return True
        board_map = self.observed_board_map()
        source_square, target_square, promotion = self._parse_move_id(move_id)
        moving_prim = self._find_piece_prim_at_square(source_square)
        if moving_prim is None:
            return False
        moving_symbol = self._piece_symbols_by_prim.get(moving_prim, "")
        final_symbol = self._promotion_symbol(moving_symbol, promotion)
        start_position = self._get_prim_translation(moving_prim)
        if start_position is None:
            return False
        transport = PieceTransport(
            role=role,
            move_id=move_id,
            source_square=source_square,
            target_square=target_square,
            moving_prim=moving_prim,
            moving_symbol=moving_symbol,
            final_symbol=final_symbol,
            start_position=start_position,
            target_position=self._piece_position(target_square, final_symbol),
            duration_sec=max(duration_sec, 0.1),
        )

        capture_square = target_square
        captured_symbol = board_map.get(capture_square, "")
        captured_prim = self._find_piece_prim_at_square(capture_square)
        if not captured_symbol or captured_prim == moving_prim:
            capture_square = self._en_passant_capture_square(
                moving_symbol,
                source_square,
                target_square,
                board_map,
            )
            if capture_square:
                captured_symbol = board_map.get(capture_square, "")
                captured_prim = self._find_piece_prim_at_square(capture_square)
        if captured_symbol and captured_prim and captured_prim != moving_prim:
            graveyard_side = role
            slot_index = self._graveyard_counts[graveyard_side]
            self._graveyard_counts[graveyard_side] += 1
            transport.capture_square = capture_square
            transport.capture_prim = captured_prim
            transport.capture_symbol = captured_symbol
            transport.capture_start_position = self._get_prim_translation(captured_prim)
            transport.capture_target_position = self._graveyard_position(
                captured_symbol,
                slot_index,
                graveyard_side=graveyard_side,
            )

        if moving_symbol.lower() == "k" and abs(ord(target_square[0]) - ord(source_square[0])) == 2:
            rook_source = "h1" if target_square == "g1" else "a1"
            rook_target = "f1" if target_square == "g1" else "d1"
            if target_square in {"g8", "c8"}:
                rook_source = "h8" if target_square == "g8" else "a8"
                rook_target = "f8" if target_square == "g8" else "d8"
            rook_prim = self._find_piece_prim_at_square(rook_source)
            if rook_prim is not None:
                rook_symbol = self._piece_symbols_by_prim.get(rook_prim, "")
                transport.rook_prim = rook_prim
                transport.rook_symbol = rook_symbol
                transport.rook_start_position = self._get_prim_translation(rook_prim)
                transport.rook_target_position = self._piece_position(rook_target, rook_symbol)

        self._active_piece_transports[role] = transport
        return True

    def step_piece_transport(self, role: str, elapsed_sec: float) -> None:
        transport = self._active_piece_transports.get(role)
        if transport is None:
            return
        alpha = min(max(elapsed_sec / transport.duration_sec, 0.0), 1.0)
        lift = self._config.board_geometry.square_size_m * 0.9

        if transport.capture_prim and transport.capture_start_position and transport.capture_target_position:
            capture_alpha = min(alpha / 0.45, 1.0)
            self._set_prim_translation(
                transport.capture_prim,
                self._arc_position(
                    transport.capture_start_position,
                    transport.capture_target_position,
                    capture_alpha,
                    lift * 0.7,
                ),
            )
            move_alpha = max((alpha - 0.35) / 0.65, 0.0)
        else:
            move_alpha = alpha

        grip_position = self._gripper_carry_position(role, transport.moving_symbol)
        attach_distance = self._config.board_geometry.square_size_m * 0.26
        if (
            not transport.moving_attached
            and not transport.moving_released
            and grip_position is not None
            and alpha >= 0.10
            and self._distance_xy(grip_position, transport.start_position) <= attach_distance
        ):
            transport.moving_attached = True
        elif (
            not transport.failure_reason
            and not transport.moving_attached
            and not transport.moving_released
            and alpha >= 0.45
        ):
            transport.failure_reason = f"failed to attach piece for {transport.move_id}"

        if transport.moving_attached and not transport.moving_released and grip_position is not None:
            self._set_prim_translation(transport.moving_prim, grip_position)
            if alpha >= 0.82 and self._distance_xy(grip_position, transport.target_position) <= attach_distance:
                transport.moving_attached = False
                transport.moving_released = True
                transport.release_position = grip_position
        elif (
            not transport.failure_reason
            and not transport.moving_released
            and alpha >= 0.98
        ):
            transport.failure_reason = f"failed to release piece for {transport.move_id}"
        elif transport.moving_released and transport.release_position is not None:
            release_alpha = min(max((alpha - 0.82) / 0.18, 0.0), 1.0)
            self._set_prim_translation(
                transport.moving_prim,
                self._arc_position(
                    transport.release_position,
                    transport.target_position,
                    release_alpha,
                    lift * 0.15,
                ),
            )
        else:
            self._set_prim_translation(
                transport.moving_prim,
                self._arc_position(
                    transport.start_position,
                    transport.target_position,
                    move_alpha,
                    lift,
                ),
            )

        if transport.rook_prim and transport.rook_start_position and transport.rook_target_position:
            rook_alpha = max((alpha - 0.55) / 0.45, 0.0)
            self._set_prim_translation(
                transport.rook_prim,
                self._arc_position(
                    transport.rook_start_position,
                    transport.rook_target_position,
                    rook_alpha,
                    lift * 0.55,
                ),
            )

    def finish_piece_transport(self, role: str) -> TransportResult:
        transport = self._active_piece_transports.pop(role, None)
        if transport is None:
            return TransportResult(success=True)
        if (
            not transport.failure_reason
            and not transport.moving_attached
            and not transport.moving_released
            and not transport.move_id.startswith("__")
        ):
            transport.failure_reason = f"piece transport never attached for {transport.move_id}"
        if transport.failure_reason:
            self._restore_piece_transport(transport)
            return TransportResult(success=False, message=transport.failure_reason)
        if transport.capture_prim and transport.capture_target_position:
            self._set_prim_translation(transport.capture_prim, transport.capture_target_position)
        if transport.final_symbol != transport.moving_symbol:
            self._replace_piece_visual(
                transport.moving_prim,
                transport.final_symbol,
                transport.target_position,
            )
        else:
            self._set_prim_translation(transport.moving_prim, transport.target_position)
        if transport.rook_prim and transport.rook_target_position:
            self._set_prim_translation(transport.rook_prim, transport.rook_target_position)
        return TransportResult(success=True, message=f"piece transport complete for {transport.move_id}")

    def _restore_piece_transport(self, transport: PieceTransport) -> None:
        self._set_prim_translation(transport.moving_prim, transport.start_position)
        if transport.capture_prim and transport.capture_start_position:
            self._set_prim_translation(transport.capture_prim, transport.capture_start_position)
        if transport.rook_prim and transport.rook_start_position:
            self._set_prim_translation(transport.rook_prim, transport.rook_start_position)

    def _reassign_piece_symbol(self, prim_path: str, new_symbol: str) -> None:
        old_symbol = self._piece_symbols_by_prim.get(prim_path, "")
        if old_symbol == new_symbol:
            return
        if old_symbol:
            old_bucket = self._piece_pool.get(old_symbol, [])
            if prim_path in old_bucket:
                old_bucket.remove(prim_path)
        self._piece_pool.setdefault(new_symbol, [])
        if prim_path not in self._piece_pool[new_symbol]:
            self._piece_pool[new_symbol].append(prim_path)
        self._piece_symbols_by_prim[prim_path] = new_symbol

    def _replace_piece_visual(
        self,
        prim_path: str,
        new_symbol: str,
        position: tuple[float, float, float],
    ) -> None:
        if self._world is not None:
            stage = self.world.stage
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)
        self._create_piece_visual_at_path(new_symbol, prim_path, position)
        self._reassign_piece_symbol(prim_path, new_symbol)

    def _arc_position(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        alpha: float,
        lift_height: float,
    ) -> tuple[float, float, float]:
        alpha = min(max(alpha, 0.0), 1.0)
        x = (1.0 - alpha) * start[0] + alpha * end[0]
        y = (1.0 - alpha) * start[1] + alpha * end[1]
        z = (1.0 - alpha) * start[2] + alpha * end[2] + (4.0 * alpha * (1.0 - alpha) * lift_height)
        return (x, y, z)

    def _distance_xy(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return float(np.linalg.norm(np.array(a[:2], dtype=float) - np.array(b[:2], dtype=float)))

    def _find_piece_prim_at_square(self, square: str) -> str | None:
        target = np.array(self._config.board_geometry.square_center(square)[:2], dtype=float)
        max_distance = self._config.board_geometry.square_size_m * 0.45
        best_prim = None
        best_distance = None
        for prim_path in self._piece_symbols_by_prim:
            position = self._get_prim_translation(prim_path)
            if position is None:
                continue
            distance = float(np.linalg.norm(np.array(position[:2], dtype=float) - target))
            if distance > max_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_prim = prim_path
                best_distance = distance
        return best_prim

    def _gripper_carry_position(self, role: str, piece_symbol: str) -> tuple[float, float, float] | None:
        robot_prim_path = self._robot_prim_paths.get(role)
        if not robot_prim_path:
            return None
        for suffix, z_offset in (
            ("panda_rightfinger", -self._config.board_geometry.square_size_m * 0.02),
            ("panda_hand", -self._config.board_geometry.square_size_m * 0.10),
        ):
            world_position = self._get_prim_world_translation(f"{robot_prim_path}/{suffix}")
            if world_position is None:
                continue
            return (
                world_position[0],
                world_position[1],
                world_position[2] + z_offset + self._piece_height(piece_symbol),
            )
        return None

    def _graveyard_position(
        self,
        piece_symbol: str,
        slot_index: int,
        graveyard_side: str | None = None,
    ) -> tuple[float, float, float]:
        if graveyard_side is None:
            graveyard_side = "white" if piece_symbol.isupper() else "black"
        color_offset = -0.06 if graveyard_side == "white" else 0.06
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

    def _get_prim_translation(self, prim_path: str) -> tuple[float, float, float] | None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            return None
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                value = op.Get()
                return (float(value[0]), float(value[1]), float(value[2]))
        return None

    def _get_prim_world_translation(self, prim_path: str) -> tuple[float, float, float] | None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            return None
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        return (float(translation[0]), float(translation[1]), float(translation[2]))

    def _add_franka(self, placement: RobotPlacement) -> SingleManipulator:
        robot_prim_path = self._resolve_franka_prim(placement.prim_path)
        self._robot_prim_paths[placement.role] = robot_prim_path
        robot = get_prim_at_path(robot_prim_path)
        if robot.IsValid() and robot.HasVariantSets():
            variant_sets = robot.GetVariantSets()
            if variant_sets.HasVariantSet("Gripper"):
                variant_sets.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            if variant_sets.HasVariantSet("Mesh"):
                variant_sets.GetVariantSet("Mesh").SetVariantSelection("Quality")

        finger_prim_path = f"{robot_prim_path}/panda_rightfinger"

        gripper = ParallelGripper(
            end_effector_prim_path=finger_prim_path,
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )
        manipulator = self._world.scene.add(
            SingleManipulator(
                prim_path=robot_prim_path,
                name=f"franka_{placement.role}",
                end_effector_prim_path=finger_prim_path,
                gripper=gripper,
            )
        )
        self._set_robot_pose(manipulator, placement)
        manipulator.gripper.set_default_state(manipulator.gripper.joint_opened_positions)
        return manipulator

    def _set_robot_pose(self, manipulator: SingleManipulator, placement: RobotPlacement) -> None:
        manipulator.set_world_pose(
            position=np.array(placement.world_position),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(0, 0, 1), placement.yaw_deg)
            ),
        )

    def _resolve_franka_prim(self, target_prim_path: str) -> str:
        prefer_remote = os.environ.get("ISAAC_PREFER_REMOTE_FRANKA", "0").lower() not in {
            "0",
            "false",
            "no",
        }
        attempts = (
            (self._add_remote_franka_reference, "Isaac USD reference"),
            (self._add_local_franka_generated_reference, "generated local Franka USD"),
            (self._import_local_franka_urdf, "local URDF import"),
        )
        if not prefer_remote:
            attempts = (
                (self._add_local_franka_generated_reference, "generated local Franka USD"),
                (self._import_local_franka_urdf, "local URDF import"),
                (self._add_remote_franka_reference, "Isaac USD reference"),
            )

        errors: list[str] = []
        for loader, label in attempts:
            try:
                resolved = loader(target_prim_path)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                continue
            if resolved and self._franka_prim_is_valid(resolved):
                return resolved
            errors.append(f"{label}: prim resolved but Franka links were incomplete at {target_prim_path}")

        joined_errors = "; ".join(errors) or "no Franka loader succeeded"
        raise RuntimeError(f"Could not resolve Franka robot at {target_prim_path}: {joined_errors}")

    def _franka_prim_is_valid(self, prim_path: str) -> bool:
        robot = get_prim_at_path(prim_path)
        finger = get_prim_at_path(f"{prim_path}/panda_rightfinger")
        return robot.IsValid() and finger.IsValid()

    def _workspace_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _generated_franka_usd_path(self) -> Path:
        return self._workspace_root() / GENERATED_FRANKA_RELATIVE_PATH

    def _add_local_franka_generated_reference(self, prim_path: str) -> str | None:
        usd_path = self._ensure_local_franka_usd()
        if usd_path is None:
            return None
        add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
        return prim_path

    def _ensure_local_franka_usd(self) -> Path | None:
        generated_usd = self._generated_franka_usd_path()
        if generated_usd.is_file():
            return generated_usd

        isaac_root = os.environ.get("ISAAC_ROOT")
        if not isaac_root:
            return None

        import omni.kit.commands

        generated_usd.parent.mkdir(parents=True, exist_ok=True)

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Could not create an Isaac URDF import configuration for Franka.")
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.create_physics_scene = False
        import_config.import_inertia_tensor = True

        for relative_path in FRANKA_LOCAL_URDF_CANDIDATES:
            urdf_path = os.path.join(isaac_root, relative_path)
            if not os.path.isfile(urdf_path):
                continue
            status, _ = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                dest_path=str(generated_usd),
                get_articulation_root=False,
            )
            if status and generated_usd.is_file():
                return generated_usd
        return None

    def _import_local_franka_urdf(self, target_prim_path: str) -> str | None:
        isaac_root = os.environ.get("ISAAC_ROOT")
        if not isaac_root:
            return None

        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Could not create an Isaac URDF import configuration for Franka.")
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.make_default_prim = False
        import_config.create_physics_scene = False
        import_config.import_inertia_tensor = True

        for relative_path in FRANKA_LOCAL_URDF_CANDIDATES:
            urdf_path = os.path.join(isaac_root, relative_path)
            if not os.path.isfile(urdf_path):
                continue
            status, imported_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                get_articulation_root=True,
            )
            if status and imported_path:
                # Isaac 4.2 returns the articulation root (for example ``/panda/root_joint``)
                # even though the visual/link hierarchy and gripper live under the robot base prim.
                if imported_path.endswith("/root_joint"):
                    imported_path = imported_path.rsplit("/root_joint", 1)[0]
                if imported_path != target_prim_path:
                    move_status, _ = omni.kit.commands.execute(
                        "MovePrim",
                        path_from=imported_path,
                        path_to=target_prim_path,
                        keep_world_transform=False,
                    )
                    if not move_status:
                        raise RuntimeError(
                            f"Imported Franka prim could not be moved from {imported_path} to {target_prim_path}"
                        )
                return target_prim_path
        return None

    def _add_remote_franka_reference(self, prim_path: str) -> str:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not locate the Isaac Sim assets root.")

        for candidate in FRANKA_USD_CANDIDATES:
            try:
                add_reference_to_stage(
                    usd_path=assets_root_path + candidate,
                    prim_path=prim_path,
                )
                return prim_path
            except Exception:
                continue
        raise RuntimeError(
            "Could not resolve a Franka Panda robot from either the local URDF importer or the active Isaac assets root."
        )
