from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_arm.utils.transforms import BoardCalibration

try:
    # Isaac Sim imports (only valid inside Isaac Sim Python)
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ImportError:  # pragma: no cover
    World = object  # type: ignore[assignment]

    def add_reference_to_stage(*_, **__) -> None:
        return None


@dataclass
class ChessBoardEnv:
    world: World
    board_prim_path: str
    pieces_prim_path: str
    calib: BoardCalibration

    @classmethod
    def from_config(
        cls,
        world: World,
        sim_config: dict,
        calib: BoardCalibration,
    ) -> "ChessBoardEnv":
        board_cfg = sim_config["board"]
        board_prim = board_cfg["prim_path"]
        pieces_prim = board_cfg["piece_set_prim_path"]
        return cls(world=world, board_prim_path=board_prim, pieces_prim_path=pieces_prim, calib=calib)

    def load_assets(self, board_usd: Path, pieces_usd: Path) -> None:
        """
        Attach board and piece USD references at configured prim paths.
        """
        add_reference_to_stage(str(board_usd), self.board_prim_path)
        add_reference_to_stage(str(pieces_usd), self.pieces_prim_path)
