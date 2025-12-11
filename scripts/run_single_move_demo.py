from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import chess

from chess_arm.sim.digital_twin import FrankaChessDigitalTwin
from chess_arm.sim.board_env import ChessBoardEnv
from chess_arm.utils.transforms import BoardCalibration
from chess_arm.utils.logging_utils import configure_logging
from chess_arm.chess.move_mapping import move_to_square_names

# todo: adjust SimulationApp import if Isaac Sim version exposes it via omni.isaac.kit instead
from isaacsim import SimulationApp  # type: ignore[import]

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World  # type: ignore[import]


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single chess move with the Franka digital twin in Isaac Sim.",
    )
    parser.add_argument(
        "--move",
        type=str,
        default="e2e4",
        help="UCI move string to execute (e.g. e2e4, g1f3).",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    root = Path(__file__).resolve().parents[1]

    sim_cfg = load_yaml(root / "config" / "sim_params.yaml")
    board_cfg_raw = load_yaml(root / "config" / "board_config.yaml")

    calib = BoardCalibration(
        origin_world=np.array(board_cfg_raw["origin_world"], dtype=float),
        x_axis_world=np.array(board_cfg_raw["x_axis_world"], dtype=float),
        y_axis_world=np.array(board_cfg_raw["y_axis_world"], dtype=float),
        square_size=float(board_cfg_raw["square_size"]),
        piece_height=float(board_cfg_raw["piece_height"]),
    )

    physics_dt = float(sim_cfg.get("time_step", 0.01))

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt)
    world.scene.add_default_ground_plane()

    # Board environment and assets
    board_env = ChessBoardEnv.from_config(world=world, sim_config=sim_cfg, calib=calib)

    assets_root = root / "assets" / "usd"
    board_usd = assets_root / "chess_board.usd"
    pieces_usd = assets_root / "chess_piece_set.usd"
    if board_usd.exists() and pieces_usd.exists():
        board_env.load_assets(board_usd=board_usd, pieces_usd=pieces_usd)

    # Franka digital twin
    twin = FrankaChessDigitalTwin.create(world=world, config=sim_cfg, calib=calib)

    world.reset()

    # Parse UCI move and map to algebraic square names
    move = chess.Move.from_uci(args.move)
    from_square, to_square = move_to_square_names(move)

    # Plan move (now with IK-based joint trajectory when available)
    twin.plan_move_squares(from_square=from_square, to_square=to_square)

    # Run sim until trajectory is done, then a few extra frames
    steps_after_done = 100

    while simulation_app.is_running():
        world.step(render=True)
        twin.step(physics_dt)

        if not twin.has_active_trajectory():
            steps_after_done -= 1
            if steps_after_done <= 0:
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
