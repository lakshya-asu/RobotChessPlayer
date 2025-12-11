from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from isaacsim import SimulationApp  # type: ignore[import]
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World  # type: ignore[import]

from chess_arm.chess.engine_interface import ChessEngineInterface, EngineConfig
from chess_arm.sim.digital_twin import FrankaChessDigitalTwin
from chess_arm.utils.transforms import BoardCalibration


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-binary", type=str, required=True)
    return parser.parse_args()


def main() -> None:
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

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    twin = FrankaChessDigitalTwin.create(world, sim_cfg, calib)

    engine = ChessEngineInterface(
        EngineConfig(engine_path=Path(args.engine_binary), thinking_time_s=0.5)
    )

    world.reset()

    # Play a fixed number of moves as a demo
    n_moves = 4

    while simulation_app.is_running():
        # Step world
        world.step(render=True)
        # twin.step(dt) if you wired a trajectory here

        # For simplicity, this demo does not yet trigger engine-driven moves in real time.
        # todo: query engine.best_move(), map to twin.plan_move_squares(), and execute trajectories.


        n_moves -= 1
        if n_moves <= 0:
            break

    engine.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
