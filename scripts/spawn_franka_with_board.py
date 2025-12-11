from __future__ import annotations

from pathlib import Path

import yaml

# todo: adjust SimulationApp import if Isaac Sim version exposes it via omni.isaac.kit instead
from isaacsim import SimulationApp  # type: ignore[import]

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World  # type: ignore[import]
from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore[import]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sim_cfg = yaml.safe_load((root / "config" / "sim_params.yaml").read_text())

    physics_dt = float(sim_cfg.get("time_step", 0.01))

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt)
    world.scene.add_default_ground_plane()

    assets_root = root / "assets" / "usd"
    add_reference_to_stage(str(assets_root / "franka_panda.usd"), sim_cfg["franka"]["prim_path"])
    add_reference_to_stage(str(assets_root / "chess_board.usd"), sim_cfg["board"]["prim_path"])
    add_reference_to_stage(str(assets_root / "chess_piece_set.usd"), sim_cfg["board"]["piece_set_prim_path"])

    world.reset()

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
