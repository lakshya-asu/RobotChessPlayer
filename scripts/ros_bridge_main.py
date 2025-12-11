from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from chess_arm.sim.digital_twin import FrankaChessDigitalTwin
from chess_arm.sim.board_env import ChessBoardEnv
from chess_arm.utils.transforms import BoardCalibration
from chess_arm.utils.logging_utils import configure_logging
from chess_arm.ros.ros_bridge import RosBridgeConfig, FrankaRosBridge

# todo: adjust SimulationApp import if Isaac Sim version exposes it via omni.isaac.kit instead
from isaacsim import SimulationApp  # type: ignore[import]

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World  # type: ignore[import]

import rclpy


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    configure_logging()

    root = Path(__file__).resolve().parents[1]

    sim_cfg = load_yaml(root / "config" / "sim_params.yaml")
    board_cfg_raw = load_yaml(root / "config" / "board_config.yaml")
    ros_cfg_raw = load_yaml(root / "config" / "ros_topics.yaml")

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

    # Board environment and optional assets
    board_env = ChessBoardEnv.from_config(world=world, sim_config=sim_cfg, calib=calib)

    assets_root = root / "assets" / "usd"
    board_usd = assets_root / "chess_board.usd"
    pieces_usd = assets_root / "chess_piece_set.usd"
    if board_usd.exists() and pieces_usd.exists():
        board_env.load_assets(board_usd=board_usd, pieces_usd=pieces_usd)

    # Franka digital twin
    twin = FrankaChessDigitalTwin.create(world=world, config=sim_cfg, calib=calib)

    # ROS2 bridge configuration from YAML
    ros_config = RosBridgeConfig(
        joint_state_topic=str(ros_cfg_raw["joint_state_topic"]),
        joint_trajectory_topic=str(ros_cfg_raw["joint_trajectory_topic"]),
        joint_names=list(ros_cfg_raw["joint_names"]),
    )

    rclpy.init()
    bridge_node = FrankaRosBridge(twin=twin, config=ros_config)

    world.reset()

    try:
        while simulation_app.is_running():
            # Handle ROS callbacks (subscriptions, timers)
            rclpy.spin_once(bridge_node, timeout_sec=0.0)

            # Step Isaac world and execute any active trajectories
            world.step(render=True)
            twin.step(physics_dt)
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
