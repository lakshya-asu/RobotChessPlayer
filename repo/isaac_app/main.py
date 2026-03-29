"""Launch the standalone Isaac Sim GUI app for the chess manipulator demo."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple


def _bootstrap_pythonpath() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_bootstrap_pythonpath()

from isaacsim import SimulationApp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim chess manipulator demo app")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--square-size-m", type=float, default=0.076)
    parser.add_argument("--board-origin-x", type=float, default=0.235)
    parser.add_argument("--board-origin-y", type=float, default=-0.266)
    parser.add_argument("--board-origin-z", type=float, default=0.31)
    parser.add_argument("--command-topic", default="/isaac/command/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/isaac/joint_states")
    parser.add_argument("--status-topic", default="/isaac/status")
    parser.add_argument("--execution-result-topic", default="/isaac/execution_result")
    parser.add_argument("--board-state-topic", default="/chess/board_state")
    parser.add_argument(
        "--sync-board-state",
        action="store_true",
        help="Continuously mirror /chess/board_state into visible piece markers.",
    )
    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Isaac Sim must start before Omniverse and ROS bridge imports.
    simulation_app = SimulationApp({"renderer": args.renderer, "headless": args.headless})

    import omni.timeline
    from isaacsim.core.utils.extensions import enable_extension

    from isaac_app.chessboard import BoardGeometry
    from isaac_app.board_sync import IsaacBoardStateNode
    from isaac_app.scene import ChessIsaacScene, SceneConfig
    from isaac_app.trajectory_executor import IsaacTrajectoryExecutor

    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    import rclpy

    initialized_here = False
    try:
        if not rclpy.ok():
            rclpy.init(args=[])
            initialized_here = True
    except RuntimeError as exc:
        if "Context.init() must only be called once" not in str(exc):
            raise

    board = BoardGeometry(
        origin_x=args.board_origin_x,
        origin_y=args.board_origin_y,
        board_z_m=args.board_origin_z,
        square_size_m=args.square_size_m,
    )
    scene = ChessIsaacScene(
        SceneConfig(
            board_geometry=board,
            show_graveyard=True,
        )
    )
    franka = scene.build()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    executor = IsaacTrajectoryExecutor(
        manipulator=franka,
        command_topic=args.command_topic,
        joint_state_topic=args.joint_state_topic,
        status_topic=args.status_topic,
        execution_result_topic=args.execution_result_topic,
    )
    board_state_node = None
    if args.sync_board_state:
        board_state_node = IsaacBoardStateNode(scene=scene, board_topic=args.board_state_topic)

    try:
        while simulation_app.is_running():
            scene.world.step(render=True)
            rclpy.spin_once(executor, timeout_sec=0.0)
            if board_state_node is not None:
                rclpy.spin_once(board_state_node, timeout_sec=0.0)
                pending_fen = board_state_node.consume_pending_fen()
                if pending_fen is not None:
                    scene.sync_board_state(pending_fen)
            executor.step()
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        executor.destroy_node()
        if board_state_node is not None:
            board_state_node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
