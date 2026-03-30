"""Launch the standalone Isaac Sim GUI app for the chess manipulator demo."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Tuple


def _bootstrap_pythonpath() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_bootstrap_pythonpath()

from isaac_app.channels import RobotTopicSet, default_dual_robot_topic_sets
from isaac_app.compat import ISAAC_API_FLAVOR, SimulationApp, enable_ros2_bridge


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim chess manipulator demo app")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--perception-image-size-px", type=int, default=640)
    parser.add_argument("--perception-period-sec", type=float, default=0.35)
    parser.add_argument("--square-size-m", type=float, default=0.076)
    parser.add_argument("--board-origin-x", type=float, default=0.235)
    parser.add_argument("--board-origin-y", type=float, default=-0.266)
    parser.add_argument("--board-origin-z", type=float, default=0.31)
    parser.add_argument("--board-state-topic", default="/chess/board_state")
    parser.add_argument(
        "--sync-board-state",
        action="store_true",
        help="Continuously mirror /chess/board_state into visible piece markers.",
    )
    for role in ("white", "black"):
        parser.add_argument(
            f"--{role}-command-topic",
            default=f"/{role}/command/joint_trajectory",
        )
        parser.add_argument(f"--{role}-joint-state-topic", default=f"/{role}/joint_states")
        parser.add_argument(f"--{role}-status-topic", default=f"/{role}/status")
        parser.add_argument(
            f"--{role}-execution-result-topic",
            default=f"/{role}/execution_result",
        )
    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Isaac Sim must start before Omniverse and ROS bridge imports.
    simulation_app = SimulationApp({"renderer": args.renderer, "headless": args.headless})

    import omni.timeline

    from isaac_app.chessboard import BoardGeometry
    from isaac_app.board_sync import IsaacBoardStateNode
    from isaac_app.fen_hud import IsaacFenHud
    from isaac_app.perception_publisher import IsaacPerceptionPublisher
    from isaac_app.scene import ChessIsaacScene, SceneConfig
    from isaac_app.trajectory_executor import IsaacTrajectoryExecutor

    enable_ros2_bridge()
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
    robots = scene.build()
    topic_sets: dict[str, RobotTopicSet] = default_dual_robot_topic_sets()
    for role in topic_sets:
        topic_sets[role] = RobotTopicSet(
            role=role,
            command_topic=str(getattr(args, f"{role}_command_topic")),
            joint_state_topic=str(getattr(args, f"{role}_joint_state_topic")),
            status_topic=str(getattr(args, f"{role}_status_topic")),
            execution_result_topic=str(getattr(args, f"{role}_execution_result_topic")),
        )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    executors = [
        IsaacTrajectoryExecutor(manipulator=robots[role], channels=topic_sets[role], scene=scene)
        for role in ("white", "black")
        if role in robots
    ]
    for executor in executors:
        executor.get_logger().info(
            f"Isaac compatibility mode: {ISAAC_API_FLAVOR}; command_topic={executor.channels.command_topic}"
        )
    board_state_node = None
    fen_hud = None
    perception_publisher = IsaacPerceptionPublisher(
        board_state_provider=scene.observed_board_map,
        board_topic=args.board_state_topic,
        image_size_px=args.perception_image_size_px,
        publish_period_sec=args.perception_period_sec,
    )
    if not args.headless:
        fen_hud = IsaacFenHud()
    if args.sync_board_state:
        board_state_node = IsaacBoardStateNode(scene=scene, board_topic=args.board_state_topic)

    # Let the native app fully settle before the control loop starts. In headless
    # mode, entering the loop too early can make the app look "not running" even
    # though the stage and ROS bridge are still coming online.
    for _ in range(3):
        simulation_app.update()

    if executors:
        executors[0].get_logger().info(
            f"Isaac app entering control loop (headless={args.headless}, running={simulation_app.is_running()})."
        )

    try:
        first_step_logged = False
        while not simulation_app.is_exiting():
            scene.world.step(render=True)
            if not first_step_logged:
                if executors:
                    executors[0].get_logger().info("Isaac app completed the first world step.")
                first_step_logged = True
            for executor in executors:
                rclpy.spin_once(executor, timeout_sec=0.0)
                executor.step()
            rclpy.spin_once(perception_publisher, timeout_sec=0.0)
            if fen_hud is not None:
                rclpy.spin_once(fen_hud, timeout_sec=0.0)
            if board_state_node is not None:
                rclpy.spin_once(board_state_node, timeout_sec=0.0)
                pending_fen = board_state_node.consume_pending_fen()
                if pending_fen is not None:
                    scene.sync_board_state(pending_fen)
    except KeyboardInterrupt:
        pass
    except Exception:  # pragma: no cover - runtime safety
        traceback.print_exc()
        raise
    finally:
        timeline.stop()
        for executor in executors:
            executor.destroy_node()
        perception_publisher.destroy_node()
        if fen_hud is not None:
            fen_hud.destroy_node()
        if board_state_node is not None:
            board_state_node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
        try:
            simulation_app.close()
        except Exception:  # pragma: no cover - Isaac shutdown is occasionally noisy
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
