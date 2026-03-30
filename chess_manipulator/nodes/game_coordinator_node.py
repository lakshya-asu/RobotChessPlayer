#!/usr/bin/env python3
"""ROS runtime coordinator for alternating white/black robot execution."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from builtin_interfaces.msg import Duration
import chess
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from chess_manipulator.coordinator import (
    GameCoordinatorError,
    GameLog,
    PlyLog,
    build_player_adapter,
    expected_fen_after_move,
    legal_moves_for_fen,
    perception_confirmation_valid,
    side_to_move_from_fen,
    write_game_log,
)
from chess_manipulator.players import PlayerAdapter, PlayerError
from chess_manipulator.sim import decode_feedback
from chess_manipulator_msgs.action import ExecuteChessMove
from chess_manipulator_msgs.msg import ExecutionCommand


class RuntimeGameCoordinatorNode(Node):
    """Run an alternating engine-vs-policy loop against white and black executors."""

    def __init__(self) -> None:
        super().__init__("game_coordinator")
        self.declare_parameter(
            "initial_fen",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        self.declare_parameter("board_state_topic", "/chess/board_state")
        self.declare_parameter("observed_fen_topic", "/perception/observed_fen")
        self.declare_parameter("fen_confidence_topic", "/perception/fen_confidence")
        self.declare_parameter("board_pose_topic", "/perception/board_pose")
        self.declare_parameter("current_fen_topic", "/game/current_fen")
        self.declare_parameter("status_topic", "/game/status")
        self.declare_parameter("white_executor_action_name", "/white/execute_move")
        self.declare_parameter("black_executor_action_name", "/black/execute_move")
        self.declare_parameter("white_planned_trajectory_topic", "/white/planned_trajectory")
        self.declare_parameter("black_planned_trajectory_topic", "/black/planned_trajectory")
        self.declare_parameter("white_execution_feedback_topic", "/white/execution_feedback")
        self.declare_parameter("black_execution_feedback_topic", "/black/execution_feedback")
        self.declare_parameter("white_backend", "stockfish")
        self.declare_parameter("black_backend", "dqn")
        self.declare_parameter("white_engine_executable", "")
        self.declare_parameter("black_engine_executable", "")
        self.declare_parameter("white_dqn_checkpoint", "")
        self.declare_parameter("black_dqn_checkpoint", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("think_time_sec", 0.1)
        self.declare_parameter("max_plies", 8)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("start_delay_sec", 1.0)
        self.declare_parameter("execution_timeout_sec", 45.0)
        self.declare_parameter("use_observed_fen", True)
        self.declare_parameter("require_perception_confirmation", False)
        self.declare_parameter("require_board_pose_confirmation", True)
        self.declare_parameter("perception_confidence_threshold", 0.65)
        self.declare_parameter("perception_verification_timeout_sec", 5.0)
        self.declare_parameter("board_pose_max_age_sec", 1.0)
        self.declare_parameter("parking_move_time_sec", 2.5)
        self.declare_parameter("white_parking_joint_positions", [1.35, -1.05, 0.0, -2.2, 0.0, 2.15, 0.8])
        self.declare_parameter("black_parking_joint_positions", [-1.35, -1.05, 0.0, -2.2, 0.0, 2.15, 0.8])
        self.declare_parameter("game_log_dir", "")

        self._initial_fen = str(self.get_parameter("initial_fen").value)
        self._latest_board_fen = self._initial_fen
        self._observed_fen = self._initial_fen
        self._accepted_fen = self._initial_fen
        self._observed_confidence = 0.0
        self._observed_fen_update_time = 0.0
        self._latest_board_fen_update_time = 0.0
        self._board_pose_update_time = 0.0
        self._execution_timeout_sec = float(self.get_parameter("execution_timeout_sec").value)
        self._perception_timeout_sec = float(
            self.get_parameter("perception_verification_timeout_sec").value
        )
        self._perception_threshold = float(
            self.get_parameter("perception_confidence_threshold").value
        )
        self._board_pose_max_age_sec = float(self.get_parameter("board_pose_max_age_sec").value)
        self._use_observed_fen = bool(self.get_parameter("use_observed_fen").value)
        self._require_perception_confirmation = bool(
            self.get_parameter("require_perception_confirmation").value
        )
        self._require_board_pose_confirmation = bool(
            self.get_parameter("require_board_pose_confirmation").value
        )
        self._max_plies = int(self.get_parameter("max_plies").value)
        self._parking_move_time_sec = float(self.get_parameter("parking_move_time_sec").value)
        self._white_parking_joint_positions = [
            float(value) for value in self.get_parameter("white_parking_joint_positions").value
        ]
        self._black_parking_joint_positions = [
            float(value) for value in self.get_parameter("black_parking_joint_positions").value
        ]
        self._feedback_state = {
            "white": {"move": "", "state": "", "message": ""},
            "black": {"move": "", "state": "", "message": ""},
        }
        self._white_backend = str(self.get_parameter("white_backend").value)
        self._black_backend = str(self.get_parameter("black_backend").value)
        self._white_checkpoint = str(self.get_parameter("white_dqn_checkpoint").value)
        self._black_checkpoint = str(self.get_parameter("black_dqn_checkpoint").value)
        self._game_log_dir = str(self.get_parameter("game_log_dir").value).strip()

        think_time_sec = float(self.get_parameter("think_time_sec").value)
        device = str(self.get_parameter("device").value)
        self._white_player = build_player_adapter(
            "white",
            str(self.get_parameter("white_backend").value),
            str(self.get_parameter("white_engine_executable").value),
            think_time_sec,
            str(self.get_parameter("white_dqn_checkpoint").value),
            device,
        )
        self._black_player = build_player_adapter(
            "black",
            str(self.get_parameter("black_backend").value),
            str(self.get_parameter("black_engine_executable").value),
            think_time_sec,
            str(self.get_parameter("black_dqn_checkpoint").value),
            device,
        )

        self._current_fen_publisher = self.create_publisher(
            String,
            str(self.get_parameter("current_fen_topic").value),
            10,
        )
        self._board_state_publisher = self.create_publisher(
            String,
            str(self.get_parameter("board_state_topic").value),
            10,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("board_state_topic").value),
            self._on_board_state,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("observed_fen_topic").value),
            self._on_observed_fen,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("fen_confidence_topic").value),
            self._on_fen_confidence,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("board_pose_topic").value),
            self._on_board_pose,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("white_execution_feedback_topic").value),
            lambda msg: self._on_execution_feedback("white", msg),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("black_execution_feedback_topic").value),
            lambda msg: self._on_execution_feedback("black", msg),
            10,
        )

        self._white_executor_action_name = str(
            self.get_parameter("white_executor_action_name").value
        )
        self._black_executor_action_name = str(
            self.get_parameter("black_executor_action_name").value
        )
        self._white_executor = ActionClient(
            self,
            ExecuteChessMove,
            self._white_executor_action_name,
        )
        self._black_executor = ActionClient(
            self,
            ExecuteChessMove,
            self._black_executor_action_name,
        )
        self._white_parking_pub = self.create_publisher(
            ExecutionCommand,
            str(self.get_parameter("white_planned_trajectory_topic").value),
            10,
        )
        self._black_parking_pub = self.create_publisher(
            ExecutionCommand,
            str(self.get_parameter("black_planned_trajectory_topic").value),
            10,
        )

        self._match_thread: Optional[threading.Thread] = None
        self._match_lock = threading.Lock()
        self._auto_start_timer = None
        if bool(self.get_parameter("auto_start").value):
            delay = max(float(self.get_parameter("start_delay_sec").value), 0.0)
            self._auto_start_timer = self.create_timer(delay, self._start_match_once)
        self._publish_board_state(self._accepted_fen)
        self._publish_current_fen(self._accepted_fen)
        self._publish_status("coordinator ready")

    def _on_board_state(self, msg: String) -> None:
        if msg.data:
            self._latest_board_fen = msg.data
            self._latest_board_fen_update_time = time.monotonic()

    def _on_observed_fen(self, msg: String) -> None:
        if msg.data:
            self._observed_fen = msg.data
            self._observed_fen_update_time = time.monotonic()

    def _on_fen_confidence(self, msg: Float32) -> None:
        self._observed_confidence = float(msg.data)

    def _on_board_pose(self, msg: PoseStamped) -> None:
        del msg
        self._board_pose_update_time = time.monotonic()

    def _on_execution_feedback(self, side: str, msg: String) -> None:
        self._feedback_state[side] = decode_feedback(msg.data)

    def _start_match_once(self) -> None:
        if self._auto_start_timer is not None:
            self._auto_start_timer.cancel()
            self._auto_start_timer = None
        self.start_match()

    def start_match(self) -> bool:
        if self._match_thread is not None and self._match_thread.is_alive():
            self.get_logger().warning("game_coordinator match already running")
            return False
        self._match_thread = threading.Thread(target=self._run_match, daemon=True)
        self._match_thread.start()
        return True

    def _run_match(self) -> None:
        with self._match_lock:
            self._publish_status("match starting")
            current_fen = self._current_source_fen()
            self._accepted_fen = current_fen
            self._publish_board_state(current_fen)
            self._publish_current_fen(current_fen)
            turn_logs: list[PlyLog] = []
            result_code = "*"
            termination_reason = "max_plies_reached"

            for ply_index in range(self._max_plies):
                side = side_to_move_from_fen(current_fen)
                player = self._player_for_side(side)
                try:
                    move = player.select_move(current_fen)
                    expected_fen = expected_fen_after_move(current_fen, move)
                except (PlayerError, GameCoordinatorError, ValueError) as exc:
                    termination_reason = "move_selection_failed"
                    self._publish_status(f"failed selecting {side} move: {exc}")
                    self._write_game_log(
                        turns=turn_logs,
                        initial_fen=self._initial_fen,
                        final_fen=current_fen,
                        result=result_code,
                        termination_reason=termination_reason,
                    )
                    return

                self._publish_status(
                    f"ply {ply_index + 1}: {side} using {player.identity.name} selected {move}"
                )
                self._park_inactive_robot(side)
                move_dispatch_time = time.monotonic()
                result = self._dispatch_move(side, move)
                if not result.success:
                    termination_reason = "execution_failed"
                    turn_logs.append(
                        PlyLog(
                            ply_index=ply_index + 1,
                            side_to_move=side,
                            player_name=player.identity.name,
                            player_backend=player.identity.backend,
                            checkpoint_path=self._checkpoint_for_side(side),
                            fen_before=current_fen,
                            legal_moves=legal_moves_for_fen(current_fen),
                            chosen_move=move,
                            expected_fen=expected_fen,
                            confirmed_fen=result.final_fen or current_fen,
                            perception_confidence=self._observed_confidence,
                            execution_success=False,
                            perception_confirmed=False,
                        )
                    )
                    self._publish_status(
                        f"ply {ply_index + 1}: {side} execution failed for {move}: {result.message}"
                    )
                    self._write_game_log(
                        turns=turn_logs,
                        initial_fen=self._initial_fen,
                        final_fen=result.final_fen or current_fen,
                        result=result_code,
                        termination_reason=termination_reason,
                    )
                    return

                confirmed_fen = self._wait_for_confirmation(
                    expected_fen,
                    result.final_fen,
                    min_update_time=move_dispatch_time,
                )
                if confirmed_fen is None:
                    termination_reason = "perception_confirmation_failed"
                    turn_logs.append(
                        PlyLog(
                            ply_index=ply_index + 1,
                            side_to_move=side,
                            player_name=player.identity.name,
                            player_backend=player.identity.backend,
                            checkpoint_path=self._checkpoint_for_side(side),
                            fen_before=current_fen,
                            legal_moves=legal_moves_for_fen(current_fen),
                            chosen_move=move,
                            expected_fen=expected_fen,
                            confirmed_fen=result.final_fen or current_fen,
                            perception_confidence=self._observed_confidence,
                            execution_success=True,
                            perception_confirmed=False,
                        )
                    )
                    self._publish_status(
                        f"ply {ply_index + 1}: {side} move {move} was not perception-confirmed"
                    )
                    self._write_game_log(
                        turns=turn_logs,
                        initial_fen=self._initial_fen,
                        final_fen=result.final_fen or current_fen,
                        result=result_code,
                        termination_reason=termination_reason,
                    )
                    return

                turn_logs.append(
                    PlyLog(
                        ply_index=ply_index + 1,
                        side_to_move=side,
                        player_name=player.identity.name,
                        player_backend=player.identity.backend,
                        checkpoint_path=self._checkpoint_for_side(side),
                        fen_before=current_fen,
                        legal_moves=legal_moves_for_fen(current_fen),
                        chosen_move=move,
                        expected_fen=expected_fen,
                        confirmed_fen=confirmed_fen,
                        perception_confidence=self._observed_confidence,
                        execution_success=True,
                        perception_confirmed=True,
                    )
                )
                current_fen = confirmed_fen
                self._accepted_fen = confirmed_fen
                self._publish_board_state(confirmed_fen)
                self._publish_current_fen(confirmed_fen)
                self._publish_status(
                    f"ply {ply_index + 1}: {side} executed {move}, confirmed fen {confirmed_fen}"
                )
                result_code, termination_reason = self._result_for_fen(current_fen)
                if termination_reason == "game_over":
                    self._publish_status(
                        f"match complete by game over after {ply_index + 1} plies: result {result_code}"
                    )
                    self._write_game_log(
                        turns=turn_logs,
                        initial_fen=self._initial_fen,
                        final_fen=current_fen,
                        result=result_code,
                        termination_reason=termination_reason,
                    )
                    return

            self._publish_status(f"match complete after {self._max_plies} plies")
            self._write_game_log(
                turns=turn_logs,
                initial_fen=self._initial_fen,
                final_fen=current_fen,
                result=result_code,
                termination_reason=termination_reason,
            )

    def _current_source_fen(self) -> str:
        pose_is_fresh = (
            not self._require_board_pose_confirmation
            or (time.monotonic() - self._board_pose_update_time) <= self._board_pose_max_age_sec
        )
        if (
            self._use_observed_fen
            and pose_is_fresh
            and self._observed_fen
            and self._observed_confidence >= self._perception_threshold
        ):
            return self._observed_fen
        if self._latest_board_fen:
            return self._latest_board_fen
        return self._accepted_fen or self._initial_fen

    def _player_for_side(self, side: str) -> PlayerAdapter:
        return self._white_player if side == "white" else self._black_player

    def _executor_for_side(self, side: str) -> ActionClient:
        return self._white_executor if side == "white" else self._black_executor

    def _parking_publisher_for_side(self, side: str):
        return self._white_parking_pub if side == "white" else self._black_parking_pub

    def _parking_joint_positions_for_side(self, side: str):
        return (
            self._white_parking_joint_positions
            if side == "white"
            else self._black_parking_joint_positions
        )

    def _park_inactive_robot(self, active_side: str) -> None:
        inactive_side = "black" if active_side == "white" else "white"
        move_id = f"__park__:{inactive_side}:{int(time.monotonic() * 1000)}"
        command = ExecutionCommand()
        command.move_id = move_id
        command.trajectory = self._parking_trajectory(inactive_side)
        self._feedback_state[inactive_side] = {"move": "", "state": "", "message": ""}
        self._parking_publisher_for_side(inactive_side).publish(command)
        deadline = time.monotonic() + self._execution_timeout_sec
        while time.monotonic() < deadline:
            payload = self._feedback_state[inactive_side]
            if payload.get("move") == move_id and payload.get("state") in {"completed", "failed"}:
                if payload.get("state") != "completed":
                    raise RuntimeError(
                        f"{inactive_side} parking command failed: {payload.get('message')}"
                    )
                return
            time.sleep(0.05)
        raise RuntimeError(f"Timed out parking the inactive {inactive_side} robot.")

    def _parking_trajectory(self, side: str) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        point = JointTrajectoryPoint()
        point.positions = list(self._parking_joint_positions_for_side(side))
        point.time_from_start = Duration(
            sec=int(self._parking_move_time_sec),
            nanosec=int((self._parking_move_time_sec % 1.0) * 1e9),
        )
        trajectory.points.append(point)
        return trajectory

    def _dispatch_move(self, side: str, move: str) -> ExecuteChessMove.Result:
        executor = self._executor_for_side(side)
        action_name = (
            self._white_executor_action_name if side == "white" else self._black_executor_action_name
        )
        if not executor.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{side} executor is unavailable on {action_name}")

        goal = ExecuteChessMove.Goal()
        goal.uci_move = move
        send_future = executor.send_goal_async(goal)
        deadline = time.monotonic() + self._execution_timeout_sec
        while not send_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not send_future.done():
            raise RuntimeError(f"Timed out dispatching {move} to {side} executor")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"{side} executor rejected the move goal {move}")

        result_future = goal_handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not result_future.done():
            raise RuntimeError(f"Timed out waiting for {side} executor result on {move}")
        return result_future.result().result

    def _wait_for_confirmation(
        self,
        expected_fen: str,
        fallback_fen: str,
        min_update_time: float,
    ) -> Optional[str]:
        deadline = time.monotonic() + self._perception_timeout_sec
        while time.monotonic() < deadline:
            if perception_confirmation_valid(
                expected_fen=expected_fen,
                observed_fen=self._observed_fen,
                observed_confidence=self._observed_confidence,
                confidence_threshold=self._perception_threshold,
                observed_update_time=self._observed_fen_update_time,
                board_pose_update_time=self._board_pose_update_time
                if self._require_board_pose_confirmation
                else min_update_time,
                min_update_time=min_update_time,
            ):
                return self._observed_fen
            if (
                not self._require_perception_confirmation
                and self._latest_board_fen == expected_fen
                and self._latest_board_fen_update_time >= min_update_time
            ):
                return self._latest_board_fen
            time.sleep(0.1)

        if not self._require_perception_confirmation and fallback_fen == expected_fen:
            return fallback_fen
        return None

    def _publish_current_fen(self, fen: str) -> None:
        self._current_fen_publisher.publish(String(data=fen))

    def _publish_board_state(self, fen: str) -> None:
        self._board_state_publisher.publish(String(data=fen))

    def _publish_status(self, text: str) -> None:
        self._status_publisher.publish(String(data=text))
        self.get_logger().info(text)

    def _result_for_fen(self, fen: str) -> tuple[str, str]:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            return board.result(claim_draw=True), "game_over"
        return "*", "max_plies_reached"

    def _checkpoint_for_side(self, side: str) -> str:
        return self._white_checkpoint if side == "white" else self._black_checkpoint

    def _write_game_log(
        self,
        turns: list[PlyLog],
        initial_fen: str,
        final_fen: str,
        result: str,
        termination_reason: str,
    ) -> None:
        if not self._game_log_dir:
            return
        log = GameLog(
            schema_version=1,
            initial_fen=initial_fen,
            final_fen=final_fen,
            result=result,
            termination_reason=termination_reason,
            white_backend=self._white_backend,
            black_backend=self._black_backend,
            white_checkpoint=self._white_checkpoint,
            black_checkpoint=self._black_checkpoint,
            max_plies=self._max_plies,
            turns=turns,
            metadata={
                "observed_confidence": self._observed_confidence,
                "timestamp_sec": time.time(),
            },
        )
        stem = f"game_{int(time.time())}"
        path = write_game_log(log, self._game_log_dir, stem)
        self.get_logger().info(f"wrote game log to {path}")


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = RuntimeGameCoordinatorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
