#!/usr/bin/env python3
"""Reusable robot-side move executor with homing + MoveIt planning."""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence, Tuple

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from chess_manipulator_msgs.action import ExecuteChessMove
from chess_manipulator_msgs.msg import ExecutionCommand
from chess_manipulator.chess.board import BoardError, ChessBoardManager
from chess_manipulator.motion import BoardCalibration, MoveItTrajectoryPlanner, TrajectoryPlanner
from chess_manipulator.sim import decode_feedback


DEFAULT_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


class RobotExecutorNode(Node):
    """Execute one chess move for a single robot namespace."""

    def __init__(self) -> None:
        super().__init__("robot_executor")
        self.declare_parameter("execute_action_name", "/white/execute_move")
        self.declare_parameter("planned_trajectory_topic", "/white/planned_trajectory")
        self.declare_parameter("execution_feedback_topic", "/white/execution_feedback")
        self.declare_parameter("board_state_topic", "/chess/board_state")
        self.declare_parameter("planner_backend", "moveit")
        self.declare_parameter("square_size_m", 0.076)
        self.declare_parameter("board_origin", [0.235, -0.266, 0.31])
        self.declare_parameter("hover_height_m", 0.12)
        self.declare_parameter("pickup_height_m", 0.015)
        self.declare_parameter("move_time_sec", 1.0)
        self.declare_parameter("execution_timeout_sec", 20.0)
        self.declare_parameter("home_joint_positions", DEFAULT_HOME)
        self.declare_parameter("home_move_time_sec", 2.0)

        board_origin = tuple(self.get_parameter("board_origin").value)
        calibration = BoardCalibration.from_xyz(
            board_origin=board_origin,
            square_size_m=float(self.get_parameter("square_size_m").value),
        )
        planner_backend = str(self.get_parameter("planner_backend").value)
        planner_kwargs = dict(
            calibration=calibration,
            hover_height_m=float(self.get_parameter("hover_height_m").value),
            pickup_height_m=float(self.get_parameter("pickup_height_m").value),
            stage_time_sec=float(self.get_parameter("move_time_sec").value),
        )
        if planner_backend == "moveit":
            self._planner = MoveItTrajectoryPlanner(self, **planner_kwargs)
        else:
            self._planner = TrajectoryPlanner(**planner_kwargs)

        self._home_joint_positions = [
            float(value) for value in self.get_parameter("home_joint_positions").value
        ]
        self._home_move_time_sec = float(self.get_parameter("home_move_time_sec").value)
        self._execution_timeout_sec = float(self.get_parameter("execution_timeout_sec").value)
        self._planned_topic = str(self.get_parameter("planned_trajectory_topic").value)
        self._feedback_topic = str(self.get_parameter("execution_feedback_topic").value)

        self._board = ChessBoardManager()
        self._feedback_event = threading.Event()
        self._feedback_lock = threading.Lock()
        self._last_feedback = {}

        self._trajectory_publisher = self.create_publisher(
            ExecutionCommand,
            self._planned_topic,
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
            self._feedback_topic,
            self._on_execution_feedback,
            10,
        )
        self._action_server = ActionServer(
            self,
            ExecuteChessMove,
            str(self.get_parameter("execute_action_name").value),
            execute_callback=self._execute_move,
        )

    def _on_board_state(self, msg: String) -> None:
        try:
            self._board.set_fen(msg.data)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.get_logger().error(f"Failed to sync board state into robot executor: {exc}")

    def _on_execution_feedback(self, msg: String) -> None:
        payload = decode_feedback(msg.data)
        with self._feedback_lock:
            self._last_feedback = payload
        if payload.get("state") in {"completed", "failed"}:
            self._feedback_event.set()

    def _execute_move(self, goal_handle) -> ExecuteChessMove.Result:
        move = goal_handle.request.uci_move
        feedback = ExecuteChessMove.Feedback()
        try:
            description = self._board.describe_move(move)
            capture_slot = self._board.captured_piece_count()

            feedback.stage = "homing"
            feedback.progress = 0.1
            goal_handle.publish_feedback(feedback)
            self._dispatch_trajectory(
                move_id=f"{move}:home_start",
                trajectory=self._home_trajectory(),
            )

            feedback.stage = "planning"
            feedback.progress = 0.35
            goal_handle.publish_feedback(feedback)
            if isinstance(self._planner, MoveItTrajectoryPlanner):
                plan = self._planner.plan_move_from_positions(
                    description,
                    start_positions=self._home_joint_positions,
                    capture_slot=capture_slot,
                )
            else:
                plan = self._planner.plan_move(description, capture_slot=capture_slot)

            feedback.stage = "executing"
            feedback.progress = 0.7
            goal_handle.publish_feedback(feedback)
            self._dispatch_trajectory(
                move_id=move,
                trajectory=plan.to_joint_trajectory(self._planner.stage_time_sec),
            )

            feedback.stage = "returning_home"
            feedback.progress = 0.9
            goal_handle.publish_feedback(feedback)
            self._dispatch_trajectory(
                move_id=f"{move}:home_end",
                trajectory=self._home_trajectory(),
            )

            feedback.stage = "complete"
            feedback.progress = 1.0
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
            return ExecuteChessMove.Result(
                success=True,
                final_fen=self._board.fen,
                message=f"Robot executor completed {move}",
            )
        except (BoardError, RuntimeError, ValueError) as exc:
            self.get_logger().error(str(exc))
            goal_handle.abort()
            return ExecuteChessMove.Result(
                success=False,
                final_fen=self._board.fen,
                message=str(exc),
            )

    def _home_trajectory(self) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(self._home_joint_positions)
        point.time_from_start = Duration(
            sec=int(self._home_move_time_sec),
            nanosec=int((self._home_move_time_sec % 1.0) * 1e9),
        )
        trajectory.points.append(point)
        return trajectory

    def _dispatch_trajectory(self, move_id: str, trajectory: JointTrajectory) -> None:
        command = ExecutionCommand()
        command.move_id = move_id
        command.trajectory = trajectory
        self._feedback_event.clear()
        with self._feedback_lock:
            self._last_feedback = {}
        self._trajectory_publisher.publish(command)
        feedback = self._wait_for_feedback(move_id)
        if feedback.get("state") != "completed":
            raise RuntimeError(feedback.get("message") or f"Execution failed for {move_id}")

    def _wait_for_feedback(self, move_id: str) -> dict:
        deadline = time.monotonic() + self._execution_timeout_sec
        while time.monotonic() < deadline:
            with self._feedback_lock:
                payload = dict(self._last_feedback)
            if payload.get("move") == move_id and payload.get("state") in {"completed", "failed"}:
                return payload
            remaining = max(deadline - time.monotonic(), 0.05)
            self._feedback_event.wait(timeout=min(0.2, remaining))
            self._feedback_event.clear()
        return {
            "state": "failed",
            "move": move_id,
            "message": f"Timed out waiting for feedback on {move_id}",
        }


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = RobotExecutorNode()
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
