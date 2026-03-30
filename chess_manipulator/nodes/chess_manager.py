#!/usr/bin/env python3
"""ROS node exposing chess state, engine service, and execution action."""

from __future__ import annotations

import time
from typing import Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from chess_manipulator_msgs.action import ExecuteChessMove
from chess_manipulator_msgs.srv import GetBestMove
from chess_manipulator.chess.board import BoardError, ChessBoardManager
from chess_manipulator.chess.engine import ChessEngineAdapter, EngineConfig, EngineError


class ChessManagerNode(Node):
    """Central coordinator for board state, planning, and engine inference."""

    def __init__(self) -> None:
        super().__init__("chess_manager")
        self.declare_parameter("engine_backend", "stockfish")
        self.declare_parameter("engine_executable", "")
        self.declare_parameter("engine_think_time_sec", 0.25)
        self.declare_parameter("square_size_m", 0.076)
        self.declare_parameter("board_origin", [0.235, -0.266, 0.31])
        self.declare_parameter("hover_height_m", 0.12)
        self.declare_parameter("pickup_height_m", 0.015)
        self.declare_parameter("move_time_sec", 1.0)
        self.declare_parameter("execution_timeout_sec", 45.0)
        self.declare_parameter("white_executor_action_name", "/white/execute_move")
        self.declare_parameter("black_executor_action_name", "/black/execute_move")

        self._board = ChessBoardManager()
        self._engine = ChessEngineAdapter(
            EngineConfig(
                backend=str(self.get_parameter("engine_backend").value),
                executable=str(self.get_parameter("engine_executable").value or "") or None,
                think_time_sec=float(self.get_parameter("engine_think_time_sec").value),
            )
        )

        self._board_publisher = self.create_publisher(String, "/chess/board_state", 10)
        self._status_publisher = self.create_publisher(String, "/chess/execution_status", 10)
        self._white_executor = ActionClient(
            self,
            ExecuteChessMove,
            str(self.get_parameter("white_executor_action_name").value),
        )
        self._black_executor = ActionClient(
            self,
            ExecuteChessMove,
            str(self.get_parameter("black_executor_action_name").value),
        )
        self.create_service(GetBestMove, "/chess/get_best_move", self._get_best_move)
        self._execute_action = ActionServer(
            self,
            ExecuteChessMove,
            "/chess/execute_move",
            execute_callback=self._execute_move,
        )
        self._publish_board_state("ready")

    def _get_best_move(self, request: GetBestMove.Request, response: GetBestMove.Response) -> GetBestMove.Response:
        fen = request.fen or self._board.fen
        try:
            response.uci_move = self._engine.get_best_move(fen, request.think_time_sec or None)
            response.engine_name = self._engine.backend
        except EngineError as exc:
            self.get_logger().error(str(exc))
            response.uci_move = ""
            response.engine_name = f"error:{self._engine.backend}"
        return response

    def _execute_move(self, goal_handle) -> ExecuteChessMove.Result:
        move = goal_handle.request.uci_move
        feedback = ExecuteChessMove.Feedback()
        try:
            description = self._board.describe_move(move)
            side = self._current_side_to_move()
            self._publish_status("planning", move, message=f"side={side}")
            feedback.stage = "planning"
            feedback.progress = 0.1
            goal_handle.publish_feedback(feedback)

            feedback.stage = "dispatching"
            feedback.progress = 0.3
            goal_handle.publish_feedback(feedback)

            execution_result = self._forward_to_executor(side, move)
            if not execution_result.success:
                raise RuntimeError(execution_result.message or "Robot executor failed.")

            self._board.push_uci(move)
            self._publish_board_state("executed")
            self._publish_status(
                "executed",
                move,
                message=execution_result.message,
            )

            feedback.stage = "complete"
            feedback.progress = 1.0
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
            return ExecuteChessMove.Result(
                success=True,
                final_fen=self._board.fen,
                message=execution_result.message or f"Executed {move}.",
            )
        except (BoardError, EngineError, RuntimeError, ValueError) as exc:
            self.get_logger().error(str(exc))
            self._publish_status("failed", move, message=str(exc))
            goal_handle.abort()
            return ExecuteChessMove.Result(
                success=False,
                final_fen=self._board.fen,
                message=str(exc),
            )

    def _publish_board_state(self, phase: str) -> None:
        del phase
        self._board_publisher.publish(String(data=self._board.fen))

    def _publish_status(self, phase: str, move: str, message: str = "") -> None:
        text = f"{phase}: {move}"
        if message:
            text = f"{text} ({message})"
        self._status_publisher.publish(String(data=text))

    def _current_side_to_move(self) -> str:
        parts = self._board.fen.split()
        return "black" if len(parts) > 1 and parts[1] == "b" else "white"

    def _forward_to_executor(self, side: str, move: str) -> ExecuteChessMove.Result:
        executor = self._white_executor if side == "white" else self._black_executor
        if not executor.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{side.capitalize()} robot executor is unavailable.")
        goal = ExecuteChessMove.Goal()
        goal.uci_move = move
        send_future = executor.send_goal_async(goal)
        deadline = time.monotonic() + float(self.get_parameter("execution_timeout_sec").value)
        while not send_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not send_future.done():
            raise RuntimeError(f"Timed out dispatching move to {side} executor.")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"{side.capitalize()} robot executor rejected the move goal.")
        result_future = goal_handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not result_future.done():
            raise RuntimeError(f"Timed out waiting for {side} executor result.")
        return result_future.result().result


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = ChessManagerNode()
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
