#!/usr/bin/env python3
"""Run one engine-backed chess turn through the ROS APIs."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from chess_manipulator_msgs.action import ExecuteChessMove
from chess_manipulator_msgs.srv import GetBestMove


class DemoTurnClient(Node):
    """Request a move from the service, then execute it via the action server."""

    def __init__(self, think_time_sec: float) -> None:
        super().__init__("demo_turn_client")
        self._think_time_sec = think_time_sec
        self._move_service = self.create_client(GetBestMove, "/chess/get_best_move")
        self._action_client = ActionClient(self, ExecuteChessMove, "/chess/execute_move")

    def run(self, fen: str = "") -> ExecuteChessMove.Result:
        if not self._move_service.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("GetBestMove service is unavailable.")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("ExecuteChessMove action is unavailable.")

        request = GetBestMove.Request()
        request.fen = fen
        request.think_time_sec = self._think_time_sec
        future = self._move_service.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if not response or not response.uci_move:
            raise RuntimeError("Engine did not return a move.")

        goal = ExecuteChessMove.Goal()
        goal.uci_move = response.uci_move
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Move goal rejected: {response.uci_move}")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(
            f"engine={response.engine_name} move={response.uci_move} success={result.success}"
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default="", help="Optional FEN override.")
    parser.add_argument("--think-time", type=float, default=0.25)
    args = parser.parse_args()

    rclpy.init()
    node = DemoTurnClient(think_time_sec=args.think_time)
    try:
        result = node.run(fen=args.fen)
        print(f"success={result.success} final_fen={result.final_fen} message={result.message}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
