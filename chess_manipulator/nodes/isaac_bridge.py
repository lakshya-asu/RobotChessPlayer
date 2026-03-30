#!/usr/bin/env python3
"""ROS-side bridge shim for Isaac Sim topic integration."""

from __future__ import annotations

from functools import partial
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from chess_manipulator_msgs.msg import ExecutionCommand
from chess_manipulator.sim import (
    SimulationBackend,
    SimulationConfig,
    decode_isaac_result,
    encode_feedback,
    execution_command_duration_sec,
)


class IsaacBridgeNode(Node):
    """Expose a ROS-facing contract for Isaac Sim topic integration."""

    def __init__(self) -> None:
        super().__init__("isaac_bridge")
        self.declare_parameter("sim_backend", "isaac")
        self.declare_parameter("planned_trajectory_topic", "/white/planned_trajectory")
        self.declare_parameter("execution_feedback_topic", "/white/execution_feedback")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("source_joint_state_topic", "")
        self.declare_parameter("command_topic", "")
        self.declare_parameter("isaac_execution_result_topic", "/isaac/execution_result")
        self.declare_parameter("simulate_isaac_completion", False)
        self.declare_parameter("isaac_completion_padding_sec", 0.05)
        backend = SimulationBackend(
            SimulationConfig(backend=str(self.get_parameter("sim_backend").value))
        )
        command_topic = str(self.get_parameter("command_topic").value) or backend.command_topic
        source_joint_state_topic = (
            str(self.get_parameter("source_joint_state_topic").value) or backend.source_joint_state_topic
        )
        self._joint_state_pub = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            10,
        )
        self._feedback_pub = self.create_publisher(
            String,
            str(self.get_parameter("execution_feedback_topic").value),
            10,
        )
        self._command_pub = self.create_publisher(ExecutionCommand, command_topic, 10)
        self._timers = []
        self._active_move = ""
        self._active_duration_sec = 0.0
        self._isaac_ready = False
        self._queued_command: ExecutionCommand | None = None
        self._isaac_joint_state_sub = self.create_subscription(
            JointState,
            source_joint_state_topic,
            self._handle_isaac_joint_state,
            10,
        )
        self._isaac_result_sub = self.create_subscription(
            String,
            str(self.get_parameter("isaac_execution_result_topic").value),
            self._handle_isaac_result,
            10,
        )
        self._command_sub = self.create_subscription(
            ExecutionCommand,
            str(self.get_parameter("planned_trajectory_topic").value),
            self._handle_command,
            10,
        )

    def _handle_isaac_joint_state(self, msg: JointState) -> None:
        if not self._isaac_ready and len(msg.name) >= 7 and len(msg.position) >= 7:
            self._isaac_ready = True
            self.get_logger().info("Isaac bridge is ready after the first joint-state heartbeat.")
            if self._queued_command is not None:
                queued = self._queued_command
                self._queued_command = None
                self.get_logger().info(
                    f"Dispatching queued Isaac trajectory for {queued.move_id or 'unknown'} after readiness."
                )
                self._forward_command(queued)
        self._joint_state_pub.publish(msg)

    def _handle_command(self, msg: ExecutionCommand) -> None:
        move = msg.move_id or "unknown"
        if not self._isaac_ready:
            self._queued_command = msg
            self.get_logger().info(
                f"Queueing {move} until Isaac joint states are available."
            )
            return

        self._forward_command(msg)

    def _forward_command(self, msg: ExecutionCommand) -> None:
        move = msg.move_id or "unknown"
        duration_sec = execution_command_duration_sec(msg)
        self._active_move = move
        self._active_duration_sec = duration_sec
        self._command_pub.publish(msg)
        self._feedback_pub.publish(
            String(
                data=encode_feedback(
                    state="accepted",
                    backend="isaac",
                    move=move,
                    duration_sec=duration_sec,
                    message="isaac bridge accepted trajectory",
                )
            )
        )
        if bool(self.get_parameter("simulate_isaac_completion").value):
            delay = max(
                duration_sec + float(self.get_parameter("isaac_completion_padding_sec").value),
                0.05,
            )
            timer = self.create_timer(delay, partial(self._publish_simulated_completion, move))
            self._timers.append(timer)

    def _handle_isaac_result(self, msg: String) -> None:
        payload = decode_isaac_result(msg.data)
        if not self._active_move:
            self.get_logger().warning("Ignoring Isaac execution result without an active move.")
            return
        duration_sec = float(payload.get("duration_sec") or self._active_duration_sec or 0.0)
        self._feedback_pub.publish(
            String(
                data=encode_feedback(
                    state=payload.get("state") or "completed",
                    backend="isaac",
                    move=self._active_move,
                    duration_sec=duration_sec,
                    message=payload.get("message") or "isaac execution result",
                )
            )
        )
        if payload.get("state") in {"completed", "failed"}:
            self._active_move = ""
            self._active_duration_sec = 0.0

    def _publish_simulated_completion(self, move: str) -> None:
        if not self._timers:
            return
        timer = self._timers.pop(0)
        timer.cancel()
        if self._active_move != move:
            return
        self._feedback_pub.publish(
            String(
                data=encode_feedback(
                    state="completed",
                    backend="isaac",
                    move=move,
                    duration_sec=self._active_duration_sec,
                    message="simulated isaac completion",
                )
            )
        )
        self._active_move = ""
        self._active_duration_sec = 0.0


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = IsaacBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
