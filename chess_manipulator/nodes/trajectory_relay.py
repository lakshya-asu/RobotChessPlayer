#!/usr/bin/env python3
"""Relay planned trajectories to the configured simulator backend."""

from __future__ import annotations

from functools import partial
from typing import Tuple

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from chess_manipulator_msgs.msg import ExecutionCommand
from chess_manipulator.sim import (
    SimulationBackend,
    SimulationConfig,
    encode_feedback,
    execution_command_duration_sec,
    trajectory_duration_sec,
)


class TrajectoryRelayNode(Node):
    """Forward `/chess/planned_trajectory` to Gazebo or Isaac-specific topics."""

    def __init__(self) -> None:
        super().__init__("trajectory_relay")
        self.declare_parameter("sim_backend", "isaac")
        self.declare_parameter("planned_trajectory_topic", "/white/planned_trajectory")
        self.declare_parameter("execution_feedback_topic", "/white/execution_feedback")
        self.declare_parameter(
            "controller_action_name",
            "/panda_arm_controller/follow_joint_trajectory",
        )
        backend = SimulationBackend(
            SimulationConfig(backend=str(self.get_parameter("sim_backend").value))
        )
        self._backend_name = backend.backend
        self._publisher = self.create_publisher(JointTrajectory, backend.command_topic, 10)
        self._feedback_publisher = self.create_publisher(
            String,
            str(self.get_parameter("execution_feedback_topic").value),
            10,
        )
        self._action_client = None
        self._timers = []
        self._subscription = None
        if self._backend_name in {"gazebo", "ros_gz"}:
            self._subscription = self.create_subscription(
                ExecutionCommand,
                str(self.get_parameter("planned_trajectory_topic").value),
                self._relay,
                10,
            )
            if self._backend_name == "ros_gz":
                self._action_client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    str(self.get_parameter("controller_action_name").value),
                )
                self.get_logger().info(
                    "trajectory_relay is using FollowJointTrajectory for ros_gz execution."
                )
        else:
            self.get_logger().info(
                f"trajectory_relay is idle for backend {self._backend_name}; "
                "Isaac bridge consumes the command payload."
            )

    def _relay(self, msg: ExecutionCommand) -> None:
        move = msg.move_id or "unknown"
        if self._backend_name == "ros_gz":
            self._execute_ros_gz(msg)
            return

        trajectory = msg.trajectory
        self._publisher.publish(trajectory)
        duration_sec = trajectory_duration_sec(trajectory)
        self._publish_feedback(
            state="accepted",
            move=move,
            duration_sec=duration_sec,
            message="trajectory relayed",
        )
        if self._backend_name == "gazebo":
            self._schedule_completion(move, duration_sec)

    def _execute_ros_gz(self, msg: ExecutionCommand) -> None:
        if self._action_client is None:
            self._publish_feedback(
                state="failed",
                move=msg.move_id or "unknown",
                duration_sec=0.0,
                message="controller action client unavailable",
            )
            return
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self._publish_feedback(
                state="failed",
                move=msg.move_id or "unknown",
                duration_sec=0.0,
                message="panda_arm_controller action server not available",
            )
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = msg.trajectory
        duration_sec = execution_command_duration_sec(msg)
        send_future = self._action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            partial(
                self._on_goal_response,
                move=msg.move_id or "unknown",
                duration_sec=duration_sec,
            )
        )

    def _on_goal_response(self, future, move: str, duration_sec: float) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive async guard
            self._publish_feedback(
                state="failed",
                move=move,
                duration_sec=duration_sec,
                message=f"goal dispatch failed: {exc}",
            )
            return

        if not goal_handle.accepted:
            self._publish_feedback(
                state="failed",
                move=move,
                duration_sec=duration_sec,
                message="controller rejected trajectory",
            )
            return

        self._publish_feedback(
            state="accepted",
            move=move,
            duration_sec=duration_sec,
            message="controller accepted trajectory",
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._on_goal_result, move=move, duration_sec=duration_sec)
        )

    def _on_goal_result(self, future, move: str, duration_sec: float) -> None:
        try:
            wrapped_result = future.result()
        except Exception as exc:  # pragma: no cover - defensive async guard
            self._publish_feedback(
                state="failed",
                move=move,
                duration_sec=duration_sec,
                message=f"controller result failed: {exc}",
            )
            return

        result = wrapped_result.result
        error_code = int(getattr(result, "error_code", 1))
        error_string = getattr(result, "error_string", "")
        if error_code == 0:
            self._publish_feedback(
                state="completed",
                move=move,
                duration_sec=duration_sec,
                message="controller execution complete",
            )
            return
        self._publish_feedback(
            state="failed",
            move=move,
            duration_sec=duration_sec,
            message=error_string or f"controller execution failed with code {error_code}",
        )

    def _schedule_completion(self, move: str, duration_sec: float) -> None:
        delay = max(duration_sec, 0.05)
        timer = self.create_timer(
            delay,
            partial(self._publish_completion, move=move, duration_sec=duration_sec),
        )
        self._timers.append(timer)

    def _publish_completion(self, move: str, duration_sec: float) -> None:
        timer = self._timers.pop(0)
        timer.cancel()
        self._publish_feedback(
            state="completed",
            move=move,
            duration_sec=duration_sec,
            message="legacy gazebo completion",
        )

    def _publish_feedback(
        self,
        state: str,
        move: str,
        duration_sec: float,
        message: str,
    ) -> None:
        self._feedback_publisher.publish(
            String(
                data=encode_feedback(
                    state=state,
                    backend=self._backend_name,
                    move=move,
                    duration_sec=duration_sec,
                    message=message,
                )
            )
        )


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = TrajectoryRelayNode()
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
