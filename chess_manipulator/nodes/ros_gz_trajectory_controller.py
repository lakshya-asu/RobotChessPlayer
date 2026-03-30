#!/usr/bin/env python3
"""Expose a FollowJointTrajectory action backed by ros_gz joint position topics."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Sequence, Tuple

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectoryPoint


DEFAULT_JOINT_NAMES = [f"panda_joint{joint_index}" for joint_index in range(1, 8)]
DEFAULT_COMMAND_TOPICS = [
    f"/panda_joint{joint_index}_position_cmd" for joint_index in range(1, 8)
]


class RosGzTrajectoryControllerNode(Node):
    """Drive Gazebo Sim joint position controllers from FollowJointTrajectory goals."""

    def __init__(self) -> None:
        super().__init__("ros_gz_trajectory_controller")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("action_name", "/panda_arm_controller/follow_joint_trajectory")
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("joint_command_topics", DEFAULT_COMMAND_TOPICS)
        self.declare_parameter("goal_tolerance", 0.08)
        self.declare_parameter("goal_timeout_padding_sec", 10.0)
        self.declare_parameter("feedback_period_sec", 0.1)
        self.declare_parameter("command_period_sec", 0.02)
        self.declare_parameter("joint_state_wait_sec", 5.0)

        callback_group = ReentrantCallbackGroup()
        self._joint_names = [
            str(name) for name in self.get_parameter("joint_names").value
        ]
        command_topics = [
            str(topic) for topic in self.get_parameter("joint_command_topics").value
        ]
        if len(command_topics) != len(self._joint_names):
            raise ValueError("joint_command_topics must match joint_names length")

        self._joint_publishers = {
            joint_name: self.create_publisher(Float64, command_topic, 10)
            for joint_name, command_topic in zip(self._joint_names, command_topics)
        }
        self._joint_positions: Dict[str, float] = {}
        self._joint_state_event = threading.Event()
        self._state_lock = threading.Lock()

        self._subscription = self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            50,
            callback_group=callback_group,
        )
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("action_name").value),
            execute_callback=self._execute,
            callback_group=callback_group,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        with self._state_lock:
            for name, position in zip(msg.name, msg.position):
                self._joint_positions[name] = position
        self._joint_state_event.set()

    def _snapshot_positions(self, joint_names: Sequence[str]) -> Dict[str, float]:
        with self._state_lock:
            return {
                name: self._joint_positions[name]
                for name in joint_names
                if name in self._joint_positions
            }

    def _wait_for_initial_positions(self, joint_names: Sequence[str]) -> Dict[str, float]:
        deadline = time.monotonic() + float(self.get_parameter("joint_state_wait_sec").value)
        while time.monotonic() < deadline:
            positions = self._snapshot_positions(joint_names)
            if len(positions) == len(joint_names):
                return positions
            self._joint_state_event.wait(timeout=0.2)
            self._joint_state_event.clear()
        return {}

    def _execute(self, goal_handle) -> FollowJointTrajectory.Result:
        trajectory = goal_handle.request.trajectory
        if not trajectory.joint_names or not trajectory.points:
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.INVALID_GOAL,
                "Trajectory must include joints and at least one point.",
            )

        joint_names = list(trajectory.joint_names)
        if any(name not in self._joint_publishers for name in joint_names):
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.INVALID_JOINTS,
                "Trajectory references joints that are not backed by Gazebo position controllers.",
            )

        if any(len(point.positions) != len(joint_names) for point in trajectory.points):
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.INVALID_GOAL,
                "Every trajectory point must include one position per joint.",
            )

        initial_positions = self._wait_for_initial_positions(joint_names)
        if len(initial_positions) != len(joint_names):
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.INVALID_GOAL,
                "Timed out waiting for initial joint states from Gazebo Sim.",
            )

        final_point = trajectory.points[-1]
        final_time_sec = max(self._point_time_sec(final_point), 0.1)
        deadline = time.monotonic() + final_time_sec + float(
            self.get_parameter("goal_timeout_padding_sec").value
        )
        tolerance = float(self.get_parameter("goal_tolerance").value)
        feedback_period = float(self.get_parameter("feedback_period_sec").value)
        command_period = float(self.get_parameter("command_period_sec").value)
        start_time = time.monotonic()
        last_feedback_time = 0.0
        last_command_time = 0.0
        last_max_error = float("inf")

        self.get_logger().info(
            f"Executing ros_gz joint-position trajectory with {len(trajectory.points)} points "
            f"over {final_time_sec:.2f}s."
        )

        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    "Trajectory goal was canceled.",
                )

            now = time.monotonic()
            elapsed = max(now - start_time, 0.0)
            desired_positions = self._interpolate_positions(
                trajectory_joint_names=joint_names,
                points=trajectory.points,
                start_positions=initial_positions,
                elapsed_sec=elapsed,
            )
            if now - last_command_time >= command_period:
                self._publish_joint_commands(joint_names, desired_positions)
                last_command_time = now

            positions = self._snapshot_positions(joint_names)
            if len(positions) == len(joint_names):
                actual_positions = [positions[name] for name in joint_names]
                desired_point = self._desired_point(
                    desired_positions=desired_positions,
                    elapsed_sec=min(elapsed, final_time_sec),
                )
                error = self._position_error(
                    desired=desired_positions,
                    actual_positions=actual_positions,
                )
                last_max_error = max(abs(value) for value in error.positions)

                if now - last_feedback_time >= feedback_period:
                    goal_handle.publish_feedback(
                        self._build_feedback(
                            joint_names=joint_names,
                            desired=desired_point,
                            actual_positions=actual_positions,
                            error=error,
                        )
                    )
                    last_feedback_time = now

                if elapsed >= final_time_sec and last_max_error <= tolerance:
                    goal_handle.succeed()
                    return self._result(
                        FollowJointTrajectory.Result.SUCCESSFUL,
                        "Gazebo Sim reached the commanded joint positions.",
                    )

            self._joint_state_event.wait(timeout=min(feedback_period, 0.05))
            self._joint_state_event.clear()

        self.get_logger().error(
            "ros_gz joint-position execution timed out with max joint error "
            f"{last_max_error:.4f} rad."
        )
        goal_handle.abort()
        return self._result(
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
            "Timed out waiting for Gazebo Sim to reach the commanded joint positions.",
        )

    def _publish_joint_commands(
        self,
        joint_names: Sequence[str],
        desired_positions: Sequence[float],
    ) -> None:
        for joint_name, desired_position in zip(joint_names, desired_positions):
            self._joint_publishers[joint_name].publish(Float64(data=float(desired_position)))

    def _interpolate_positions(
        self,
        trajectory_joint_names: Sequence[str],
        points: Sequence[JointTrajectoryPoint],
        start_positions: Dict[str, float],
        elapsed_sec: float,
    ) -> List[float]:
        if not points:
            return [start_positions[name] for name in trajectory_joint_names]

        waypoint_times = [0.0] + [self._point_time_sec(point) for point in points]
        waypoint_positions = [
            [start_positions[name] for name in trajectory_joint_names]
        ] + [list(point.positions) for point in points]

        if elapsed_sec <= 0.0:
            return list(waypoint_positions[0])

        final_time_sec = waypoint_times[-1]
        if elapsed_sec >= final_time_sec:
            return list(waypoint_positions[-1])

        for index in range(1, len(waypoint_times)):
            segment_end_time = waypoint_times[index]
            if elapsed_sec <= segment_end_time:
                segment_start_time = waypoint_times[index - 1]
                duration = max(segment_end_time - segment_start_time, 1e-6)
                ratio = (elapsed_sec - segment_start_time) / duration
                start_waypoint = waypoint_positions[index - 1]
                end_waypoint = waypoint_positions[index]
                return [
                    start_value + ((end_value - start_value) * ratio)
                    for start_value, end_value in zip(start_waypoint, end_waypoint)
                ]

        return list(waypoint_positions[-1])

    def _build_feedback(
        self,
        joint_names: Sequence[str],
        desired: JointTrajectoryPoint,
        actual_positions: Sequence[float],
        error: JointTrajectoryPoint,
    ) -> FollowJointTrajectory.Feedback:
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(joint_names)
        feedback.desired = desired
        actual = JointTrajectoryPoint()
        actual.positions = list(actual_positions)
        feedback.actual = actual
        feedback.error = error
        return feedback

    def _desired_point(
        self,
        desired_positions: Sequence[float],
        elapsed_sec: float,
    ) -> JointTrajectoryPoint:
        point = JointTrajectoryPoint()
        point.positions = list(desired_positions)
        point.time_from_start.sec = int(elapsed_sec)
        point.time_from_start.nanosec = int((elapsed_sec % 1.0) * 1e9)
        return point

    def _position_error(
        self,
        desired: Sequence[float],
        actual_positions: Sequence[float],
    ) -> JointTrajectoryPoint:
        error = JointTrajectoryPoint()
        error.positions = [
            desired_position - actual_position
            for desired_position, actual_position in zip(desired, actual_positions)
        ]
        return error

    def _point_time_sec(self, point: JointTrajectoryPoint) -> float:
        return float(point.time_from_start.sec) + (float(point.time_from_start.nanosec) / 1e9)

    def _result(self, error_code: int, message: str) -> FollowJointTrajectory.Result:
        result = FollowJointTrajectory.Result()
        result.error_code = error_code
        result.error_string = message
        return result


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = RosGzTrajectoryControllerNode()
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
