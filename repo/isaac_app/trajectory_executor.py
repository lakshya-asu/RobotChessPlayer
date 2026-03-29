"""Trajectory subscription and articulation playback for Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import omni.timeline
from isaacsim.core.utils.types import ArticulationAction
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from chess_manipulator.sim import encode_isaac_result


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


def duration_to_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) / 1e9


def lerp(start: Sequence[float], end: Sequence[float], alpha: float) -> List[float]:
    alpha = max(0.0, min(1.0, alpha))
    return [float(a + (b - a) * alpha) for a, b in zip(start, end)]


def arm_positions(values: Iterable[float]) -> List[float]:
    """Normalize articulation state down to the seven Panda arm joints."""
    return [float(value) for value in list(values)[: len(JOINT_NAMES)]]


def sample_trajectory(
    start_positions: Sequence[float],
    point_times: Sequence[float],
    point_positions: Sequence[Sequence[float]],
    elapsed_sec: float,
) -> Tuple[List[float], bool]:
    """Sample a piecewise-linear trajectory at the provided elapsed time."""
    if not point_times or not point_positions:
        return list(start_positions), True

    previous_time = 0.0
    previous_positions = list(start_positions)

    for point_time, next_positions in zip(point_times, point_positions):
        if elapsed_sec <= point_time:
            span = max(point_time - previous_time, 1e-6)
            alpha = (elapsed_sec - previous_time) / span
            return lerp(previous_positions, next_positions, alpha), False
        previous_time = point_time
        previous_positions = list(next_positions)

    return list(point_positions[-1]), True


@dataclass
class ActiveTrajectory:
    start_time_sec: float
    start_positions: List[float]
    point_times_sec: List[float]
    point_positions: List[List[float]]
    finished: bool = False


class IsaacTrajectoryExecutor(Node):
    """Consume the Isaac joint-trajectory topic and drive the Panda articulation."""

    def __init__(
        self,
        manipulator,
        command_topic: str = "/isaac/command/joint_trajectory",
        joint_state_topic: str = "/isaac/joint_states",
        status_topic: str = "/isaac/status",
        execution_result_topic: str = "/isaac/execution_result",
    ) -> None:
        super().__init__("isaac_trajectory_executor")
        self._manipulator = manipulator
        self._articulation_controller = manipulator.get_articulation_controller()
        self._timeline = omni.timeline.get_timeline_interface()
        self._joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._result_pub = self.create_publisher(String, execution_result_topic, 10)
        self._trajectory_sub = self.create_subscription(
            JointTrajectory,
            command_topic,
            self._on_trajectory,
            10,
        )
        self._active_trajectory: Optional[ActiveTrajectory] = None
        self._expected_joint_names = list(JOINT_NAMES)

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _on_trajectory(self, msg: JointTrajectory) -> None:
        if list(msg.joint_names) and list(msg.joint_names) != self._expected_joint_names:
            self.get_logger().warning(
                "Received joint trajectory with unexpected joint order: %s",
                ",".join(msg.joint_names),
            )
        if not msg.points:
            self._result_pub.publish(
                String(data=encode_isaac_result("failed", 0.0, "trajectory has no points"))
            )
            return
        point_times = [duration_to_seconds(point.time_from_start) for point in msg.points]
        point_positions = [list(point.positions) for point in msg.points]
        start_positions = arm_positions(self._manipulator.get_joint_positions())
        self._active_trajectory = ActiveTrajectory(
            start_time_sec=self._timeline.get_current_time(),
            start_positions=start_positions,
            point_times_sec=point_times,
            point_positions=point_positions,
        )
        self._publish_status(
            f"received trajectory with {len(msg.points)} points on {','.join(msg.joint_names or self._expected_joint_names)}"
        )

    def step(self) -> bool:
        """Advance the current trajectory and publish the live joint state."""
        if self._active_trajectory is None:
            self._publish_joint_state(arm_positions(self._manipulator.get_joint_positions()))
            return False

        current_time_sec = self._timeline.get_current_time()
        elapsed_sec = current_time_sec - self._active_trajectory.start_time_sec
        positions, finished = sample_trajectory(
            self._active_trajectory.start_positions,
            self._active_trajectory.point_times_sec,
            self._active_trajectory.point_positions,
            elapsed_sec,
        )
        action = ArticulationAction(joint_positions=np.array(positions, dtype=float))
        self._articulation_controller.apply_action(action)
        self._publish_joint_state(positions)

        if finished:
            self._active_trajectory.finished = True
            self._active_trajectory = None
            self._publish_status(f"completed trajectory in {max(elapsed_sec, 0.0):.3f}s")
            self._result_pub.publish(
                String(
                    data=encode_isaac_result(
                        "completed",
                        max(elapsed_sec, 0.0),
                        "isaac trajectory playback complete",
                    )
                )
            )
            return True
        return False

    def _publish_joint_state(self, positions: Iterable[float]) -> None:
        normalized_positions = arm_positions(positions)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._expected_joint_names)
        msg.position = normalized_positions
        msg.velocity = [0.0] * len(self._expected_joint_names)
        msg.effort = [0.0] * len(self._expected_joint_names)
        self._joint_state_pub.publish(msg)
