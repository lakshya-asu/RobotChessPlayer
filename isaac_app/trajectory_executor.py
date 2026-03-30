"""Trajectory subscription and articulation playback for Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import omni.timeline
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from isaac_app.channels import RobotTopicSet
from isaac_app.compat import ArticulationAction
from chess_manipulator_msgs.msg import ExecutionCommand
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
FINGER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
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
    move_id: str
    start_time_sec: float
    start_positions: List[float]
    point_times_sec: List[float]
    point_positions: List[List[float]]
    stage_names: List[str]
    stage_end_times_sec: List[float]
    finished: bool = False


class IsaacTrajectoryExecutor(Node):
    """Consume the Isaac joint-trajectory topic and drive the Panda articulation."""

    def __init__(
        self,
        manipulator,
        channels: RobotTopicSet,
        scene=None,
        node_name: str | None = None,
    ) -> None:
        super().__init__(node_name or f"isaac_trajectory_executor_{channels.role}")
        self._manipulator = manipulator
        self._channels = channels
        self._scene = scene
        self._articulation_controller = manipulator.get_articulation_controller()
        self._joint_state_pub = self.create_publisher(JointState, channels.joint_state_topic, 10)
        self._status_pub = self.create_publisher(String, channels.status_topic, 10)
        self._result_pub = self.create_publisher(String, channels.execution_result_topic, 10)
        self._trajectory_sub = self.create_subscription(
            ExecutionCommand,
            channels.command_topic,
            self._on_trajectory,
            10,
        )
        self._active_trajectory: Optional[ActiveTrajectory] = None
        self._expected_joint_names = list(JOINT_NAMES)
        dof_names = list(getattr(self._manipulator, "dof_names", []) or [])
        if dof_names:
            self._arm_joint_indices = [dof_names.index(name) for name in self._expected_joint_names]
        else:
            self._arm_joint_indices = list(range(len(self._expected_joint_names)))
        self._finger_joint_indices = [
            dof_names.index(name) for name in FINGER_JOINT_NAMES if name in dof_names
        ]
        gripper = getattr(self._manipulator, "gripper", None)
        self._gripper_open_positions = list(
            getattr(gripper, "joint_opened_positions", [0.04, 0.04])
        )
        self._gripper_closed_positions = list(
            getattr(gripper, "joint_closed_positions", [0.02, 0.02])
        )
        self.get_logger().info(
            f"Listening on {channels.command_topic}; publishing joint states to {channels.joint_state_topic}"
        )

    @property
    def channels(self) -> RobotTopicSet:
        return self._channels

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _on_trajectory(self, msg: ExecutionCommand) -> None:
        trajectory = msg.trajectory
        move_id = msg.move_id or f"__{self._channels.role}_trajectory__"
        if list(trajectory.joint_names) and list(trajectory.joint_names) != self._expected_joint_names:
            self.get_logger().warning(
                f"Received joint trajectory with unexpected joint order: {','.join(trajectory.joint_names)}"
            )
        if not trajectory.points:
            self._result_pub.publish(
                String(data=encode_isaac_result("failed", 0.0, "trajectory has no points"))
            )
            return
        point_times = [duration_to_seconds(point.time_from_start) for point in trajectory.points]
        point_positions = [list(point.positions) for point in trajectory.points]
        start_positions = arm_positions(self._manipulator.get_joint_positions())
        self._active_trajectory = ActiveTrajectory(
            move_id=move_id,
            start_time_sec=time.monotonic(),
            start_positions=start_positions,
            point_times_sec=point_times,
            point_positions=point_positions,
            stage_names=list(msg.stage_names),
            stage_end_times_sec=[duration_to_seconds(item) for item in msg.stage_end_times],
        )
        target_duration_sec = point_times[-1] if point_times else 0.0
        if self._scene is not None:
            started = self._scene.begin_piece_transport(self._channels.role, move_id, target_duration_sec)
            if not started:
                self._active_trajectory = None
                message = f"piece transport setup failed for {move_id}"
                self.get_logger().warning(message)
                self._publish_status(message)
                self._result_pub.publish(
                    String(data=encode_isaac_result("failed", 0.0, message))
                )
                return
        self.get_logger().info(
            f"Accepted Isaac trajectory {move_id} with {len(trajectory.points)} point(s), target duration {target_duration_sec:.2f}s."
        )
        self._publish_status(
            f"{self._channels.role} received trajectory {move_id} with {len(trajectory.points)} points on {','.join(trajectory.joint_names or self._expected_joint_names)}"
        )

    def step(self) -> bool:
        """Advance the current trajectory and publish the live joint state."""
        if self._active_trajectory is None:
            self._publish_joint_state(arm_positions(self._manipulator.get_joint_positions()))
            return False

        current_time_sec = time.monotonic()
        elapsed_sec = current_time_sec - self._active_trajectory.start_time_sec
        positions, finished = sample_trajectory(
            self._active_trajectory.start_positions,
            self._active_trajectory.point_times_sec,
            self._active_trajectory.point_positions,
            elapsed_sec,
        )
        if self._scene is not None:
            self._scene.step_piece_transport(self._channels.role, elapsed_sec)
        action_positions = list(positions)
        action_indices = list(self._arm_joint_indices)
        gripper_positions = self._gripper_positions_for(self._active_trajectory.move_id, elapsed_sec)
        if self._finger_joint_indices and gripper_positions:
            action_positions.extend(gripper_positions)
            action_indices.extend(self._finger_joint_indices)
        action = ArticulationAction(
            joint_positions=np.array(action_positions, dtype=float),
            joint_indices=np.array(action_indices, dtype=np.int64),
        )
        self._articulation_controller.apply_action(action)
        self._publish_joint_state(positions)

        if finished:
            self._active_trajectory.finished = True
            move_id = self._active_trajectory.move_id
            transport_result = None
            if self._scene is not None:
                transport_result = self._scene.finish_piece_transport(self._channels.role)
            self._active_trajectory = None
            success = transport_result.success if transport_result is not None else True
            message = (
                transport_result.message
                if transport_result is not None and transport_result.message
                else f"isaac trajectory playback complete for {move_id}"
            )
            state = "completed" if success else "failed"
            self.get_logger().info(
                f"Completed {self._channels.role} Isaac trajectory {move_id} in {max(elapsed_sec, 0.0):.3f}s with state={state}."
            )
            self._publish_status(
                f"{self._channels.role} completed trajectory {move_id} in {max(elapsed_sec, 0.0):.3f}s with state={state}"
            )
            self._result_pub.publish(
                String(
                    data=encode_isaac_result(
                        state,
                        max(elapsed_sec, 0.0),
                        message,
                    )
                )
            )
            return success
        return False

    def _gripper_positions_for(self, move_id: str, elapsed_sec: float) -> List[float]:
        if move_id.startswith("__") or ":home_" in move_id:
            return list(self._gripper_open_positions)
        if self._active_trajectory is None:
            return list(self._gripper_open_positions)
        if self._active_trajectory.stage_names and self._active_trajectory.stage_end_times_sec:
            return self._gripper_positions_for_stage(elapsed_sec)
        total_duration = max(self._active_trajectory.point_times_sec[-1], 1e-3)
        alpha = min(max(elapsed_sec / total_duration, 0.0), 1.0)
        if alpha < 0.16:
            return list(self._gripper_open_positions)
        if alpha < 0.84:
            return list(self._gripper_closed_positions)
        if alpha < 0.94:
            blend = (alpha - 0.84) / 0.10
            return lerp(self._gripper_closed_positions, self._gripper_open_positions, blend)
        return list(self._gripper_open_positions)

    def _gripper_positions_for_stage(self, elapsed_sec: float) -> List[float]:
        if self._active_trajectory is None:
            return list(self._gripper_open_positions)
        stage_name, stage_progress = self._current_stage(elapsed_sec)
        if stage_name is None:
            return list(self._gripper_open_positions)
        if "approach" in stage_name or "descend" in stage_name:
            return list(self._gripper_open_positions)
        if "lift" in stage_name or "transit" in stage_name:
            return list(self._gripper_closed_positions)
        if "place" in stage_name:
            return lerp(self._gripper_closed_positions, self._gripper_open_positions, stage_progress)
        return list(self._gripper_open_positions)

    def _current_stage(self, elapsed_sec: float) -> tuple[str | None, float]:
        if self._active_trajectory is None:
            return None, 0.0
        previous_end = 0.0
        for stage_name, stage_end in zip(
            self._active_trajectory.stage_names,
            self._active_trajectory.stage_end_times_sec,
        ):
            if elapsed_sec <= stage_end:
                span = max(stage_end - previous_end, 1e-6)
                progress = min(max((elapsed_sec - previous_end) / span, 0.0), 1.0)
                return stage_name, progress
            previous_end = stage_end
        if self._active_trajectory.stage_names:
            return self._active_trajectory.stage_names[-1], 1.0
        return None, 0.0

    def _publish_joint_state(self, positions: Iterable[float]) -> None:
        if not rclpy.ok():
            return
        normalized_positions = arm_positions(positions)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._expected_joint_names)
        msg.position = normalized_positions
        msg.velocity = [0.0] * len(self._expected_joint_names)
        msg.effort = [0.0] * len(self._expected_joint_names)
        try:
            self._joint_state_pub.publish(msg)
        except Exception:
            return
