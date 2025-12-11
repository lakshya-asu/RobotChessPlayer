from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from chess_arm.sim.digital_twin import FrankaChessDigitalTwin
from chess_arm.control.trajectory_generator import JointTrajectory as InternalJointTrajectory

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory as RosJointTrajectory
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object  # type: ignore[assignment]
    JointState = object  # type: ignore[assignment]
    RosJointTrajectory = object  # type: ignore[assignment]


@dataclass
class RosBridgeConfig:
    joint_state_topic: str
    joint_trajectory_topic: str
    joint_names: List[str]


class FrankaRosBridge(Node):
    """ROS2 bridge publishing joint states and accepting joint trajectories."""

    def __init__(self, twin: FrankaChessDigitalTwin, config: RosBridgeConfig) -> None:
        if rclpy is None or Node is object:
            raise RuntimeError("ROS2 (rclpy) not available")

        super().__init__("franka_chess_bridge")
        self._twin = twin
        self._config = config

        self._js_pub = self.create_publisher(
            JointState, config.joint_state_topic, 10
        )
        self._traj_sub = self.create_subscription(
            RosJointTrajectory,
            config.joint_trajectory_topic,
            self._traj_callback,
            10,
        )

        self._timer = self.create_timer(0.01, self._publish_joint_state)

    # ------------------------------------------------------------------------- #
    # Joint state publishing
    # ------------------------------------------------------------------------- #

    def _publish_joint_state(self) -> None:
        q = self._twin.get_joint_positions()
        msg = JointState()
        msg.name = self._config.joint_names
        msg.position = q.tolist()
        self._js_pub.publish(msg)

    # ------------------------------------------------------------------------- #
    # Trajectory handling
    # ------------------------------------------------------------------------- #

    def _ros_to_internal_trajectory(
        self,
        msg: RosJointTrajectory,
    ) -> InternalJointTrajectory:
        """
        Convert a ROS JointTrajectory message into an internal JointTrajectory.
        """
        if not msg.points:
            raise ValueError("JointTrajectory message has no points")

        names_config = list(self._config.joint_names)
        names_msg = list(msg.joint_names)

        if names_msg:
            indices: List[int] = []
            for name in names_config:
                if name not in names_msg:
                    raise ValueError(f"Joint name '{name}' not found in JointTrajectory message")
                indices.append(names_msg.index(name))
        else:
            # todo: handle case where joint_names is empty more robustly
            indices = list(range(len(names_config)))

        times = []
        positions = []

        for pt in msg.points:
            if not pt.positions:
                raise ValueError("JointTrajectory point has no positions")
            t = float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9
            pos = np.array([pt.positions[i] for i in indices], dtype=float)
            times.append(t)
            positions.append(pos)

        times_arr = np.asarray(times, dtype=float)
        positions_arr = np.stack(positions, axis=0)

        return InternalJointTrajectory(times=times_arr, positions=positions_arr)

    def _traj_callback(self, msg: RosJointTrajectory) -> None:
        try:
            traj = self._ros_to_internal_trajectory(msg)
            self._twin.set_trajectory(traj)
            self.get_logger().info(
                f"Applied JointTrajectory with {len(msg.points)} points."
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to apply JointTrajectory: {exc}")


def run_ros_bridge(twin: FrankaChessDigitalTwin, config: RosBridgeConfig) -> None:
    """
    Blocking ROS2 spin helper (no Isaac Sim stepping).

    This is intended for environments where the sim is stepped elsewhere.
    """
    if rclpy is None:
        raise RuntimeError("rclpy not available")

    rclpy.init()
    node = FrankaRosBridge(twin, config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
