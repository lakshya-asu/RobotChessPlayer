from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from chess_arm.sim.digital_twin import FrankaChessDigitalTwin

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
except ImportError:
    rclpy = None
    Node = object  # type: ignore[assignment]
    JointState = object  # type: ignore[assignment]
    JointTrajectory = object  # type: ignore[assignment]


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
            JointTrajectory,
            config.joint_trajectory_topic,
            self._traj_callback,
            10,
        )

        self._timer = self.create_timer(0.01, self._publish_joint_state)

    def _publish_joint_state(self) -> None:
        q = self._twin.get_joint_positions()
        msg = JointState()
        msg.name = self._config.joint_names
        msg.position = q.tolist()
        self._js_pub.publish(msg)

    def _traj_callback(self, msg: JointTrajectory) -> None:
        # TODO: Convert JointTrajectory message into internal JointTrajectoryExecutor
        self.get_logger().info(
            f"Received trajectory with {len(msg.points)} points (not yet executed)."
        )


def run_ros_bridge(twin: FrankaChessDigitalTwin, config: RosBridgeConfig) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy not available")

    rclpy.init()
    node = FrankaRosBridge(twin, config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
