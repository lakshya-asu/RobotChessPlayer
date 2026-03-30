#!/usr/bin/env python3
"""Filter Gazebo joint states down to the actuated Panda joints expected by MoveIt."""

from __future__ import annotations

from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateFilterNode(Node):
    """Drop fixed / simulator-only joints from the bridged Gazebo JointState stream."""

    def __init__(self) -> None:
        super().__init__("joint_state_filter")
        self.declare_parameter("input_topic", "/joint_states_raw")
        self.declare_parameter("output_topic", "/joint_states")
        self.declare_parameter("exclude_joints", ["robot_to_world"])

        self._exclude = set(str(name) for name in self.get_parameter("exclude_joints").value)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(JointState, output_topic, 20)
        self.create_subscription(JointState, input_topic, self._on_joint_state, 20)
        self.get_logger().info(
            f"joint_state_filter forwarding {input_topic} -> {output_topic}, "
            f"excluding {sorted(self._exclude)}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        filtered = JointState()
        filtered.header = msg.header
        indexes: List[int] = [
            index
            for index, name in enumerate(msg.name)
            if name not in self._exclude
        ]
        filtered.name = [msg.name[index] for index in indexes]
        filtered.position = [msg.position[index] for index in indexes if index < len(msg.position)]
        filtered.velocity = [msg.velocity[index] for index in indexes if index < len(msg.velocity)]
        filtered.effort = [msg.effort[index] for index in indexes if index < len(msg.effort)]
        self._publisher.publish(filtered)


def main(args: Tuple[str, ...] | None = None) -> None:
    rclpy.init(args=args)
    node = JointStateFilterNode()
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
