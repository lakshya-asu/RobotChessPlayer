"""Namespaced ROS topic helpers for the dual-Panda Isaac scene."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotTopicSet:
    """Topic bundle for one robot namespace."""

    role: str
    command_topic: str
    joint_state_topic: str
    status_topic: str
    execution_result_topic: str


def default_dual_robot_topic_sets() -> dict[str, RobotTopicSet]:
    """Return the default white/black topic namespaces used by the Isaac app."""
    return {
        "white": RobotTopicSet(
            role="white",
            command_topic="/white/command/joint_trajectory",
            joint_state_topic="/white/joint_states",
            status_topic="/white/status",
            execution_result_topic="/white/execution_result",
        ),
        "black": RobotTopicSet(
            role="black",
            command_topic="/black/command/joint_trajectory",
            joint_state_topic="/black/joint_states",
            status_topic="/black/status",
            execution_result_topic="/black/execution_result",
        ),
    }
