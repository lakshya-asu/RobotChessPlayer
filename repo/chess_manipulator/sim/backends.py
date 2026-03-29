"""Simulation backend configuration and relay utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendName = Literal["isaac", "gazebo", "ros_gz"]


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime simulator configuration."""

    backend: BackendName = "isaac"
    joint_state_topic: str = "/joint_states"
    command_topic: str = "/panda_arm_controller/joint_trajectory"
    isaac_command_topic: str = "/isaac/command/joint_trajectory"
    isaac_joint_state_topic: str = "/isaac/joint_states"


class SimulationBackend:
    """Simple selector for simulator-specific topic contracts."""

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config

    @property
    def backend(self) -> BackendName:
        return self._config.backend

    @property
    def source_joint_state_topic(self) -> str:
        if self._config.backend == "isaac":
            return self._config.isaac_joint_state_topic
        return self._config.joint_state_topic

    @property
    def command_topic(self) -> str:
        if self._config.backend == "isaac":
            return self._config.isaac_command_topic
        return self._config.command_topic
