from __future__ import annotations

from typing import Optional

import numpy as np

try:
    # These imports only work inside Isaac Sim
    from omni.isaac.core.utils.types import ArticulationAction
except ImportError:
    ArticulationAction = object  # type: ignore[assignment]


class JointTrajectoryExecutor:
    """
    Simple executor that steps a JointTrajectory and returns ArticulationAction commands.
    """

    def __init__(
        self,
        joint_names: list[str],
        trajectory,
    ) -> None:
        self.joint_names = joint_names
        self.trajectory = trajectory
        self._t = 0.0

    def reset(self) -> None:
        self._t = 0.0

    def step(self, dt: float):
        """Return (ArticulationAction, done_flag)."""
        self._t += dt
        q = self.trajectory.sample(self._t)
        if ArticulationAction is object:
            # outside Isaac Sim: just return raw joint positions
            return q, self._t >= self.trajectory.times[-1]
        action = ArticulationAction(joint_positions=q)
        done = self._t >= self.trajectory.times[-1]
        return action, done
