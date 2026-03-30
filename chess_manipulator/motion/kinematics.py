"""Deterministic joint-limit validation with a lightweight Panda model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, sqrt
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class JointLimitViolation(RuntimeError):
    """Raised when a planned pose would exceed joint limits."""


@dataclass
class PandaKinematicsModel:
    """Approximate Panda IK suitable for deterministic planning and validation."""

    hover_roll_rad: float = np.pi
    joint_limits: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: (
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        )
    )

    def solve_pose(self, pose: Tuple[float, float, float]) -> List[float]:
        x, y, z = pose
        radial = sqrt(x * x + y * y)
        if radial > 0.95 or z < 0.0 or z > 1.1:
            raise JointLimitViolation(f"Pose is outside the reachable workspace: {pose}")

        q1 = atan2(y, x)
        q2 = float(np.clip((0.45 - z) * 4.5, *self.joint_limits[1]))
        q3 = float(np.clip((radial - 0.35) * 4.0, *self.joint_limits[2]))
        q4 = float(np.clip(-2.2 + (z - 0.3), *self.joint_limits[3]))
        q5 = float(np.clip(-q3 / 2.0, *self.joint_limits[4]))
        q6 = float(np.clip(1.6 - (z - 0.3) * 1.5, *self.joint_limits[5]))
        q7 = float(np.clip(q1 / 3.0, *self.joint_limits[6]))
        joints = [q1, q2, q3, q4, q5, q6, q7]
        self.validate_joint_positions(joints)
        return joints

    def validate_joint_positions(self, joints: Iterable[float]) -> None:
        joint_values = list(joints)
        if len(joint_values) != len(self.joint_limits):
            raise JointLimitViolation("Expected 7 Panda joints.")
        for index, (value, limits) in enumerate(zip(joint_values, self.joint_limits), start=1):
            low, high = limits
            if value < low or value > high:
                raise JointLimitViolation(
                    f"Joint {index} value {value:.3f} exceeds [{low:.3f}, {high:.3f}]"
                )
