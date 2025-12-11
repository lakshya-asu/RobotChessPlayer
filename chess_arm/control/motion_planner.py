from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .trajectory_generator import JointTrajectory, generate_time_scaled_trajectory


CollisionChecker = Callable[[np.ndarray], bool]


@dataclass
class PlanningResult:
    success: bool
    trajectory: Optional[JointTrajectory] = None
    message: str = ""


def plan_joint_space_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_waypoints: int = 5,
    collision_checker: Optional[CollisionChecker] = None,
) -> PlanningResult:
    """
    Naive joint-space planner: straight-line interpolation with optional collision checks.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    waypoints = [
        (1 - alpha) * q_start + alpha * q_goal
        for alpha in np.linspace(0.0, 1.0, n_waypoints)
    ]

    if collision_checker is not None:
        for i, q in enumerate(waypoints):
            if not collision_checker(q):
                return PlanningResult(
                    success=False,
                    message=f"Waypoint {i} in collision",
                )

    traj = generate_time_scaled_trajectory(waypoints)
    return PlanningResult(success=True, trajectory=traj, message="OK")
