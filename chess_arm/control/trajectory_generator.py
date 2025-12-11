from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class JointTrajectory:
    times: np.ndarray           # shape (N,)
    positions: np.ndarray       # shape (N, dof)

    def sample(self, t: float) -> np.ndarray:
        """Linearly interpolate joint positions at time t."""
        if t <= self.times[0]:
            return self.positions[0]
        if t >= self.times[-1]:
            return self.positions[-1]
        idx = np.searchsorted(self.times, t) - 1
        t0, t1 = self.times[idx], self.times[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]


def generate_time_scaled_trajectory(
    waypoints: Sequence[np.ndarray],
    max_joint_vel: float = 1.0,
    dt: float = 0.01,
) -> JointTrajectory:
    """
    Very simple time-scaling: straight-line segments in joint space,
    segment duration chosen from max joint delta / max_joint_vel.
    """
    waypoints = [np.asarray(w, dtype=float) for w in waypoints]
    dof = waypoints[0].shape[0]
    assert all(w.shape == (dof,) for w in waypoints)

    times = [0.0]
    for i in range(1, len(waypoints)):
        dq = np.abs(waypoints[i] - waypoints[i - 1])
        dt_segment = float(np.max(dq) / max_joint_vel) if max_joint_vel > 0 else 1.0
        times.append(times[-1] + max(dt_segment, dt))  # at least one step

    full_times = [times[0]]
    full_positions = [waypoints[0]]

    for i in range(1, len(waypoints)):
        t0, t1 = times[i - 1], times[i]
        q0, q1 = waypoints[i - 1], waypoints[i]
        n_steps = max(2, int((t1 - t0) / dt))
        for k in range(1, n_steps):
            alpha = k / (n_steps - 1)
            full_times.append(t0 + alpha * (t1 - t0))
            full_positions.append((1 - alpha) * q0 + alpha * q1)

    return JointTrajectory(
        times=np.asarray(full_times),
        positions=np.stack(full_positions, axis=0),
    )
