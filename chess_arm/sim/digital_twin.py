from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from chess_arm.control.motion_planner import plan_joint_space_path
from chess_arm.control.controllers import JointTrajectoryExecutor
from chess_arm.chess.move_mapping import make_move_waypoints
from chess_arm.utils.transforms import BoardCalibration

try:
    # Isaac Sim Franka wrapper
    from omni.isaac.franka import Franka
    from omni.isaac.core import World
    from omni.isaac.core.utils.types import ArticulationAction
except ImportError:
    Franka = object  # type: ignore[assignment]
    World = object   # type: ignore[assignment]
    ArticulationAction = object  # type: ignore[assignment]


@dataclass
class FrankaChessDigitalTwin:
    """High-level wrapper around a Franka Panda in Isaac Sim."""

    world: World
    franka: Franka
    joint_names: List[str]
    calib: BoardCalibration

    _traj_executor: Optional[JointTrajectoryExecutor] = None

    @classmethod
    def create(cls, world: World, config: dict, calib: BoardCalibration) -> "FrankaChessDigitalTwin":
        prim_path = config["franka"]["prim_path"]
        default_q = np.asarray(config["franka"]["default_joint_positions"], dtype=float)

        if Franka is object:
            franka = object()
            joint_names = [f"joint_{i}" for i in range(default_q.shape[0])]
            return cls(world, franka, joint_names, calib)

        franka = Franka(
            prim_path=prim_path,
            name="franka_panda",
        )
        world.scene.add(franka)
        joint_names = franka.dof_names
        franka.set_joint_positions(default_q)
        return cls(world, franka, joint_names, calib)

    # --- basic articulation APIs -------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        if Franka is object:
            raise RuntimeError("Not running inside Isaac Sim")
        return self.franka.get_joint_positions()

    def set_joint_positions(self, q: np.ndarray) -> None:
        if Franka is object:
            raise RuntimeError("Not running inside Isaac Sim")
        self.franka.set_joint_positions(q)

    # --- chess-specific high-level commands -------------------------------------

    def plan_move_squares(
        self,
        from_square: str,
        to_square: str,
        max_joint_vel: float = 1.0,
    ) -> None:
        """
        Plan a full pick-and-place for a chess move.
        Stores a JointTrajectoryExecutor internally.
        """
        waypoints = make_move_waypoints(
            from_square=from_square,
            to_square=to_square,
            calib=self.calib,
            approach_offset=0.10,
            retreat_offset=0.10,
        )

        # For now: plan a trivial joint path between current configuration and goal (no IK here).
        # todo: replace with IK-based mapping from task-space waypoints to joint-space waypoints.

        q_current = self.get_joint_positions()
        # Stub: just reuse q_current -> no-op path (you'll add IK here).
        planning_result = plan_joint_space_path(q_current, q_current)

        if not planning_result.success or planning_result.trajectory is None:
            raise RuntimeError(f"Planning failed: {planning_result.message}")

        self._traj_executor = JointTrajectoryExecutor(
            joint_names=self.joint_names,
            trajectory=planning_result.trajectory,
        )

    def step(self, dt: float) -> None:
        """
        To be called each physics step; applies ArticulationAction if a trajectory is active.
        """
        if self._traj_executor is None:
            return

        action, done = self._traj_executor.step(dt)
        if ArticulationAction is object:
            return  # outside Isaac Sim, nothing to do
        self.franka.apply_action(action)

        if done:
            self._traj_executor = None
