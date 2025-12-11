from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import transforms3d.quaternions as tq

from chess_arm.control.motion_planner import plan_joint_space_path
from chess_arm.control.controllers import JointTrajectoryExecutor
from chess_arm.chess.move_mapping import make_move_waypoints
from chess_arm.utils.transforms import BoardCalibration
from chess_arm.control.ik import IKSolver
from chess_arm.control.trajectory_generator import generate_time_scaled_trajectory, JointTrajectory



try:
    # Isaac Sim Franka wrapper
    from omni.isaac.franka import Franka  # type: ignore[import]
    from omni.isaac.core import World  # type: ignore[import]
    from omni.isaac.core.utils.types import ArticulationAction  # type: ignore[import]
except ImportError:  # pragma: no cover
    Franka = object  # type: ignore[assignment]
    World = object  # type: ignore[assignment]
    ArticulationAction = object  # type: ignore[assignment]


@dataclass
class FrankaChessDigitalTwin:
    """High-level wrapper around a Franka Panda in Isaac Sim."""

    world: World
    franka: Franka
    joint_names: List[str]
    calib: BoardCalibration

    _traj_executor: Optional[JointTrajectoryExecutor] = None
    _ik_solver: Optional[IKSolver] = None

    def set_trajectory(self, trajectory: JointTrajectory) -> None:
        """
        Set a joint trajectory to be executed by the digital twin.
        """
        self._traj_executor = JointTrajectoryExecutor(
            joint_names=self.joint_names,
            trajectory=trajectory,
        )
        
    @classmethod
    def create(
        cls,
        world: World,
        config: dict,
        calib: BoardCalibration,
    ) -> "FrankaChessDigitalTwin":
        prim_path = config["franka"]["prim_path"]
        default_q = np.asarray(config["franka"]["default_joint_positions"], dtype=float)

        if Franka is object:
            franka = object()
            joint_names = [f"joint_{i}" for i in range(default_q.shape[0])]
            twin = cls(world, franka, joint_names, calib)
            return twin

        franka = Franka(
            prim_path=prim_path,
            name="franka_panda",
        )
        world.scene.add(franka)
        joint_names = franka.dof_names
        franka.set_joint_positions(default_q)

        twin = cls(world, franka, joint_names, calib)
        twin._ik_solver = IKSolver(franka=franka, ee_frame_name="right_gripper")
        return twin

    # --- basic articulation APIs -------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        if Franka is object:
            raise RuntimeError("Not running inside Isaac Sim")
        return self.franka.get_joint_positions()

    def set_joint_positions(self, q: np.ndarray) -> None:
        if Franka is object:
            raise RuntimeError("Not running inside Isaac Sim")
        self.franka.set_joint_positions(q)

    # --- internal helpers --------------------------------------------------------

    def _default_ee_orientation_wxyz(self) -> np.ndarray:
        """
        Construct a default EE orientation:

        - z-axis points down along negative board normal.
        - x-axis aligned with board x-axis.
        - y-axis computed as cross(z, x).
        """
        z = -self.calib.z_axis_world
        x = self.calib.x_axis_world

        x = x / np.linalg.norm(x)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)

        R = np.stack([x, y, z], axis=1)  # columns are basis vectors
        quat_wxyz = tq.mat2quat(R)
        return quat_wxyz

    # --- chess-specific high-level commands -------------------------------------

    def plan_move_squares(
        self,
        from_square: str,
        to_square: str,
        max_joint_vel: float = 1.0,
    ) -> None:
        """
        Plan a full pick-and-place for a chess move and store a JointTrajectoryExecutor.

        If IK is available, this uses Lula kinematics to generate joint-space waypoints
        for each board-space waypoint. If IK is not available, falls back to a trivial
        joint-space plan between the current configuration and itself.
        """
        waypoints = make_move_waypoints(
            from_square=from_square,
            to_square=to_square,
            calib=self.calib,
            approach_offset=0.10,
            retreat_offset=0.10,
        )

        q_current = self.get_joint_positions()

        if self._ik_solver is None or not self._ik_solver.is_available():
            # Fallback: trivial joint-space path (no actual motion).
            planning_result = plan_joint_space_path(q_current, q_current)
            if not planning_result.success or planning_result.trajectory is None:
                raise RuntimeError(f"Planning failed: {planning_result.message}")
            self._traj_executor = JointTrajectoryExecutor(
                joint_names=self.joint_names,
                trajectory=planning_result.trajectory,
            )
            return

        ee_rot = self._default_ee_orientation_wxyz()

        # Board-space positions for the pick-and-place
        positions = [
            waypoints.approach_from,
            waypoints.grasp_from,
            waypoints.lift_from,
            waypoints.approach_to,
            waypoints.place_to,
            waypoints.retreat_to,
        ]

        # Joint-space waypoints: start with the current configuration
        joint_waypoints = [q_current]

        for pos in positions:
            q_target = self._ik_solver.solve(
                target_pos_world=pos,
                target_rot_wxyz=ee_rot,
            )
            joint_waypoints.append(q_target)

        traj = generate_time_scaled_trajectory(
            waypoints=joint_waypoints,
            max_joint_vel=max_joint_vel,
        )

        self._traj_executor = JointTrajectoryExecutor(
            joint_names=self.joint_names,
            trajectory=traj,
        )

    def has_active_trajectory(self) -> bool:
        """
        Return True if a joint trajectory is currently being executed.
        """
        return self._traj_executor is not None

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
