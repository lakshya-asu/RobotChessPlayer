from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    # Isaac Sim motion generation and Lula kinematics
    from omni.isaac.motion_generation import (  # type: ignore[import]
        ArticulationKinematicsSolver,
        LulaKinematicsSolver,
        interface_config_loader,
    )
except ImportError:  # pragma: no cover
    ArticulationKinematicsSolver = None  # type: ignore[assignment]
    LulaKinematicsSolver = None  # type: ignore[assignment]
    interface_config_loader = None  # type: ignore[assignment]


@dataclass
class IKSolver:
    """
    Lula-based IK solver for the Franka Panda in Isaac Sim.

    This is a thin wrapper around ArticulationKinematicsSolver and LulaKinematicsSolver.
    """

    franka: object
    ee_frame_name: str = "right_gripper"

    _lula_solver: Optional[object] = None
    _art_solver: Optional[object] = None

    def __post_init__(self) -> None:
        if (
            ArticulationKinematicsSolver is None
            or LulaKinematicsSolver is None
            or interface_config_loader is None
        ):
            # IK not available in this environment
            return

        try:
            config = interface_config_loader.load_supported_lula_kinematics_solver_config(
                "Franka"
            )
            self._lula_solver = LulaKinematicsSolver(**config)
            self._art_solver = ArticulationKinematicsSolver(
                self.franka, self._lula_solver, self.ee_frame_name
            )
        except Exception:
            # todo: add structured logging for IK initialization failure
            self._lula_solver = None
            self._art_solver = None

    def is_available(self) -> bool:
        return self._art_solver is not None and self._lula_solver is not None

    def solve(
        self,
        target_pos_world: np.ndarray,
        target_rot_wxyz: np.ndarray,
    ) -> np.ndarray:
        """
        Solve IK for a target pose in world coordinates.

        Returns:
            Joint positions as a numpy array if IK succeeds.

        Raises:
            RuntimeError if IK is not available or fails to converge.
        """
        if not self.is_available():
            raise RuntimeError("IK solver is not available")

        target_pos_world = np.asarray(target_pos_world, dtype=float).reshape(3)
        target_rot_wxyz = np.asarray(target_rot_wxyz, dtype=float).reshape(4)

        # Update robot base pose for the Lula solver
        base_pos, base_rot = self.franka.get_world_pose()
        self._lula_solver.set_robot_base_pose(base_pos, base_rot)

        action, success = self._art_solver.compute_inverse_kinematics(
            target_pos_world, target_rot_wxyz
        )

        if not success:
            raise RuntimeError("IK computation did not converge")

        q = np.asarray(action.joint_positions, dtype=float)
        return q
