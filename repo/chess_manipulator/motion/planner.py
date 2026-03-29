"""Move-level deterministic trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from chess_manipulator.chess.board import MoveDescription
from .board_calibration import BoardCalibration
from .kinematics import JointLimitViolation, PandaKinematicsModel


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


@dataclass
class PlannedStage:
    """One named motion stage with a cartesian target and joint solution."""

    name: str
    pose: Tuple[float, float, float]
    joints: List[float]


@dataclass
class MovePlan:
    """Full plan for a single chess move."""

    move: str
    stages: List[PlannedStage] = field(default_factory=list)
    captured_dropoff: Optional[Tuple[float, float, float]] = None
    trajectory: Optional[JointTrajectory] = None

    def to_joint_trajectory(self, stage_time_sec: float) -> JointTrajectory:
        if self.trajectory is not None:
            return self.trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES
        elapsed = 0.0
        for stage in self.stages:
            elapsed += stage_time_sec
            point = JointTrajectoryPoint()
            point.positions = stage.joints
            point.time_from_start = Duration(
                sec=int(elapsed),
                nanosec=int((elapsed % 1.0) * 1e9),
            )
            trajectory.points.append(point)
        return trajectory


class TrajectoryPlanner:
    """Turn a legal chess move into staged robot motion."""

    def __init__(
        self,
        calibration: BoardCalibration,
        hover_height_m: float = 0.12,
        pickup_height_m: float = 0.015,
        stage_time_sec: float = 1.0,
        kinematics: Optional[PandaKinematicsModel] = None,
    ) -> None:
        self._calibration = calibration
        self._hover_height_m = hover_height_m
        self._pickup_height_m = pickup_height_m
        self._stage_time_sec = stage_time_sec
        self._kinematics = kinematics or PandaKinematicsModel()

    @property
    def stage_time_sec(self) -> float:
        return self._stage_time_sec

    def plan_move(self, description: MoveDescription, capture_slot: int = 0) -> MovePlan:
        stages: List[PlannedStage] = []
        source_hover = self._calibration.square_to_position(description.source, self._hover_height_m)
        source_pick = self._calibration.square_to_position(description.source, self._pickup_height_m)
        target_hover = self._calibration.square_to_position(description.target, self._hover_height_m)
        target_pick = self._calibration.square_to_position(description.target, self._pickup_height_m)

        if description.is_capture:
            captured_pick = self._calibration.square_to_position(description.target, self._pickup_height_m)
            captured_hover = self._calibration.square_to_position(description.target, self._hover_height_m)
            captured_dropoff = self._calibration.captured_piece_dropoff(
                capture_slot, z_offset=self._hover_height_m
            )
            drop_pick = (
                captured_dropoff[0],
                captured_dropoff[1],
                self._calibration.board_z_m + self._pickup_height_m,
            )
            stages.extend(
                self._pose_stages(
                    (
                        ("capture_approach", captured_hover),
                        ("capture_descend", captured_pick),
                        ("capture_lift", captured_hover),
                        ("capture_transit", captured_dropoff),
                        ("capture_place", drop_pick),
                        ("capture_retreat", captured_dropoff),
                    )
                )
            )
        else:
            captured_dropoff = None

        stages.extend(
            self._pose_stages(
                (
                    ("approach_source", source_hover),
                    ("descend_source", source_pick),
                    ("lift_source", source_hover),
                    ("transit_target", target_hover),
                    ("place_target", target_pick),
                    ("retreat_target", target_hover),
                )
            )
        )

        if description.is_castling:
            rook_source = "h1" if description.target == "g1" else "a1"
            rook_target = "f1" if description.target == "g1" else "d1"
            if description.target in ("g8", "c8"):
                rook_source = "h8" if description.target == "g8" else "a8"
                rook_target = "f8" if description.target == "g8" else "d8"
            rook_desc = MoveDescription(
                uci=f"{rook_source}{rook_target}",
                source=rook_source,
                target=rook_target,
            )
            rook_plan = self.plan_move(rook_desc, capture_slot=capture_slot)
            stages.extend(rook_plan.stages)

        return MovePlan(move=description.uci, stages=stages, captured_dropoff=captured_dropoff)

    def _pose_stages(self, stage_specs: Iterable[Tuple[str, Tuple[float, float, float]]]) -> List[PlannedStage]:
        stages = []
        for name, pose in stage_specs:
            self._check_pose_collision(name, pose)
            joints = self._kinematics.solve_pose(pose)
            stages.append(PlannedStage(name=name, pose=pose, joints=joints))
        return stages

    def _check_pose_collision(self, stage_name: str, pose: Tuple[float, float, float]) -> None:
        x, y, z = pose
        if z < self._calibration.board_z_m:
            raise JointLimitViolation(f"{stage_name} collides with the board plane.")
        if x < 0.05 or y < -0.65 or y > 0.45:
            raise JointLimitViolation(f"{stage_name} exits the safe planning envelope.")
