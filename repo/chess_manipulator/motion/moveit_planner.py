"""MoveIt-backed stage planner for the Panda arm in ROS GZ."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Sequence, Tuple

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan, GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from chess_manipulator.chess.board import MoveDescription
from .board_calibration import BoardCalibration
from .planner import JOINT_NAMES, MovePlan, PlannedStage


TOP_DOWN_QUATERNION = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


@dataclass(frozen=True)
class MoveItPlannerConfig:
    """Runtime tuning for the MoveIt-backed planner."""

    group_name: str = "panda_arm"
    ik_link_name: str = "panda_link8"
    planning_frame: str = "world"
    planning_time_sec: float = 1.5
    velocity_scale: float = 0.2
    acceleration_scale: float = 0.2
    joint_tolerance: float = 0.01


class MoveItTrajectoryPlanner:
    """Plan stage-wise MoveIt trajectories from chess move descriptions."""

    def __init__(
        self,
        node: Node,
        calibration: BoardCalibration,
        hover_height_m: float = 0.12,
        pickup_height_m: float = 0.015,
        stage_time_sec: float = 1.0,
        config: Optional[MoveItPlannerConfig] = None,
    ) -> None:
        self._node = node
        self._calibration = calibration
        self._hover_height_m = hover_height_m
        self._pickup_height_m = pickup_height_m
        self._stage_time_sec = stage_time_sec
        self._config = config or MoveItPlannerConfig()
        self._joint_state: Optional[JointState] = None
        self._joint_state_subscription = node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            20,
        )
        self._ik_client = node.create_client(GetPositionIK, "/compute_ik")
        self._plan_client = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self._planning_scene_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._allowed_collisions_configured = False

    @property
    def stage_time_sec(self) -> float:
        return self._stage_time_sec

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state = msg

    def plan_move(self, description: MoveDescription, capture_slot: int = 0) -> MovePlan:
        self._wait_for_services()
        current_positions = self._current_joint_positions()
        return self.plan_move_from_positions(
            description,
            start_positions=current_positions,
            capture_slot=capture_slot,
        )

    def plan_move_from_positions(
        self,
        description: MoveDescription,
        start_positions: Sequence[float],
        capture_slot: int = 0,
    ) -> MovePlan:
        self._wait_for_services()
        current_positions = list(start_positions)
        stages: List[PlannedStage] = []
        combined = JointTrajectory()
        combined.joint_names = list(JOINT_NAMES)

        for name, pose in self._stage_specs(description, capture_slot):
            target_positions = self._solve_pose(pose, current_positions)
            segment = self._plan_segment(current_positions, target_positions)
            self._append_segment(combined, segment)
            stages.append(PlannedStage(name=name, pose=pose, joints=list(target_positions)))
            current_positions = list(target_positions)

        return MovePlan(move=description.uci, stages=stages, trajectory=combined)

    def _wait_for_services(self) -> None:
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_ik is not available.")
        if not self._plan_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/plan_kinematic_path is not available.")
        if not self._planning_scene_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/apply_planning_scene is not available.")
        self._configure_allowed_collisions_once()

    def _current_joint_positions(self) -> List[float]:
        if self._joint_state is None:
            raise RuntimeError("No /joint_states received for MoveIt planning.")
        positions = {name: position for name, position in zip(self._joint_state.name, self._joint_state.position)}
        missing = [name for name in JOINT_NAMES if name not in positions]
        if missing:
            raise RuntimeError(f"Joint state is missing Panda joints: {missing}")
        return [positions[name] for name in JOINT_NAMES]

    def _stage_specs(
        self,
        description: MoveDescription,
        capture_slot: int,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        stages: List[Tuple[str, Tuple[float, float, float]]] = []
        if description.is_capture:
            captured_hover = self._calibration.square_to_position(description.target, self._hover_height_m)
            captured_pick = self._calibration.square_to_position(description.target, self._pickup_height_m)
            captured_dropoff = self._calibration.captured_piece_dropoff(
                capture_slot,
                z_offset=self._hover_height_m,
            )
            captured_place = (
                captured_dropoff[0],
                captured_dropoff[1],
                self._calibration.board_z_m + self._pickup_height_m,
            )
            stages.extend(
                [
                    ("capture_approach", captured_hover),
                    ("capture_descend", captured_pick),
                    ("capture_lift", captured_hover),
                    ("capture_transit", captured_dropoff),
                    ("capture_place", captured_place),
                    ("capture_retreat", captured_dropoff),
                ]
            )

        source_hover = self._calibration.square_to_position(description.source, self._hover_height_m)
        source_pick = self._calibration.square_to_position(description.source, self._pickup_height_m)
        target_hover = self._calibration.square_to_position(description.target, self._hover_height_m)
        target_pick = self._calibration.square_to_position(description.target, self._pickup_height_m)
        stages.extend(
            [
                ("approach_source", source_hover),
                ("descend_source", source_pick),
                ("lift_source", source_hover),
                ("transit_target", target_hover),
                ("place_target", target_pick),
                ("retreat_target", target_hover),
            ]
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
            stages.extend(self._stage_specs(rook_desc, capture_slot))

        return stages

    def _solve_pose(
        self,
        pose_xyz: Tuple[float, float, float],
        start_positions: Sequence[float],
    ) -> List[float]:
        request = GetPositionIK.Request()
        request.ik_request.group_name = self._config.group_name
        request.ik_request.ik_link_name = self._config.ik_link_name
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = self._config.planning_frame
        request.ik_request.pose_stamped.pose = Pose(
            position=self._point_from_xyz(*pose_xyz),
            orientation=TOP_DOWN_QUATERNION,
        )
        request.ik_request.robot_state = self._robot_state(start_positions)
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(sec=max(int(self._config.planning_time_sec), 1))

        future = self._ik_client.call_async(request)
        self._spin_until_future(future, timeout_sec=self._config.planning_time_sec + 1.0)
        response = future.result()
        if response is None or response.error_code.val != 1:
            raise RuntimeError(f"MoveIt IK failed for pose {pose_xyz}: {getattr(response, 'error_code', None)}")

        positions = {name: position for name, position in zip(response.solution.joint_state.name, response.solution.joint_state.position)}
        return [positions[name] for name in JOINT_NAMES]

    def _plan_segment(
        self,
        start_positions: Sequence[float],
        goal_positions: Sequence[float],
    ) -> JointTrajectory:
        request = GetMotionPlan.Request()
        motion_request = MotionPlanRequest()
        motion_request.group_name = self._config.group_name
        motion_request.start_state = self._robot_state(start_positions)
        motion_request.goal_constraints = [self._goal_constraints(goal_positions)]
        motion_request.num_planning_attempts = 3
        motion_request.allowed_planning_time = self._config.planning_time_sec
        motion_request.max_velocity_scaling_factor = self._config.velocity_scale
        motion_request.max_acceleration_scaling_factor = self._config.acceleration_scale
        request.motion_plan_request = motion_request

        future = self._plan_client.call_async(request)
        self._spin_until_future(future, timeout_sec=self._config.planning_time_sec + 2.0)
        response = future.result()
        if response is None or response.motion_plan_response.error_code.val != 1:
            raise RuntimeError("MoveIt motion planning failed for a stage.")
        trajectory = response.motion_plan_response.trajectory.joint_trajectory
        if not trajectory.points:
            raise RuntimeError("MoveIt returned an empty joint trajectory.")
        return trajectory

    def _robot_state(self, positions: Sequence[float]) -> RobotState:
        joint_state = JointState()
        joint_state.name = list(JOINT_NAMES)
        joint_state.position = list(positions)
        return RobotState(joint_state=joint_state)

    def _goal_constraints(self, goal_positions: Sequence[float]) -> Constraints:
        constraints = Constraints()
        constraints.joint_constraints = []
        for name, position in zip(JOINT_NAMES, goal_positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = self._config.joint_tolerance
            constraint.tolerance_below = self._config.joint_tolerance
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        return constraints

    def _append_segment(self, combined: JointTrajectory, segment: JointTrajectory) -> None:
        if not combined.points:
            combined.points.extend(segment.points)
            return
        offset = self._duration_sec(combined.points[-1].time_from_start)
        for point in segment.points:
            shifted = JointTrajectoryPoint()
            shifted.positions = list(point.positions)
            shifted.velocities = list(point.velocities)
            shifted.accelerations = list(point.accelerations)
            shifted.effort = list(point.effort)
            shifted.time_from_start = self._duration_from_sec(
                offset + self._duration_sec(point.time_from_start)
            )
            combined.points.append(shifted)

    def _configure_allowed_collisions_once(self) -> None:
        if self._allowed_collisions_configured:
            return
        names = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_finger",
        ]
        allowed_pairs = {
            ("panda_link0", "panda_link1"),
            ("panda_link1", "panda_link2"),
            ("panda_link2", "panda_link3"),
            ("panda_link3", "panda_link4"),
            ("panda_link4", "panda_link5"),
            ("panda_link5", "panda_link6"),
            ("panda_link6", "panda_link7"),
            ("panda_link7", "panda_link8"),
            ("panda_link7", "panda_hand"),
            ("panda_link7", "panda_finger"),
            ("panda_link8", "panda_hand"),
            ("panda_link8", "panda_finger"),
            ("panda_hand", "panda_finger"),
        }
        matrix = []
        for row_name in names:
            enabled = []
            for col_name in names:
                if row_name == col_name:
                    enabled.append(False)
                    continue
                enabled.append((row_name, col_name) in allowed_pairs or (col_name, row_name) in allowed_pairs)
            matrix.append(AllowedCollisionEntry(enabled=enabled))

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.allowed_collision_matrix = AllowedCollisionMatrix(
            entry_names=names,
            entry_values=matrix,
        )
        request = ApplyPlanningScene.Request(scene=planning_scene)
        future = self._planning_scene_client.call_async(request)
        self._spin_until_future(future, timeout_sec=3.0)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Failed to configure MoveIt allowed-collision matrix.")
        self._allowed_collisions_configured = True

    def _spin_until_future(self, future, timeout_sec: float) -> None:
        deadline = self._node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while not future.done():
            time.sleep(0.05)
            if self._node.get_clock().now().nanoseconds >= deadline:
                raise RuntimeError("Timed out waiting for MoveIt planning service.")

    def _duration_sec(self, duration: Duration) -> float:
        return float(duration.sec) + (float(duration.nanosec) / 1e9)

    def _duration_from_sec(self, seconds: float) -> Duration:
        return Duration(
            sec=int(seconds),
            nanosec=int((seconds % 1.0) * 1e9),
        )

    def _point_from_xyz(self, x: float, y: float, z: float):
        from geometry_msgs.msg import Point

        return Point(x=float(x), y=float(y), z=float(z))
