import pytest

from chess_manipulator.chess.board import MoveDescription
from chess_manipulator.motion import BoardCalibration, JointLimitViolation, TrajectoryPlanner


def test_simple_move_generates_six_stages():
    planner = TrajectoryPlanner(BoardCalibration())

    plan = planner.plan_move(MoveDescription(uci="e2e4", source="e2", target="e4"))

    assert [stage.name for stage in plan.stages] == [
        "approach_source",
        "descend_source",
        "lift_source",
        "transit_target",
        "place_target",
        "retreat_target",
    ]
    assert len(plan.to_joint_trajectory(planner.stage_time_sec).points) == 6


def test_capture_move_includes_dropoff_sequence():
    planner = TrajectoryPlanner(BoardCalibration())

    plan = planner.plan_move(
        MoveDescription(
            uci="e4d5",
            source="e4",
            target="d5",
            is_capture=True,
        ),
        capture_slot=3,
    )

    assert plan.captured_dropoff is not None
    assert plan.stages[0].name == "capture_approach"
    assert "capture_place" in [stage.name for stage in plan.stages]


def test_castling_move_includes_prefixed_rook_stages():
    planner = TrajectoryPlanner(BoardCalibration())

    plan = planner.plan_move(
        MoveDescription(
            uci="e1g1",
            source="e1",
            target="g1",
            is_castling=True,
        )
    )

    stage_names = [stage.name for stage in plan.stages]
    assert "approach_source" in stage_names
    assert "castle_rook_approach_source" in stage_names
    assert any(name.startswith("castle_rook_") for name in stage_names)
    assert len(plan.stage_end_times_sec) == len(plan.stages)


def test_invalid_pose_is_rejected():
    planner = TrajectoryPlanner(BoardCalibration(origin_x=10.0))

    with pytest.raises(JointLimitViolation):
        planner.plan_move(MoveDescription(uci="a1a2", source="a1", target="a2"))
