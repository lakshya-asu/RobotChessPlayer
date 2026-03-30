from builtin_interfaces.msg import Duration
from chess_manipulator_msgs.msg import ExecutionCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from chess_manipulator.sim import (
    decode_isaac_result,
    decode_feedback,
    encode_isaac_result,
    encode_feedback,
    execution_command_duration_sec,
    trajectory_duration_sec,
)


def test_feedback_round_trip():
    payload = encode_feedback("completed", "isaac", "e2e4", 6.5, "ok")
    decoded = decode_feedback(payload)

    assert decoded["state"] == "completed"
    assert decoded["backend"] == "isaac"
    assert decoded["move"] == "e2e4"
    assert decoded["message"] == "ok"


def test_trajectory_duration_uses_last_point():
    trajectory = JointTrajectory()
    point = JointTrajectoryPoint()
    point.time_from_start = Duration(sec=3, nanosec=250000000)
    trajectory.points.append(point)

    assert trajectory_duration_sec(trajectory) == 3.25


def test_execution_command_duration_uses_embedded_trajectory():
    command = ExecutionCommand()
    command.move_id = "e2e4"
    point = JointTrajectoryPoint()
    point.time_from_start = Duration(sec=2, nanosec=500000000)
    command.trajectory.points.append(point)

    assert execution_command_duration_sec(command) == 2.5


def test_isaac_result_round_trip():
    payload = encode_isaac_result("completed", 1.25, "done")
    decoded = decode_isaac_result(payload)

    assert decoded["state"] == "completed"
    assert decoded["duration_sec"] == "1.250"
    assert decoded["message"] == "done"
