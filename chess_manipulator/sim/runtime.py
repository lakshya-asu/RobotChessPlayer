"""Runtime helpers shared by simulator bridge nodes and benchmarks."""

from __future__ import annotations

from typing import Any, Dict

from trajectory_msgs.msg import JointTrajectory


def trajectory_duration_sec(msg: JointTrajectory) -> float:
    """Return the duration represented by the final trajectory point."""
    if not msg.points:
        return 0.0
    point = msg.points[-1].time_from_start
    return float(point.sec) + (float(point.nanosec) / 1e9)


def execution_command_duration_sec(command: Any) -> float:
    """Return the duration represented by an execution command payload."""
    return trajectory_duration_sec(command.trajectory)


def encode_feedback(state: str, backend: str, move: str, duration_sec: float, message: str = "") -> str:
    """Encode execution feedback into a stable pipe-delimited payload."""
    return f"{state}|{backend}|{move}|{duration_sec:.3f}|{message}"


def decode_feedback(payload: str) -> Dict[str, str]:
    """Decode a pipe-delimited execution feedback payload."""
    parts = payload.split("|", 4)
    while len(parts) < 5:
        parts.append("")
    state, backend, move, duration_sec, message = parts
    return {
        "state": state,
        "backend": backend,
        "move": move,
        "duration_sec": duration_sec,
        "message": message,
    }


def encode_isaac_result(state: str, duration_sec: float, message: str = "") -> str:
    """Encode Isaac-side execution results for the ROS bridge shim."""
    return f"{state}|{duration_sec:.3f}|{message}"


def decode_isaac_result(payload: str) -> Dict[str, str]:
    """Decode the Isaac-side execution result payload."""
    parts = payload.split("|", 2)
    while len(parts) < 3:
        parts.append("")
    state, duration_sec, message = parts
    return {
        "state": state,
        "duration_sec": duration_sec,
        "message": message,
    }
