import numpy as np
import chess

from chess_arm.chess.move_mapping import (
    make_move_waypoints,
    move_to_square_names,
    uci_to_square_names,
)
from chess_arm.utils.transforms import BoardCalibration


def _simple_calib() -> BoardCalibration:
    return BoardCalibration(
        origin_world=np.zeros(3),
        x_axis_world=np.array([1.0, 0.0, 0.0]),
        y_axis_world=np.array([0.0, 1.0, 0.0]),
        square_size=0.1,
        piece_height=0.02,
    )


def test_waypoints_different_for_from_and_to():
    calib = _simple_calib()
    waypoints = make_move_waypoints(
        "a1",
        "b1",
        calib,
        approach_offset=0.1,
        retreat_offset=0.1,
    )
    assert not np.allclose(waypoints.grasp_from, waypoints.place_to)


def test_move_to_square_names_from_uci():
    move = chess.Move.from_uci("e2e4")
    from_sq, to_sq = move_to_square_names(move)
    assert from_sq == "e2"
    assert to_sq == "e4"


def test_uci_to_square_names():
    from_sq, to_sq = uci_to_square_names("g1f3")
    assert from_sq == "g1"
    assert to_sq == "f3"
