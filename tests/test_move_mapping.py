import numpy as np

from chess_arm.chess.move_mapping import make_move_waypoints
from chess_arm.utils.transforms import BoardCalibration


def test_waypoints_different_for_from_and_to():
    calib = BoardCalibration(
        origin_world=np.zeros(3),
        x_axis_world=np.array([1.0, 0.0, 0.0]),
        y_axis_world=np.array([0.0, 1.0, 0.0]),
        square_size=0.1,
        piece_height=0.02,
    )
    waypoints = make_move_waypoints("a1", "b1", calib, approach_offset=0.1, retreat_offset=0.1)
    assert not (waypoints.grasp_from == waypoints.place_to).all()
