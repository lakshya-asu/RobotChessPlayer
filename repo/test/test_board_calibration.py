from chess_manipulator.motion import BoardCalibration


def test_all_square_positions_cover_entire_board():
    calibration = BoardCalibration()
    positions = calibration.all_square_positions()

    assert len(positions) == 64
    assert "a1" in positions
    assert "h8" in positions


def test_square_to_position_uses_board_origin_and_square_size():
    calibration = BoardCalibration(origin_x=1.0, origin_y=2.0, square_size_m=0.5, board_z_m=0.3)

    assert calibration.square_to_position("a1") == (1.0, 2.0, 0.3)
    assert calibration.square_to_position("b1") == (1.0, 2.5, 0.3)
    assert calibration.square_to_position("a2") == (1.5, 2.0, 0.3)
