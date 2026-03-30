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
    assert calibration.square_to_position("b1") == (1.5, 2.0, 0.3)
    assert calibration.square_to_position("a2") == (1.0, 2.5, 0.3)


def test_black_robot_positions_are_mirrored_about_board_center():
    calibration = BoardCalibration(origin_x=1.0, origin_y=2.0, square_size_m=0.5, board_z_m=0.3)

    assert calibration.square_to_position("a1", robot_side="black") == calibration.square_to_position(
        "h8"
    )
    assert calibration.square_to_position("e7", robot_side="black") == calibration.square_to_position(
        "d2"
    )


def test_black_graveyard_dropoff_is_mirrored_about_board_center():
    calibration = BoardCalibration(origin_x=1.0, origin_y=2.0, square_size_m=0.5, board_z_m=0.3)

    white_drop = calibration.captured_piece_dropoff(3, z_offset=0.1, robot_side="white")
    black_drop = calibration.captured_piece_dropoff(3, z_offset=0.1, robot_side="black")

    assert black_drop[0] == 2.0 * calibration.center_x - white_drop[0]
    assert black_drop[1] == 2.0 * calibration.center_y - white_drop[1]
    assert black_drop[2] == white_drop[2]
