import numpy as np
import pytest

from chess_arm.utils.transforms import (
    BoardCalibration,
    square_to_indices,
    indices_to_square,
    square_center_world,
)


def test_square_to_indices_basic():
    assert square_to_indices("a1") == (0, 0)
    assert square_to_indices("h8") == (7, 7)
    assert square_to_indices("e4") == (4, 3)


def test_square_to_indices_invalid():
    with pytest.raises(ValueError):
        square_to_indices("i1")
    with pytest.raises(ValueError):
        square_to_indices("a9")
    with pytest.raises(ValueError):
        square_to_indices("e")


def test_indices_to_square_roundtrip():
    for file_idx in range(8):
        for rank_idx in range(8):
            sq = indices_to_square(file_idx, rank_idx)
            fi, ri = square_to_indices(sq)
            assert fi == file_idx
            assert ri == rank_idx


def test_board_calibration_axes_orthonormal():
    calib = BoardCalibration(
        origin_world=np.zeros(3),
        x_axis_world=np.array([1.0, 0.0, 0.0]),
        y_axis_world=np.array([0.0, 1.0, 0.0]),
        square_size=0.1,
        piece_height=0.02,
    )

    x = calib.x_axis_world
    y = calib.y_axis_world
    z = calib.z_axis_world

    assert np.allclose(np.linalg.norm(x), 1.0)
    assert np.allclose(np.linalg.norm(y), 1.0)
    assert np.allclose(np.linalg.norm(z), 1.0)
    assert np.allclose(np.dot(x, y), 0.0, atol=1e-6)
    assert np.allclose(np.dot(x, z), 0.0, atol=1e-6)
    assert np.allclose(np.dot(y, z), 0.0, atol=1e-6)


def test_square_center_world_simple_plane():
    calib = BoardCalibration(
        origin_world=np.zeros(3),
        x_axis_world=np.array([1.0, 0.0, 0.0]),
        y_axis_world=np.array([0.0, 1.0, 0.0]),
        square_size=0.1,
        piece_height=0.02,
    )

    c_a1 = square_center_world("a1", calib)
    c_b1 = square_center_world("b1", calib)
    c_a2 = square_center_world("a2", calib)

    # a1 center at (0.05, 0.05, height) in this configuration
    assert np.allclose(c_a1, np.array([0.05, 0.05, 0.02]))
    # moving along file (x-direction)
    assert np.allclose(c_b1 - c_a1, np.array([0.1, 0.0, 0.0]))
    # moving along rank (y-direction)
    assert np.allclose(c_a2 - c_a1, np.array([0.0, 0.1, 0.0]))
