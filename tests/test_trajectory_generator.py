import numpy as np

from chess_arm.control.trajectory_generator import generate_time_scaled_trajectory


def test_generate_time_scaled_trajectory_monotonic_time():
    w0 = np.zeros(3)
    w1 = np.ones(3)
    traj = generate_time_scaled_trajectory([w0, w1], max_joint_vel=1.0, dt=0.1)
    assert np.all(np.diff(traj.times) > 0.0)
    assert traj.positions.shape[1] == 3
