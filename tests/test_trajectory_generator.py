import numpy as np

from chess_arm.control.trajectory_generator import (
    generate_time_scaled_trajectory,
    JointTrajectory,
)


def test_generate_time_scaled_trajectory_monotonic_time():
    w0 = np.zeros(3)
    w1 = np.ones(3)
    traj = generate_time_scaled_trajectory(
        [w0, w1],
        max_joint_vel=1.0,
        dt=0.1,
    )
    assert np.all(np.diff(traj.times) > 0.0)
    assert traj.positions.shape[1] == 3


def test_generate_time_scaled_trajectory_endpoints_match_waypoints():
    w0 = np.array([0.0, 0.0])
    w1 = np.array([0.5, -0.5])
    w2 = np.array([1.0, 0.0])
    traj = generate_time_scaled_trajectory(
        [w0, w1, w2],
        max_joint_vel=1.0,
        dt=0.05,
    )

    assert np.allclose(traj.positions[0], w0)
    assert np.allclose(traj.positions[-1], w2)


def test_joint_trajectory_sample_clamps_at_ends():
    times = np.array([0.0, 1.0, 2.0])
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    traj = JointTrajectory(times=times, positions=positions)

    before = traj.sample(-1.0)
    after = traj.sample(10.0)

    assert np.allclose(before, positions[0])
    assert np.allclose(after, positions[-1])

    mid = traj.sample(1.0)
    assert np.allclose(mid, positions[1])
