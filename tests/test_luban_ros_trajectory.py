import numpy as np
import pytest

from tools.luban_ros_trajectory_execute import read_trajectory


def test_read_trajectory_validates_synchronized_shapes(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, arm_positions=np.zeros((3, 7)), hand_positions=np.ones((3, 16)), timestamps=[0.0, 0.1, 0.2])
    result = read_trajectory(path)
    assert result["arm"].shape == (3, 7)
    assert result["hand"].shape == (3, 16)


def test_read_trajectory_rejects_non_monotonic_timestamps(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, arm_positions=np.zeros((2, 7)), hand_positions=np.zeros((2, 16)), timestamps=[0.2, 0.1])
    with pytest.raises(ValueError, match="non-decreasing"):
        read_trajectory(path)
