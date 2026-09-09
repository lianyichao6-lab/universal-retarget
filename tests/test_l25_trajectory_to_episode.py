import pickle

import numpy as np
import pytest

from tools.l25_trajectory_to_episode import load_l25_trajectory


def test_load_l25_trajectory_reads_staged_targets(tmp_path) -> None:
    path = tmp_path / "trajectory.pkl"
    with path.open("wb") as stream:
        pickle.dump([{"target": np.zeros(21), "phase": "pregrasp"}], stream)
    qpos, phase = load_l25_trajectory(path)
    assert qpos.shape == (1, 21)
    assert phase.tolist() == ["pregrasp"]


def test_load_l25_trajectory_rejects_wrong_target_shape(tmp_path) -> None:
    path = tmp_path / "trajectory.pkl"
    with path.open("wb") as stream:
        pickle.dump([{"target": np.zeros(20)}], stream)
    with pytest.raises(ValueError, match="shape"):
        load_l25_trajectory(path)
