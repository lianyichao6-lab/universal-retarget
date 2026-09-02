import numpy as np

from tools.run_ar5_l25_mujoco_episode import _read_arm_trajectory


def test_read_arm_trajectory_accepts_luban_contract(tmp_path) -> None:
    path = tmp_path / "arm.npz"
    names = np.asarray(["r_joint_7", "r_joint_1", "r_joint_2", "r_joint_3", "r_joint_4", "r_joint_5", "r_joint_6"])
    positions = np.arange(14, dtype=np.float64).reshape(2, 7)
    np.savez(path, joint_names=names, positions=positions, timestamps=np.array([0.0, 0.05]))

    result, timestamps = _read_arm_trajectory(path)

    np.testing.assert_array_equal(result[0], [1, 2, 3, 4, 5, 6, 0])
    np.testing.assert_allclose(timestamps, [0.0, 0.05])
