import numpy as np

from anydexretarget.arm_fk import AR5ForwardKinematics, reorder_ar5_positions


URDF = "/home/evolabs-5080/lianyichao/luban_framework/src/robot_model/evo_work/AR5-5_08/AR5-5_08R-W4C4A6-ZY2_description/urdf/AR5-5_08R-W4C4A6-ZY2.urdf"


def test_reorder_ar5_positions() -> None:
    names = np.asarray(["r_joint_7", "r_joint_1", "r_joint_2", "r_joint_3", "r_joint_4", "r_joint_5", "r_joint_6"])
    values = np.arange(14, dtype=np.float64).reshape(2, 7)
    result = reorder_ar5_positions(names, values)
    np.testing.assert_array_equal(result[0], [1, 2, 3, 4, 5, 6, 0])


def test_luban_ar5_fk_returns_finite_seven_dof_pose() -> None:
    fk = AR5ForwardKinematics(URDF)
    transforms = fk.flange_transforms(np.zeros((2, 7)))
    assert transforms.shape == (2, 4, 4)
    assert np.isfinite(transforms).all()
    np.testing.assert_allclose(transforms[:, 3], [[0, 0, 0, 1], [0, 0, 0, 1]])
