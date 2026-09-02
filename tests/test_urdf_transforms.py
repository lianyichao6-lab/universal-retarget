import numpy as np

from anydexretarget.urdf_transforms import find_fixed_joint_transform, rpy_to_matrix


def test_rpy_to_matrix_z_rotation() -> None:
    np.testing.assert_allclose(rpy_to_matrix([0.0, 0.0, -np.pi / 2]), [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-7)


def test_find_fixed_joint_transform(tmp_path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><joint name="mount" type="fixed"><parent link="r_tcp"/><child link="r_hand_base_link"/><origin xyz="0 0 0.1" rpy="0 0 -1.5708"/></joint></robot>', encoding="utf-8")
    transform = find_fixed_joint_transform(urdf, parent_link="r_tcp", child_link="r_hand_base_link")
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(transform[0, 1], 1.0, atol=1e-4)
