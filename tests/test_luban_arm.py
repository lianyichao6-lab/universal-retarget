import numpy as np

from anydexretarget.luban_arm import arm_flange_pose_xyzw, arm_flange_target


def _transform(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def test_arm_flange_target_uses_deployment_formula() -> None:
    contract = {"T_anchor_l25_hand": _transform(np.eye(3), [0.2, 0.0, 0.3])}
    base_anchor = _transform(np.eye(3), [1.0, 2.0, 0.0])
    flange_hand = _transform(np.eye(3), [0.0, 0.0, 0.1])
    target = arm_flange_target(
        contract,
        t_robot_base_anchor=base_anchor,
        t_arm_flange_l25_hand=flange_hand,
    )
    np.testing.assert_allclose(target[:3, 3], [1.2, 2.0, 0.2])


def test_pose_conversion_returns_ros_quaternion_order() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = _transform(rotation, [0.1, 0.2, 0.3])
    position, quaternion = arm_flange_pose_xyzw(target)
    np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(quaternion, [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
