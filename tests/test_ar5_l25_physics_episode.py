import mujoco
import numpy as np

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES
from tools.run_ar5_l25_physics_episode import (
    add_position_servos,
    apply_l25_mimic_qpos,
    failure_reason,
    right_command_target,
    warning_counts,
)


def test_l25_mimics_follow_luban_coupling() -> None:
    class Data:
        qpos = np.zeros(10)

    addresses = {
        "r_hand_thumb_mcp": 0, "r_hand_thumb_dip": 1,
        "r_hand_index_pip": 2, "r_hand_index_dip": 3,
        "r_hand_middle_pip": 4, "r_hand_middle_dip": 5,
        "r_hand_ring_pip": 6, "r_hand_ring_dip": 7,
        "r_hand_pinky_pip": 8, "r_hand_pinky_dip": 9,
    }
    data = Data()
    data.qpos[[0, 2, 4, 6, 8]] = (0.2, 0.3, 0.4, 0.5, 0.6)
    apply_l25_mimic_qpos(data, addresses)
    np.testing.assert_allclose(data.qpos[[1, 3, 5, 7, 9]], (0.206, 0.267, 0.356, 0.445, 0.534))

def test_position_servo_is_added_to_temporary_scene(tmp_path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        "<mujoco><worldbody><body><joint name=\"r_joint_1\" type=\"hinge\" range=\"-1 1\"/>"
        "<geom type=\"capsule\" size=\"0.01 0.1\" mass=\"0.1\"/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    source = mujoco.MjModel.from_xml_path(str(scene))
    add_position_servos(scene, source, ("r_joint_1",))
    controlled = mujoco.MjModel.from_xml_path(str(scene))
    assert controlled.nu == 1
    actuator = mujoco.mj_name2id(controlled, mujoco.mjtObj.mjOBJ_ACTUATOR, "anydex_servo__r_joint_1")
    assert actuator >= 0
    assert controlled.actuator_forcelimited[actuator]


def test_right_target_preserves_left_side_and_applies_l25_mimics() -> None:
    names = (*AR5_RIGHT_JOINT_NAMES, *(f"r_hand_{name}" for name in L25_QPOS_JOINTS if name != "thumb_ip"), "r_hand_thumb_dip")
    addresses = {name: index for index, name in enumerate(names)}
    initial = np.full(len(names), -0.25)
    arm = np.linspace(-0.2, 0.2, len(AR5_RIGHT_JOINT_NAMES))
    hand = np.linspace(0.0, 0.8, len(L25_QPOS_JOINTS))
    target = right_command_target(initial, addresses, arm, hand)
    np.testing.assert_allclose(target[:7], arm)
    assert target[addresses["r_hand_thumb_dip"]] == target[addresses["r_hand_thumb_mcp"]] * 1.03
    assert target[addresses["r_hand_index_dip"]] == target[addresses["r_hand_index_pip"]] * 0.89


def test_warning_counts_and_failure_reason_mark_instability() -> None:
    class Warning:
        def __init__(self, number: int) -> None:
            self.number = number

    class Data:
        warning = [Warning(0) for _ in range(7)]

    data = Data()
    data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
    assert warning_counts(data)["bad_qacc"] == 1
    assert failure_reason(False, True, True, True, False, 1.0, False, 0.02, 0.0, False, 0.01, 0.03) == "mujoco_numerical_instability"


def test_arm_object_contact_rejects_otherwise_valid_lift() -> None:
    assert failure_reason(True, True, True, True, True, 1.0, False, 0.02, 0.0, False, 0.01, 0.03) == "unexpected_arm_object_contact"
