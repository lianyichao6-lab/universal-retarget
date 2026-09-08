import numpy as np
import mujoco
import pytest

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import AR5_RIGHT_JOINT_NAMES
from tools.luban_mujoco_viewer import (
    apply_joint_state,
    apply_offline_frame,
    load_offline_episode,
    model_joint_qpos_addresses,
)


def test_viewer_applies_joint_states_by_name():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="r_joint_1" type="hinge"/><geom type="sphere" size="0.01" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    addresses = model_joint_qpos_addresses(model)
    matched = apply_joint_state(model, data, addresses, ["unknown", "r_joint_1"], [1.0, 0.25])
    assert matched == 1
    assert data.qpos[addresses["r_joint_1"]] == 0.25


def test_viewer_rejects_misaligned_positions():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    data = mujoco.MjData(model)
    with np.testing.assert_raises(ValueError):
        apply_joint_state(model, data, {}, ["r_joint_1"], [])


def _write_arm(path, frames: int = 2) -> None:
    np.savez(path, joint_names=np.asarray(AR5_RIGHT_JOINT_NAMES), positions=np.zeros((frames, 7)), timestamps=np.arange(frames))


def _write_hand(path, frames: int = 2) -> None:
    np.savez(path, qpos=np.zeros((frames, 21)), timestamps=np.arange(frames), object_position=np.zeros((frames, 3)))


def test_load_offline_episode_requires_synchronized_frames(tmp_path) -> None:
    arm_path = tmp_path / "arm.npz"
    hand_path = tmp_path / "hand.npz"
    _write_arm(arm_path)
    _write_hand(hand_path)
    arm, hand, object_positions = load_offline_episode(arm_path, hand_path)
    assert arm.shape == (2, 7)
    assert hand.shape == (2, 21)
    assert object_positions.shape == (2, 3)


def test_load_offline_episode_rejects_mismatched_frames(tmp_path) -> None:
    arm_path = tmp_path / "arm.npz"
    hand_path = tmp_path / "hand.npz"
    _write_arm(arm_path, frames=2)
    _write_hand(hand_path, frames=3)
    with pytest.raises(ValueError, match="frame counts differ"):
        load_offline_episode(arm_path, hand_path)


def test_apply_offline_frame_sets_arm_hand_and_object() -> None:
    class Data:
        qpos = np.zeros(31)

    addresses = {name: index for index, name in enumerate(AR5_RIGHT_JOINT_NAMES)}
    addresses.update({f"r_hand_{name}": index + 7 for index, name in enumerate(L25_QPOS_JOINTS)})
    addresses["anydex_episode_object_freejoint"] = 28
    arm = np.arange(7, dtype=np.float64)[None]
    hand = np.arange(21, dtype=np.float64)[None]
    object_positions = np.asarray([[0.1, 0.2, 0.3]])
    data = Data()
    apply_offline_frame(None, data, addresses, arm, hand, object_positions, 0)
    assert np.array_equal(data.qpos[:7], arm[0])
    assert np.array_equal(data.qpos[7:28], hand[0])
    assert np.array_equal(data.qpos[28:31], object_positions[0])
