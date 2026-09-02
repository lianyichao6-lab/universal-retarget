import numpy as np
import mujoco

from tools.luban_mujoco_viewer import apply_joint_state, model_joint_qpos_addresses


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
