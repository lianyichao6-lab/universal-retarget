import numpy as np

from anydexretarget.hug_o30 import retarget_hug_o30
from anydexretarget.hand_contract import O30_ACTIVE_JOINT_NAMES, O30_QPOS_JOINT_NAMES


def test_hug_o30_retarget_produces_a_finite_controller_command() -> None:
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[:, 0] = np.linspace(0.0, 0.08, 21)
    keypoints[:, 1] = np.linspace(0.0, 0.04, 21)
    keypoints[:, 2] = np.linspace(0.0, 0.12, 21)
    result = retarget_hug_o30(keypoints)
    assert result.qpos.shape == (20,)
    assert result.command_positions.shape == (20,)
    assert np.isfinite(result.qpos).all()
    assert np.isfinite(result.command_positions).all()
    assert result.qpos_joint_names == O30_QPOS_JOINT_NAMES
    assert result.command_joint_names == O30_ACTIVE_JOINT_NAMES
