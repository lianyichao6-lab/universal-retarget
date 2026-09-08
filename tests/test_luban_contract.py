import numpy as np
import pytest

from anydexretarget.luban_contract import (
    AR5_RIGHT_JOINT_NAMES,
    L25_ACTIVE_INDICES,
    l25_active_joint_names,
    l25_qpos_to_luban_active,
)


def test_luban_contract_has_ar5_seven_joints_and_l25_sixteen_active() -> None:
    assert AR5_RIGHT_JOINT_NAMES == tuple(f"r_joint_{i}" for i in range(1, 8))
    assert L25_ACTIVE_INDICES.shape == (16,)
    assert l25_active_joint_names()[-1] == "r_hand_pinky_pip"


def test_l25_qpos_extracts_active_joints_in_luban_order() -> None:
    qpos = np.arange(21, dtype=np.float64)
    result = l25_qpos_to_luban_active(qpos)
    np.testing.assert_array_equal(result, qpos[L25_ACTIVE_INDICES])
    assert result.shape == (16,)


def test_l25_qpos_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        l25_qpos_to_luban_active(np.zeros(16))
