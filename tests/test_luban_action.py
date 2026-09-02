import numpy as np
import pytest

from anydexretarget.luban_action import action_contract, build_luban_action


def test_build_luban_action_converts_l25_and_keeps_arm_order() -> None:
    action = build_luban_action(
        np.arange(7, dtype=np.float64),
        np.arange(21, dtype=np.float64),
        time_from_start_s=0.25,
    )
    np.testing.assert_array_equal(action.arm_positions, np.arange(7))
    assert action.hand_positions.shape == (16,)
    assert action.time_from_start_s == 0.25


def test_action_contract_exposes_luban_names() -> None:
    contract = action_contract()
    assert len(contract["arm_joint_names"]) == 7
    assert len(contract["hand_joint_names"]) == 16
    assert contract["arm_action"].endswith("follow_joint_trajectory")


def test_build_luban_action_rejects_bad_arm_shape() -> None:
    with pytest.raises(ValueError, match="arm_positions"):
        build_luban_action(np.zeros(6), np.zeros(21), time_from_start_s=0.0)
