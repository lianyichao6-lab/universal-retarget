import numpy as np
import pytest

from anydexretarget.tactile_adapter import TactileState, tactile_state_from_wrenches


def test_wrench_adapter_marks_contact_per_finger() -> None:
    wrenches = np.zeros((5, 6), dtype=np.float32)
    wrenches[0, 2] = 0.2
    wrenches[3, 0] = 0.11
    state = tactile_state_from_wrenches(wrenches, contact_force_threshold=0.1)
    np.testing.assert_array_equal(state.contact, [True, False, False, True, False])
    assert state.contact_count == 2


def test_tactile_state_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="wrenches"):
        TactileState(0.0, np.zeros(5), np.zeros((4, 6)))


def test_tactile_state_observation_is_copied() -> None:
    state = tactile_state_from_wrenches(np.zeros((5, 6)))
    observation = state.as_observation()
    observation["contact"][0] = True
    assert not state.contact[0]
