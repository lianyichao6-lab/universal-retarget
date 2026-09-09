import numpy as np

from tools.make_ar5_l25_sim_calibration import build_sim_calibration


class _FakeFk:
    def flange_transform(self, positions):
        result = np.eye(4)
        result[:3, 3] = np.asarray(positions)[:3]
        return result


def test_sim_calibration_places_hug_hand_at_arm_flange() -> None:
    contract = {"T_anchor_l25_hand": np.eye(4)}
    flange_hand = np.eye(4)
    result = build_sim_calibration(contract, flange_hand, [1, 2, 3, 0, 0, 0, 0], _FakeFk())
    np.testing.assert_allclose(result["T_robot_base_anchor_capture"][:3, 3], [1, 2, 3])
    np.testing.assert_allclose(result["T_robot_base_l25_hand_target"], result["T_robot_base_arm_flange_target"])
    assert bool(result["simulation_only"])
