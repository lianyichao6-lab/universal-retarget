import numpy as np

from tools.plan_ar5_l25_cumotion import _hand_timeline, _trajectory_from_result


def test_trajectory_result_reorders_luban_joint_names() -> None:
    result = {
        "success": True,
        "trajectory": {
            "joint_names": ["r_joint_7", "r_joint_1", "r_joint_2", "r_joint_3", "r_joint_4", "r_joint_5", "r_joint_6"],
            "points": [{"positions": [7, 1, 2, 3, 4, 5, 6]}],
        },
    }
    trajectory = _trajectory_from_result(result, np.zeros(7))
    np.testing.assert_array_equal(trajectory[-1], [1, 2, 3, 4, 5, 6, 7])


def test_hand_timeline_holds_final_grasp_after_close_and_lift() -> None:
    hand, phase = _hand_timeline(
        np.asarray(["pregrasp", "approach", "close", "close", "close", "close_hold", "lift", "lift_hold"]),
        np.ones(21, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        close_frames=3,
    )
    assert hand.shape == (8, 21)
    assert phase.tolist() == ["pregrasp", "approach", "close", "close", "close", "close_hold", "lift", "lift_hold"]
    np.testing.assert_allclose(hand[5:], 1.0)
