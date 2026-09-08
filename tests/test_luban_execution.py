import numpy as np

from anydexretarget.luban_execution import build_luban_grasp_request


def test_request_uses_capture_pose_and_active_l25_joints() -> None:
    contract = {
        "anchor_frame": np.asarray("hand_camera_color_optical_frame"),
        "candidate_id": np.asarray("candidate_017"),
        "l25_qpos": np.arange(21, dtype=np.float32),
        "T_anchor_l25_hand": np.eye(4),
    }
    capture = np.eye(4)
    capture[:3, 3] = (1.0, 2.0, 3.0)
    request = build_luban_grasp_request(
        contract,
        t_robot_base_anchor_capture=capture,
        t_arm_flange_l25_hand=np.eye(4),
        base_frame="r_base_link",
        expected_anchor_frame="hand_camera_color_optical_frame",
        pregrasp_offset_hand_m=(0.0, 0.0, -0.1),
    )
    np.testing.assert_allclose(request["T_robot_base_arm_flange_target"], capture)
    np.testing.assert_allclose(
        request["T_robot_base_arm_flange_pregrasp"][:3, 3], (1.0, 2.0, 2.9)
    )
    assert request["l25_qpos"].shape == (21,)
    assert request["l25_active_positions"].shape == (16,)
    assert request["candidate_id"].item() == "candidate_017"


def test_request_uses_explicit_lift_direction() -> None:
    contract = {
        "anchor_frame": np.asarray("hand_camera_color_optical_frame"),
        "l25_qpos": np.zeros(21, dtype=np.float32),
        "T_anchor_l25_hand": np.eye(4),
    }
    request = build_luban_grasp_request(
        contract,
        t_robot_base_anchor_capture=np.eye(4),
        t_arm_flange_l25_hand=np.eye(4),
        base_frame="r_base_link",
        pregrasp_offset_hand_m=(0.0, 0.0, -0.1),
        lift_direction_base=(0.0, 1.0, 0.0),
    )
    np.testing.assert_allclose(request["T_robot_base_arm_flange_lift"][:3, 3], (0.0, 0.03, 0.0))
    np.testing.assert_allclose(request["lift_direction_base"], (0.0, 1.0, 0.0))
