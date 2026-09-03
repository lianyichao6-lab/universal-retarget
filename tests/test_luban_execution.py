import numpy as np

from anydexretarget.luban_execution import build_luban_grasp_request
from tools.luban_ros_grasp_execute import _controller_state_converged


def test_controller_state_fallback_requires_new_converged_reference() -> None:
    previous = np.asarray((0.0, 0.0, 0.0))
    current = np.asarray((0.1, 0.0, 0.0))
    assert _controller_state_converged(
        previous, current, np.asarray((0.001, 0.0, 0.0)), 0.01
    )
    assert not _controller_state_converged(
        previous, previous, np.zeros(3), 0.01
    )
    assert not _controller_state_converged(
        previous, current, np.asarray((0.02, 0.0, 0.0)), 0.01
    )


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
