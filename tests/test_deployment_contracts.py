import json
from pathlib import Path

import numpy as np
import pytest

from anydexretarget.deployment import (
    build_grasp_execution_plan,
    rigid_transform,
)


def _plan() -> dict[str, np.ndarray]:
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return {
        "qpos": np.linspace(0.0, 1.0, 21, dtype=np.float32),
        "robot_joint_names": np.asarray([f"joint_{index}" for index in range(21)]),
        "camera_to_l25_rotation": rotation,
        "camera_to_l25_translation": np.asarray((0.1, -0.2, 0.3)),
        "retarget_backend": np.asarray("vector"),
        "active_contact_mask": np.asarray((1, 1, 0, 0, 0), dtype=np.uint8),
    }


def test_execution_plan_exports_inverse_wrist_pose() -> None:
    result = build_grasp_execution_plan(
        _plan(),
        source_plan=Path("final_plan.npz"),
        anchor_frame="waist_camera_color_optical_frame",
        candidate_id="candidate_017",
    )

    np.testing.assert_allclose(
        result["T_l25_hand_anchor"] @ result["T_anchor_l25_hand"],
        np.eye(4),
        atol=1e-7,
    )
    assert result["l25_qpos"].shape == (21,)
    assert not bool(result["hardware_ready"].item())
    assert not bool(result["pregrasp_defined"].item())


def test_pregrasp_offset_is_applied_in_hand_coordinates() -> None:
    result = build_grasp_execution_plan(
        _plan(),
        source_plan=Path("final_plan.npz"),
        anchor_frame="waist_camera_color_optical_frame",
        pregrasp_offset_hand_m=np.asarray((0.0, 0.0, -0.1)),
    )
    relative = np.linalg.inv(result["T_anchor_l25_hand"]) @ result[
        "T_anchor_pregrasp_l25_hand"
    ]
    np.testing.assert_allclose(relative[:3, 3], (0.0, 0.0, -0.1), atol=1e-8)


def test_non_rigid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        rigid_transform(np.eye(3) * 2.0, np.zeros(3))


def test_execution_report_uses_strict_json_null_for_undefined_pregrasp() -> None:
    from tools.export_grasp_execution_plan import _json_value

    result = build_grasp_execution_plan(
        _plan(),
        source_plan=Path("final_plan.npz"),
        anchor_frame="waist_camera_color_optical_frame",
    )
    report = {key: _json_value(value) for key, value in result.items()}
    encoded = json.dumps(report, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["pregrasp_offset_hand_m"] == [None, None, None]
    assert all(value is None for row in decoded["T_anchor_pregrasp_l25_hand"] for value in row)


def test_execution_plan_preserves_portable_asset_references() -> None:
    result = build_grasp_execution_plan(
        _plan(),
        source_plan=Path("final_plan.npz"),
        anchor_frame="waist_camera_color_optical_frame",
        object_mesh=Path("reconstruction/object_mesh_anchor.ply"),
        reconstruction_result=Path("reconstruction/reconstruction_metadata.json"),
    )

    assert result["source_object_mesh"].item() == "reconstruction/object_mesh_anchor.ply"
    assert (
        result["source_reconstruction_result"].item()
        == "reconstruction/reconstruction_metadata.json"
    )
