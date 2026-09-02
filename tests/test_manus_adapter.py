import numpy as np
import pytest

from anydexretarget.manus_adapter import (
    MANUS_TO_21,
    canonical_hand_frame_from_manus,
    manus_keypoints_to_21,
)


def _manus_hand() -> tuple[np.ndarray, np.ndarray]:
    wrist = np.asarray((0.1, -0.2, 0.3), dtype=np.float64)
    points = np.zeros((25, 3), dtype=np.float64)
    for finger in range(5):
        lateral = (finger - 2) * 0.018
        for slot in range(5):
            points[finger * 5 + slot] = wrist + (
                0.02 + slot * 0.018,
                lateral,
                slot * 0.004,
            )
    points[2] = np.nan
    return wrist, points


def test_manus_semantic_layout_maps_to_standard_21_points() -> None:
    wrist, source = _manus_hand()
    mask = sum(1 << int(index) for index in MANUS_TO_21)
    points, valid = manus_keypoints_to_21(wrist, source, mask)

    assert points.shape == (21, 3)
    assert valid.all()
    np.testing.assert_allclose(points[0], wrist)
    np.testing.assert_allclose(points[1:], source[MANUS_TO_21])


def test_manus_frame_round_trips_through_canonical_coordinates() -> None:
    wrist, source = _manus_hand()
    mask = sum(1 << int(index) for index in MANUS_TO_21)
    frame = canonical_hand_frame_from_manus(
        wrist,
        source,
        mask,
        handedness="right",
        timestamp_s=12.5,
        ergonomics=np.linspace(0.0, 1.0, 20),
    )

    assert frame.source == "manus"
    assert frame.source_joint_angles.shape == (20,)
    np.testing.assert_allclose(
        frame.keypoints_for_retargeting(), frame.keypoints_21, atol=1e-6
    )


def test_missing_required_manus_landmark_is_rejected() -> None:
    wrist, source = _manus_hand()
    mask = sum(1 << int(index) for index in MANUS_TO_21)
    mask &= ~(1 << 9)

    with pytest.raises(ValueError, match="missing required"):
        canonical_hand_frame_from_manus(wrist, source, mask, handedness="right")
