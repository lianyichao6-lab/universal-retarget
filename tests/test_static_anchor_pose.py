from __future__ import annotations

import numpy as np

from tools.create_static_anchor_pose import pose_transform


def test_pose_transform_uses_urdf_rpy_order() -> None:
    transform = pose_transform((1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2.0))
    np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(transform[:3, :3] @ (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), atol=1e-7)


def test_pose_transform_rejects_invalid_shape() -> None:
    try:
        pose_transform((0.0, 0.0, 0.0))
    except ValueError as exc:
        assert "six" in str(exc)
    else:
        raise AssertionError("expected invalid pose to fail")
