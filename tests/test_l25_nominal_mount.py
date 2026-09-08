from __future__ import annotations

import numpy as np
import pytest

from tools.create_l25_nominal_mount import rpy_transform


def test_l25_nominal_mount_matches_urdf_default() -> None:
    transform = rpy_transform((0.0, 0.0, -1.5708), (0.0, 0.0, 0.0))
    assert np.allclose(transform[:3, 3], 0.0)
    assert transform[3, 3] == 1.0
    assert transform[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert transform[1, 0] == pytest.approx(-1.0, abs=1e-5)


def test_l25_nominal_mount_rejects_bad_translation() -> None:
    with pytest.raises(ValueError, match="xyz"):
        rpy_transform((0.0, 0.0, 0.0), (0.0, 0.0))
