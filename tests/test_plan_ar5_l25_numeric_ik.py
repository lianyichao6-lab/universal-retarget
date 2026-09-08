import numpy as np

from tools.plan_ar5_l25_numeric_ik import _joint_segment


def test_joint_segment_limits_each_increment() -> None:
    segment = _joint_segment(np.zeros(7), np.full(7, 0.1), 0.04)
    assert segment.shape == (4, 7)
    assert np.max(np.abs(np.diff(segment, axis=0))) <= 0.04 + 1e-12
