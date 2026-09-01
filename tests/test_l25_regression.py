from pathlib import Path
import unittest

import numpy as np

from anydexretarget.dex_backend import (
    L25_REFERENCE,
    L25_VISUAL_URDF,
    _assert_l25_kinematic_equivalence,
)
from anydexretarget.hand_representation import load_canonical_grasp_state


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "outputs/grasp/red_cup/fresh2_hug/candidate_006/canonical_grasp.npz"


class L25RegressionTests(unittest.TestCase):
    def test_dex_and_visual_urdf_are_kinematically_equivalent(self):
        _assert_l25_kinematic_equivalence(
            L25_REFERENCE / "linkerhand_l25_right.urdf", L25_VISUAL_URDF
        )

    def test_canonical_round_trip_is_finite_and_preserves_camera_points(self):
        self.assertTrue(CANONICAL.is_file(), CANONICAL)
