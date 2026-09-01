from pathlib import Path
import unittest

import numpy as np

from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.l25_target_chain import (
    FINGER_CHAINS,
    build_l25_target_chain,
    chain_error_metrics,
    l25_chain_points,
)
from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "outputs/grasp/red_cup/fresh2_hug/candidate_006/canonical_grasp.npz"
CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"


class L25TargetChainTests(unittest.TestCase):
    def test_continuous_target_preserves_l25_fk_segment_lengths(self):
        state = load_canonical_grasp_state(CANONICAL)
        retargeter = Retargeter.from_yaml(str(CONFIG), hand_side="right")
        qpos, verbose = retargeter.retarget_verbose(
            state.keypoints_for_retargeting(), apply_filter=False
        )
        target = build_l25_target_chain(retargeter.optimizer, verbose["mediapipe_kp"])
        self.assertEqual(target.target_points.shape, (21, 3))
        self.assertTrue(np.isfinite(target.target_points).all())
        measured = np.asarray(
            [
                [
                    np.linalg.norm(
                        target.target_points[chain[i + 1]] - target.target_points[chain[i]]
                    )
                    for i in range(4)
                ]
                for chain in FINGER_CHAINS
            ]
        )
        np.testing.assert_allclose(measured, target.segment_lengths, atol=1e-10)
        metrics = chain_error_metrics(
            target, l25_chain_points(retargeter.optimizer, qpos)
        )
        self.assertTrue(np.isfinite(list(metrics.values())).all())


if __name__ == "__main__":
    unittest.main()
