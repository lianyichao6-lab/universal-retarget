from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from tools.build_l25_object_relative_scene import prepare_mujoco_mesh_proxy

from tools.plan_l25_rigid_object_relative_grasp import _apply, _rigid
from tools.refine_l25_collision_aware import _finger_from_geom
from tools.rerank_l25_object_relative_candidates import (
    _contact_cache_valid,
    _contact_metrics,
    _hardware_metrics,
    _score,
)


class ContactPlannerV2Tests(unittest.TestCase):
    def test_mesh_proxy_cache_reuses_matching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "object.ply"
            proxy = root / "proxy.ply"
            trimesh.creation.box(extents=(0.1, 0.2, 0.3)).export(source)
            _, cold = prepare_mujoco_mesh_proxy(source, proxy, max_faces=100)
            first_mtime = proxy.stat().st_mtime_ns
            _, warm = prepare_mujoco_mesh_proxy(source, proxy, max_faces=100)
            second_mtime = proxy.stat().st_mtime_ns
        self.assertFalse(cold["cache_hit"])
        self.assertTrue(warm["cache_hit"])
        self.assertEqual(first_mtime, second_mtime)

    def test_contact_plan_cache_checks_sources_and_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.npz"
            mesh = root / "mesh.ply"
            cache = root / "contact.npz"
            canonical.touch()
            mesh.touch()
            np.savez_compressed(
                cache,
                source_canonical_grasp=np.asarray(str(canonical.resolve())),
                source_object_mesh=np.asarray(str(mesh.resolve())),
                near_surface_gap_m=np.asarray(0.025),
            )
            self.assertTrue(
                _contact_cache_valid(cache, canonical, mesh, 25.0)
            )
            self.assertFalse(
                _contact_cache_valid(cache, canonical, mesh, 20.0)
            )

    def test_two_contacts_plus_wrist_define_rigid_fit(self) -> None:
        source = np.asarray(
            ((0.02, 0.01, 0.10), (-0.02, 0.01, 0.10), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        rotation_true = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        translation_true = np.asarray((0.1, -0.03, 0.02), dtype=np.float64)
        target = _apply(source, rotation_true, translation_true)
        rotation, translation = _rigid(source, target)
        np.testing.assert_allclose(
            _apply(source, rotation, translation), target, atol=1e-10
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

    def test_cross_finger_classifier(self) -> None:
        self.assertEqual(_finger_from_geom("index_distal_visual"), "index")
        self.assertEqual(_finger_from_geom("thumb_proximal_visual"), "thumb")
        self.assertIsNone(_finger_from_geom("hand_base_link_visual"))

    def test_contact_metrics_capture_opposition_and_span(self) -> None:
        active = np.asarray((1, 1, 0, 0, 0), dtype=np.uint8)
        normals = np.asarray(
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        anchors = np.zeros((5, 3), dtype=np.float32)
        anchors[1, 0] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.npz"
            np.savez_compressed(
                path,
                near_surface=active,
                surface_normal_camera=normals,
                surface_anchor_camera=anchors,
                finger_names=np.asarray(
                    ("thumb", "index", "middle", "ring", "pinky")
                ),
                contact_point_kind=np.asarray(
                    ("distal_pad", "tip", "tip", "tip", "tip")
                ),
            )
            metrics = _contact_metrics(path, object_diagonal=0.2)
        self.assertTrue(metrics["thumb_opposed"])
        self.assertEqual(metrics["thumb_opposed_fingers"], "index")
        self.assertAlmostEqual(metrics["opposed_anchor_span_ratio"], 0.5)

    @staticmethod
    def _base_score_row() -> dict[str, object]:
        return {
            "thumb_opposed": True,
            "thumb_best_normal_dot": -1.0,
            "opposed_anchor_span_ratio": 0.4,
            "contact_mean_error_mm": 4.0,
            "contact_max_error_mm": 7.0,
            "joint_saturation_count": 1,
            "joint_min_normalized_margin": 0.08,
            "mujoco_max_penetration_mm": 0.1,
            "mujoco_max_self_penetration_mm": 0.0,
            "posture_delta_rms": 0.05,
            "active_contact_fingers": 2,
        }

    def test_contact_count_is_not_a_score_term(self) -> None:
        row = self._base_score_row()
        score_two = _score(row, minimum_span=0.08)
        row["active_contact_fingers"] = 5
        self.assertEqual(score_two, _score(row, minimum_span=0.08))

    def test_self_collision_and_missing_opposition_are_penalized(self) -> None:
        row = self._base_score_row()
        baseline = _score(row, minimum_span=0.08)
        row["mujoco_max_self_penetration_mm"] = 1.0
        self.assertGreater(_score(row, minimum_span=0.08), baseline)
        row["mujoco_max_self_penetration_mm"] = 0.0
        row["thumb_opposed"] = False
        self.assertGreater(_score(row, minimum_span=0.08), baseline)

    def test_hardware_report_tracking_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "candidate_007.json"
            target = np.arange(25, dtype=np.int64)
            measured = target.copy()
            measured[3] += 5
            report_path.write_text(
                json.dumps({
                    "target_command_0_255": target.tolist(),
                    "state_after_0_255": measured.tolist(),
                }),
                encoding="utf-8",
            )
            metrics = _hardware_metrics(Path(directory), "candidate_007")
        self.assertTrue(metrics["hardware_reproduction_validated"])
        self.assertAlmostEqual(metrics["hardware_tracking_mean_error_0_255"], 0.2)
        self.assertEqual(metrics["hardware_tracking_max_error_0_255"], 5.0)


if __name__ == "__main__":
    unittest.main()
