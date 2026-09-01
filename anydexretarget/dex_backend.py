"""Optional backend for the local dex-retargeting implementation."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .mediapipe import apply_mediapipe_transformations

ROOT = Path(__file__).resolve().parents[1]
DEX_SOURCE = ROOT / "external" / "dex-retargeting" / "src"
if str(DEX_SOURCE) not in sys.path:
    sys.path.insert(0, str(DEX_SOURCE))
L25_REFERENCE = ROOT / "external" / "dex-retargeting" / "l25_reference"
L25_VISUAL_URDF = ROOT / "assets" / "linkerhand_l25" / "right" / "linkerhand_l25_right.urdf"
DEX_CONFIGS = {
    "dexpilot": L25_REFERENCE / "linkerhand_l25_right_dexpilot.yml",
    "joint_angle": L25_REFERENCE / "linkerhand_l25_right_joint_angle.yml",
}


def _assert_l25_kinematic_equivalence(reference_path: Path, visual_path: Path) -> None:
    """Reject DexPilot if its active-joint URDF drifts from the visual model."""
    def active(path: Path) -> dict[str, tuple]:
        root = ET.parse(path).getroot()
        result = {}
        for joint in root.findall("joint"):
            if joint.attrib.get("type") == "fixed":
                continue
            fields = []
            for tag in ("origin", "axis", "limit", "mimic"):
                element = joint.find(tag)
                fields.append(None if element is None else tuple(sorted(element.attrib.items())))
            result[joint.attrib["name"]] = (joint.attrib["type"], *fields)
        return result

    reference, visual = active(reference_path), active(visual_path)
    if reference != visual:
        missing = sorted(set(reference) - set(visual))
        extra = sorted(set(visual) - set(reference))
        raise ValueError(f"L25 Dex/visual URDF mismatch; reference-only={missing}, visual-only={extra}")


class DexRetargetBackend:
    """Run a migrated dex-retargeting optimizer for the L25 baseline."""

    def __init__(self, optimizer: str, hand_side: str = "right", scaling_factor: float | None = None, project_dist: float | None = None, escape_dist: float | None = None) -> None:
        if hand_side != "right":
            raise ValueError("The migrated L25 backend currently supports right hand only")
        if optimizer not in DEX_CONFIGS:
            raise ValueError(f"Unsupported dex-retargeting optimizer: {optimizer}")
        for name, value in (("scaling_factor", scaling_factor), ("project_dist", project_dist), ("escape_dist", escape_dist)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        from dex_retargeting.retargeting_config import RetargetingConfig

        urdf_path = L25_REFERENCE / "linkerhand_l25_right.urdf"
        if not L25_VISUAL_URDF.is_file():
            raise FileNotFoundError(f"Canonical visual L25 URDF is missing: {L25_VISUAL_URDF}")
        _assert_l25_kinematic_equivalence(urdf_path, L25_VISUAL_URDF)
        RetargetingConfig.set_default_urdf_dir(urdf_path.parent)
        override = {"urdf_path": str(urdf_path)}
        if scaling_factor is not None:
            override["scaling_factor"] = float(scaling_factor)
        if project_dist is not None:
            override["project_dist"] = float(project_dist)
        if escape_dist is not None:
            override["escape_dist"] = float(escape_dist)
        config = RetargetingConfig.load_from_file(DEX_CONFIGS[optimizer], override=override)
        self.optimizer_name = optimizer
        self.scaling_factor = float(getattr(config, "scaling_factor", 1.0))
        self.retargeting = config.build()
        self.optimizer = self.retargeting.optimizer
        self.joint_names = [str(name) for name in self.retargeting.joint_names]

    def retarget(self, raw_keypoints: np.ndarray) -> tuple[np.ndarray, dict]:
        """Retarget camera-frame HUG/MediaPipe landmarks to L25 qpos."""
        points = np.asarray(raw_keypoints, dtype=np.float64)
        if points.shape != (21, 3):
            raise ValueError(f"Expected keypoints shape (21, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("Keypoints contain NaN or Inf")
        transformed = apply_mediapipe_transformations(points, "right").astype(np.float32)
        indices = np.asarray(self.optimizer.target_link_human_indices, dtype=int)
        reference = transformed[indices[1]] - transformed[indices[0]]
        qpos = np.asarray(self.retargeting.retarget(reference), dtype=np.float32)
        if qpos.shape != (len(self.joint_names),):
            raise ValueError(f"Backend returned qpos shape {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("Backend returned NaN or Inf qpos")
        return qpos, {"mediapipe_kp": transformed, "joint_names": self.joint_names}


__all__ = ["DEX_CONFIGS", "DexRetargetBackend"]
