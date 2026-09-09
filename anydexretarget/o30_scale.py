"""Versioned geometry scale profiles for the physical 20-axis O30 hand.

The Vector YAML contains one scale per human-to-robot key vector.  Keeping
those values in a profile makes the geometry used by HUG retargeting,
collision checking and hardware export explicit and reproducible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_o30.yaml"
DEFAULT_URDF = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid O30 Vector YAML: {path}")
    vectors = config.get("retarget", {}).get("key_vectors")
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("O30 Vector config lacks retarget.key_vectors")
    return config


@dataclass(frozen=True)
class O30ScaleProfile:
    """Scale data bound to one O30 Vector YAML and URDF revision."""

    key_vector_scales: np.ndarray
    config_sha256: str
    urdf_sha256: str
    method: str
    source_samples: int
    version: int = 1

    def validate(self, *, config: Path = DEFAULT_CONFIG, urdf: Path = DEFAULT_URDF) -> None:
        vectors = _load_config(config)["retarget"]["key_vectors"]
        scales = np.asarray(self.key_vector_scales, dtype=np.float64)
        if scales.shape != (len(vectors),) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("O30 scale profile has invalid key_vector_scales")
        if self.config_sha256 != file_sha256(config):
            raise ValueError("O30 scale profile was created for a different Vector config")
        if self.urdf_sha256 != file_sha256(urdf):
            raise ValueError("O30 scale profile was created for a different O30 URDF")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "anydexretarget.o30_scale_profile/v1",
            "version": self.version,
            "hand_model": "o30",
            "side": "right",
            "key_vector_scales": np.asarray(self.key_vector_scales, dtype=np.float64).tolist(),
            "config_sha256": self.config_sha256,
            "urdf_sha256": self.urdf_sha256,
            "method": self.method,
            "source_samples": self.source_samples,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, config: Path = DEFAULT_CONFIG, urdf: Path = DEFAULT_URDF) -> "O30ScaleProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "anydexretarget.o30_scale_profile/v1":
            raise ValueError(f"Not an O30 scale profile: {path}")
        profile = cls(
            key_vector_scales=np.asarray(payload["key_vector_scales"], dtype=np.float64),
            config_sha256=str(payload["config_sha256"]),
            urdf_sha256=str(payload["urdf_sha256"]),
            method=str(payload.get("method", "unknown")),
            source_samples=int(payload.get("source_samples", 0)),
            version=int(payload.get("version", 1)),
        )
        profile.validate(config=config, urdf=urdf)
        return profile


def nominal_profile(*, config: Path = DEFAULT_CONFIG, urdf: Path = DEFAULT_URDF) -> O30ScaleProfile:
    """Return the audited YAML baseline as an explicit, hash-bound profile."""
    vectors = _load_config(config)["retarget"]["key_vectors"]
    return O30ScaleProfile(
        key_vector_scales=np.asarray([item.get("scale", 1.0) for item in vectors], dtype=np.float64),
        config_sha256=file_sha256(config),
        urdf_sha256=file_sha256(urdf),
        method="audited_yaml_baseline",
        source_samples=0,
    )


def calibrate_profile(
    keypoints: np.ndarray,
    *,
    config: Path = DEFAULT_CONFIG,
    urdf: Path = DEFAULT_URDF,
) -> O30ScaleProfile:
    """Fit each Vector scale from representative 21-point hand poses.

    It matches the median human vector length to the corresponding O30 neutral
    FK vector.  This deliberately calibrates robot morphology only; object
    geometry is never rescaled.
    """
    samples = np.asarray(keypoints, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples[None]
    if samples.ndim != 3 or samples.shape[1:] != (21, 3) or len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("O30 scale calibration expects finite N x 21 x 3 keypoints")
    cfg = _load_config(config)
    retargeter = Retargeter.from_config(copy.deepcopy(cfg), hand_side="right")
    optimizer = retargeter.optimizer
    transformed = []
    for sample in samples:
        points = sample.copy()
        from .mediapipe import apply_mediapipe_transformations

        points = apply_mediapipe_transformations(points, "right")
        if retargeter.rotation_xyz:
            points = retargeter._apply_rotation(points)
        transformed.append(points)
    transformed_samples = np.asarray(transformed)
    human_vectors = transformed_samples[:, optimizer._task_kp_indices] - transformed_samples[:, optimizer._origin_kp_indices]
    human_lengths = np.median(np.linalg.norm(human_vectors, axis=2), axis=0)
    neutral = np.clip(np.zeros(optimizer.num_joints), optimizer.opt_lower_bounds, optimizer.opt_upper_bounds)
    robot_points = optimizer.robot.compute_points_batch(
        neutral, optimizer._kv_computed_link_indices, optimizer._kv_computed_link_offsets
    )
    robot_vectors = robot_points[optimizer._kv_task_indices] - robot_points[optimizer._kv_origin_indices]
    robot_lengths = np.linalg.norm(robot_vectors, axis=1)
    if np.any(human_lengths < 1e-5) or np.any(robot_lengths < 1e-6):
        raise ValueError("Scale calibration contains degenerate human or robot vectors")
    return O30ScaleProfile(
        key_vector_scales=robot_lengths / human_lengths,
        config_sha256=file_sha256(config),
        urdf_sha256=file_sha256(urdf),
        method="median_neutral_fk_to_canonical_vectors",
        source_samples=int(len(samples)),
    )


def retargeter_from_profile(
    profile: O30ScaleProfile | None,
    *,
    config: Path = DEFAULT_CONFIG,
    urdf: Path = DEFAULT_URDF,
) -> Retargeter:
    cfg = _load_config(config)
    active = nominal_profile(config=config, urdf=urdf) if profile is None else profile
    active.validate(config=config, urdf=urdf)
    for item, scale in zip(cfg["retarget"]["key_vectors"], active.key_vector_scales, strict=True):
        item["scale"] = float(scale)
    return Retargeter.from_config(cfg, hand_side="right")


__all__ = [
    "DEFAULT_CONFIG", "DEFAULT_URDF", "O30ScaleProfile", "calibrate_profile",
    "file_sha256", "nominal_profile", "retargeter_from_profile",
]
