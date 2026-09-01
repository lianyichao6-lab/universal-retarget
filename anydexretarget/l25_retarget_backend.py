"""Common static L25 retargeting baseline for native and dex backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dex_backend import DEX_CONFIGS, DexRetargetBackend
from .retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"
ADAPTIVE_CONFIG = ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml"
NATIVE_BACKENDS = ("vector", "adaptive")
BACKENDS = (*NATIVE_BACKENDS, *DEX_CONFIGS.keys())


@dataclass(frozen=True)
class L25RetargetBaseline:
    """A backend result normalized to the audited AnyDex L25 joint order."""

    backend: str
    qpos: np.ndarray
    joint_names: list[str]
    transformed_keypoints: np.ndarray
    geometry_retargeter: Retargeter


def _map_by_name(values: np.ndarray, source_names: list[str], target_names: list[str]) -> np.ndarray:
    source = {name.lower(): index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name.lower() not in source]
    if missing:
        raise ValueError("L25 backend output is missing joints: " + ", ".join(missing))
    return np.asarray([values[source[name.lower()]] for name in target_names], dtype=np.float64)


def retarget_l25_static(
    backend: str,
    keypoints: np.ndarray,
    *,
    native_config: Path | None = None,
    geometry_config: Path = VECTOR_CONFIG,
    dex_scaling: float | None = None,
    dex_project_dist: float | None = None,
    dex_escape_dist: float | None = None,
) -> L25RetargetBaseline:
    """Return an L25 baseline with a common joint order and FK provider."""
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported L25 backend: {backend}")
    geometry_retargeter = Retargeter.from_yaml(str(geometry_config), hand_side="right")
    target_names = [str(name) for name in geometry_retargeter.optimizer.robot.dof_joint_names]
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
        raise ValueError("Expected finite HUG/Canonical keypoints with shape 21 x 3")

    if backend in NATIVE_BACKENDS:
        config = native_config or (VECTOR_CONFIG if backend == "vector" else ADAPTIVE_CONFIG)
        retargeter = Retargeter.from_yaml(str(config), hand_side="right")
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        source_names = [str(name) for name in retargeter.optimizer.robot.dof_joint_names]
        transformed = np.asarray(verbose["mediapipe_kp"], dtype=np.float64)
    else:
        dex = DexRetargetBackend(
            backend,
            hand_side="right",
            scaling_factor=dex_scaling,
            project_dist=dex_project_dist,
            escape_dist=dex_escape_dist,
        )
        qpos, verbose = dex.retarget(keypoints)
        source_names = [str(name) for name in dex.joint_names]
        transformed = np.asarray(verbose["mediapipe_kp"], dtype=np.float64)

    qpos = _map_by_name(np.asarray(qpos, dtype=np.float64), source_names, target_names)
    if not np.isfinite(qpos).all() or not np.isfinite(transformed).all():
        raise ValueError("L25 backend produced non-finite data")
    return L25RetargetBaseline(
        backend=backend,
        qpos=qpos,
        joint_names=target_names,
        transformed_keypoints=transformed,
        geometry_retargeter=geometry_retargeter,
    )


__all__ = [
    "ADAPTIVE_CONFIG",
    "BACKENDS",
    "L25RetargetBaseline",
    "VECTOR_CONFIG",
    "retarget_l25_static",
]
