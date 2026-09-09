"""Static O30 retargeting baselines normalized to the audited 20-axis order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dex_backend import DexRetargetBackend
from .retarget import Retargeter
from .o30_scale import O30ScaleProfile, retargeter_from_profile

ROOT = Path(__file__).resolve().parents[1]
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_o30.yaml"
ADAPTIVE_CONFIG = ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_o30.yaml"
NATIVE_BACKENDS = ("vector", "adaptive")
BACKENDS = (*NATIVE_BACKENDS, "dexpilot", "joint_angle")


@dataclass(frozen=True)
class O30RetargetBaseline:
    qpos: np.ndarray
    joint_names: list[str]
    transformed_keypoints: np.ndarray
    geometry_retargeter: Retargeter


def _map_by_name(values: np.ndarray, source_names: list[str], target_names: list[str]) -> np.ndarray:
    source = {name.lower(): index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name.lower() not in source]
    if missing:
        raise ValueError("O30 backend output is missing joints: " + ", ".join(missing))
    return np.asarray([values[source[name.lower()]] for name in target_names], dtype=np.float64)


def retarget_o30_static(
    keypoints: np.ndarray,
    *,
    backend: str = "vector",
    hand_side: str = "right",
    native_config: Path | None = None,
    geometry_config: Path = VECTOR_CONFIG,
    scale_profile: O30ScaleProfile | None = None,
    dex_scaling: float | None = None,
    dex_project_dist: float | None = None,
    dex_escape_dist: float | None = None,
) -> O30RetargetBaseline:
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
        raise ValueError("Expected finite MANUS/HUG keypoints with shape 21 x 3")
    if hand_side != "right":
        raise ValueError("O30 scale profiles currently support the right hand only")
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported O30 backend: {backend}")

    geometry = retargeter_from_profile(scale_profile, config=geometry_config)
    target_names = [str(name) for name in geometry.optimizer.robot.dof_joint_names]
    if backend in NATIVE_BACKENDS:
        config = native_config or (VECTOR_CONFIG if backend == "vector" else ADAPTIVE_CONFIG)
        retargeter = retargeter_from_profile(scale_profile, config=config) if backend == "vector" else Retargeter.from_yaml(str(config), hand_side=hand_side)
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        names = [str(name) for name in retargeter.optimizer.robot.dof_joint_names]
    else:
        dex = DexRetargetBackend(backend, hand_side=hand_side, scaling_factor=dex_scaling,
                                 project_dist=dex_project_dist, escape_dist=dex_escape_dist,
                                 robot_model="o30")
        qpos, verbose = dex.retarget(keypoints)
        names = [str(name) for name in dex.joint_names]
    qpos = _map_by_name(np.asarray(qpos, dtype=np.float64), names, target_names)
    if qpos.shape != (20,) or len(target_names) != 20 or not np.isfinite(qpos).all():
        raise ValueError("O30 backend must produce 20 finite joint positions")
    return O30RetargetBaseline(
        qpos=qpos,
        joint_names=target_names,
        transformed_keypoints=np.asarray(verbose["mediapipe_kp"], dtype=np.float64),
        geometry_retargeter=geometry,
    )


def retarget_o30_vector(keypoints: np.ndarray, **kwargs: object) -> O30RetargetBaseline:
    """Backward-compatible Vector shorthand."""
    return retarget_o30_static(keypoints, backend="vector", **kwargs)


__all__ = [
    "ADAPTIVE_CONFIG", "BACKENDS", "O30RetargetBaseline", "VECTOR_CONFIG",
    "retarget_o30_static", "retarget_o30_vector",
]
