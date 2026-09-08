"""Vector retargeting baseline for the 20-axis LinkerHand O30."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_o30.yaml"
BACKENDS = ("vector",)


@dataclass(frozen=True)
class O30RetargetBaseline:
    """One O30 Vector result in the URDF/HOP 20-axis order."""

    qpos: np.ndarray
    joint_names: list[str]
    transformed_keypoints: np.ndarray
    geometry_retargeter: Retargeter


def retarget_o30_vector(
    keypoints: np.ndarray,
    *,
    hand_side: str = "right",
    config: Path = VECTOR_CONFIG,
) -> O30RetargetBaseline:
    """Retarget one finite standard 21-point hand to an O30 qpos."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
        raise ValueError("Expected finite MANUS/HUG keypoints with shape 21 x 3")
    retargeter = Retargeter.from_yaml(str(config), hand_side=hand_side)
    qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
    qpos = np.asarray(qpos, dtype=np.float64)
    names = [str(name) for name in retargeter.optimizer.robot.dof_joint_names]
    if qpos.shape != (20,) or len(names) != 20 or not np.isfinite(qpos).all():
        raise ValueError("O30 Vector retargeter must produce 20 finite joint positions")
    return O30RetargetBaseline(
        qpos=qpos,
        joint_names=names,
        transformed_keypoints=np.asarray(verbose["mediapipe_kp"], dtype=np.float64),
        geometry_retargeter=retargeter,
    )


__all__ = ["BACKENDS", "O30RetargetBaseline", "VECTOR_CONFIG", "retarget_o30_vector"]
