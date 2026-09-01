"""Adapters for feeding official HUG predictions into AnyDexRetarget.

HUG stores predictions as a GraspData dictionary.  Its ``grasp.landmarks_3d``
field is a MANO-derived 21x3 array in the camera frame, in meters.  The
retargeting core already accepts the same 21-point semantic layout, so this
module deliberately performs validation and extraction only; coordinate-frame
normalization remains in :class:`anydexretarget.retarget.Retargeter`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class HUGHandFrame:
    """A validated HUG hand prediction suitable for ``Retargeter.retarget``."""

    timestamp: float
    keypoints_3d: np.ndarray
    handedness: str = "right"
    confidence: np.ndarray | None = None


def _get_grasp(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    grasp = payload.get("grasp", payload)
    if not isinstance(grasp, Mapping):
        raise ValueError("HUG prediction field 'grasp' must be a mapping")
    return grasp


def landmarks_from_prediction(payload: Mapping[str, Any]) -> np.ndarray:
    """Extract HUG ``landmarks_3d`` and validate its shape/values.

    The returned array is copied as float64 and remains in HUG's camera frame
    and meter units.  Retargeter performs wrist-frame conversion afterwards.
    """
    landmarks = _get_grasp(payload).get("landmarks_3d")
    if landmarks is None:
        raise KeyError("HUG prediction does not contain grasp.landmarks_3d")
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected HUG landmarks shape (21, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("HUG landmarks contain NaN or Inf")
    return points.copy()


def load_prediction(path: str | Path, *, timestamp: float = 0.0) -> HUGHandFrame:
    """Load one official HUG ``grasp_pred/*.pkl`` file."""
    path = Path(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"HUG prediction must be a mapping, got {type(payload)!r}")
    grasp = _get_grasp(payload)
    handedness = str(payload.get("handedness", "right"))
    return HUGHandFrame(
        timestamp=float(timestamp),
        keypoints_3d=landmarks_from_prediction(payload),
        handedness=handedness,
    )


__all__ = ["HUGHandFrame", "landmarks_from_prediction", "load_prediction"]
