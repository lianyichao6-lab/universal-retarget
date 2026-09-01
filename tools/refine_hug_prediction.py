#!/usr/bin/env python3
"""Convert one HUG prediction into the shared canonical grasp state.

This stage is robot-independent.  It preserves HUG's MANO pose/mesh and adds
canonical wrist-frame geometry, so the same ``.npz`` can feed several robot
morphology adapters later.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from anydexretarget.hand_representation import canonical_grasp_from_hug


TARGET_SIZE = 224


def _camera_matrix(payload: Mapping[str, Any]) -> np.ndarray:
    camera = payload.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("HUG prediction does not contain camera intrinsics")
    K = np.asarray(camera.get("K"), dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"camera.K must have shape (3,3), got {K.shape}")
    return K


def _condition_uv(payload: Mapping[str, Any]) -> np.ndarray | None:
    value = payload.get("condition_point")
    if value is not None:
        uv = np.asarray(value, dtype=np.float32).reshape(-1)
        if uv.size == 2:
            return uv
    # HUG's interactive app stores the normalized click in object_mask as two
    # float32 values, while benchmark samples store a real PNG mask.
    raw = payload.get("object_mask", b"")
    if isinstance(raw, (bytes, bytearray)) and len(raw) == 8:
        normalized = np.frombuffer(raw, dtype=np.float32)
        if normalized.shape == (2,) and np.isfinite(normalized).all() and (normalized >= 0).all() and (normalized <= 1).all():
            return normalized * TARGET_SIZE
    return None


def _object_point_from_prediction(payload: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    uv = _condition_uv(payload)
    if uv is None or not isinstance(payload.get("depth"), (bytes, bytearray)):
        return None, uv
    depth = cv2.imdecode(
        np.frombuffer(payload["depth"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if depth is None or depth.ndim != 2:
        return None, uv
    u = int(np.clip(round(float(uv[0])), 0, depth.shape[1] - 1))
    v = int(np.clip(round(float(uv[1])), 0, depth.shape[0] - 1))
    valid = depth[(depth > 0) & (depth < 65535)]
    value = int(depth[v, u])
    if not (0 < value < 65535):
        if valid.size == 0:
            return None, uv
        value = int(np.median(valid))
    K = _camera_matrix(payload)
    depth_m = value / 1000.0
    point = np.array(
        [(uv[0] - K[0, 2]) * depth_m / K[0, 0],
         (uv[1] - K[1, 2]) * depth_m / K[1, 1],
         depth_m],
        dtype=np.float32,
    )
    return point, uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--object-point-camera",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional object point in camera meters; overrides the clicked depth point.",
    )
    args = parser.parse_args()
    with args.prediction.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"HUG prediction must be a mapping, got {type(payload)!r}")

    inferred_object, condition_uv = _object_point_from_prediction(payload)
    object_point = (
        np.asarray(args.object_point_camera, dtype=np.float32)
        if args.object_point_camera is not None
        else inferred_object
    )
    state = canonical_grasp_from_hug(
        payload,
        handedness=args.hand,
        object_point_camera=object_point,
        condition_point_224=condition_uv,
    )
    state.to_npz(args.output)
    metadata = {
        "schema_version": 1,
        "source_prediction": str(args.prediction.resolve()),
        "output": str(args.output.resolve()),
        "source": state.source,
        "handedness": state.handedness,
        "refinement": "MANO-preserving canonicalization; no contact optimization",
        "keypoints_camera_shape": list(state.keypoints_camera.shape),
        "keypoints_canonical_shape": list(state.keypoints_canonical.shape),
        "mano_mesh_shape": list(state.mano_mesh_vertices_camera.shape),
        "object_point_from_click_depth": bool(inferred_object is not None and args.object_point_camera is None),
        "object_point_overridden": bool(args.object_point_camera is not None),
        "finite": bool(
            np.isfinite(state.keypoints_canonical).all()
            and np.isfinite(state.mano_mesh_vertices_camera).all()
        ),
        "bone_length_min_m": float(state.bone_lengths.min()),
        "bone_length_max_m": float(state.bone_lengths.max()),
        "pinch_distances_m": state.pinch_distances.tolist(),
        "fingertip_to_object_distance_m": state.fingertip_to_object_distance.tolist(),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Canonical HUG grasp state written")
    print(f"  canonical keypoints: {state.keypoints_canonical.shape}, finite={np.isfinite(state.keypoints_canonical).all()}")
    print(f"  MANO mesh: {state.mano_mesh_vertices_camera.shape}")
    print(f"  object point available: {np.isfinite(state.object_point_camera).all()}")
    print(f"  output: {args.output}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
