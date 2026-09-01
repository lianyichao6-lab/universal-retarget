#!/usr/bin/env python3
"""Refine a HUG canonical grasp toward an RGB-D object's visible surface.

This is an offline surface-proximity refinement, not full collision checking:
an RGB-D point cloud has no reliable hidden-side geometry or surface normals.
The tool makes small MANO pose changes and keeps the original state untouched.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from adapt_canonical_grasp import (
    _evaluate,
    _rebuild_state,
    _score,
    _trajectory_records,
)
from anydexretarget.hand_representation import FINGERTIP_INDICES, load_canonical_grasp_state
from hug.models.mano import MANO
from scipy.spatial.transform import Rotation
import torch


def nearest_tip_distances(state, object_points: np.ndarray) -> np.ndarray:
    tips = np.asarray(state.fingertip_positions_camera, dtype=np.float64)
    points = np.asarray(object_points, dtype=np.float64)
    distances = np.linalg.norm(tips[:, None, :] - points[None, :, :], axis=2)
    return distances.min(axis=1)


def surface_loss(distances: np.ndarray, target_gap: float) -> float:
    # Use a symmetric soft band. With no normals, this is deliberately only a
    # surface-proximity term and cannot claim to detect object penetration.
    return float(np.mean(np.square(distances - target_gap)) / max(target_gap**2, 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--optimizer", choices=("vector", "adaptive"), default="vector")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--pose-step-rad", type=float, default=0.03)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.10)
    parser.add_argument("--target-gap-m", type=float, default=0.006)
    parser.add_argument("--contact-weight", type=float, default=0.12)
    parser.add_argument("--pinch-weight", type=float, default=0.15)
    args = parser.parse_args()
    if args.rounds <= 0 or args.frames <= 0 or args.fps <= 0:
        parser.error("rounds, frames, and fps must be positive")
    if args.pose_step_rad <= 0 or args.max_pose_delta_rad < args.pose_step_rad:
        parser.error("max-pose-delta-rad must be >= pose-step-rad")
    if args.target_gap_m <= 0 or args.contact_weight < 0 or args.pinch_weight < 0:
        parser.error("target-gap-m must be positive and weights non-negative")

    base = load_canonical_grasp_state(args.input)
    if base.handedness != "right":
        raise ValueError("Only right-hand L25 refinement is currently supported")
    with np.load(args.pointcloud, allow_pickle=False) as data:
        points = np.asarray(data["points_camera"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"Invalid camera point cloud: {points.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano_model = MANO().to(device).eval()
    l25_model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml")
    )
    original_pose = base.mano_pose.reshape(15, 3).astype(np.float32)
    norms = np.linalg.norm(original_pose, axis=1)
    directions = np.divide(original_pose, norms[:, None], out=np.zeros_like(original_pose), where=norms[:, None] > 1e-6)
    active = np.flatnonzero(norms > 1e-6)

    baseline = _evaluate(base, args.optimizer, l25_model)
    base_cost = float(baseline["metrics"]["cost"])
    base_contact = surface_loss(nearest_tip_distances(base, points), args.target_gap_m)
    base_pinch = np.asarray(base.pinch_distances, dtype=np.float64)

    def score(result, delta):
        robot_score = _score(result, base_cost, delta, args.max_pose_delta_rad)
        contact = surface_loss(nearest_tip_distances(result["state"], points), args.target_gap_m)
        pinch = np.asarray(result["state"].pinch_distances, dtype=np.float64)
        pinch_error = float(np.mean(np.square(pinch - base_pinch)) / 1e-4)
        return (robot_score
                + args.contact_weight * contact / max(base_contact, 1e-8)
                + args.pinch_weight * pinch_error)

    best = baseline
    best_delta = np.zeros(15, dtype=np.float32)
    best_score = score(best, best_delta)
    evaluations = 1
    for _ in range(args.rounds):
        improved = False
        for index in active:
            local_best, local_score, local_delta = best, best_score, best_delta
            for sign in (-1.0, 1.0):
                proposal = best_delta.copy()
                proposal[index] = np.clip(proposal[index] + sign * args.pose_step_rad, -args.max_pose_delta_rad, args.max_pose_delta_rad)
                if proposal[index] == best_delta[index]:
                    continue
                pose = original_pose + proposal[:, None] * directions
                candidate_state = _rebuild_state(base, pose, mano_model, device)
                candidate = _evaluate(candidate_state, args.optimizer, l25_model)
                evaluations += 1
                candidate_score = score(candidate, proposal)
                if candidate_score + 1e-9 < local_score:
                    local_best, local_score, local_delta = candidate, candidate_score, proposal
            if local_score + 1e-9 < best_score:
                best, best_score, best_delta = local_best, local_score, local_delta
                improved = True
        if not improved:
            break

    adapted = best["state"]
    adapted.to_npz(args.output)
    trajectory = None
    final_metrics = best["metrics"]
    if args.trajectory_output is not None:
        records, final_metrics = _trajectory_records(adapted, args.optimizer, args.frames, args.fps)
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trajectory_output.open("wb") as stream:
            pickle.dump(records, stream)
        trajectory = str(args.trajectory_output.resolve())

    before_dist = nearest_tip_distances(base, points)
    after_dist = nearest_tip_distances(adapted, points)
    metadata = {
        "source_canonical_grasp": str(args.input.resolve()),
        "object_pointcloud": str(args.pointcloud.resolve()),
        "output_canonical_grasp": str(args.output.resolve()),
        "optimizer": args.optimizer,
        "method": "mano_pose_surface_proximity_coordinate_descent",
        "contact_collision_checked": False,
        "hidden_surface_geometry_used": False,
        "target_gap_m": args.target_gap_m,
        "contact_weight": args.contact_weight,
        "pinch_weight": args.pinch_weight,
        "evaluations": evaluations,
        "pose_delta_rad": best_delta.tolist(),
        "pose_delta_rms_rad": float(np.sqrt(np.mean(np.square(best_delta)))),
        "surface_distance_before_m": before_dist.tolist(),
        "surface_distance_after_m": after_dist.tolist(),
        "surface_loss_before": base_contact,
        "surface_loss_after": surface_loss(after_dist, args.target_gap_m),
        "pinch_distances_before_m": base_pinch.tolist(),
        "pinch_distances_after_m": np.asarray(adapted.pinch_distances).tolist(),
        "solver_cost_before": float(baseline["metrics"]["cost"]),
        "solver_cost_after": float(final_metrics["cost"]),
        "saturation_before": int(baseline["saturation_count"]),
        "saturation_after": int(best["saturation_count"]),
        "trajectory": trajectory,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("L25 contact-aware surface refinement completed (offline)")
    print(f"  surface loss: {base_contact:.4f} -> {metadata['surface_loss_after']:.4f}")
    print(f"  solver cost: {metadata['solver_cost_before']:.6f} -> {metadata['solver_cost_after']:.6f}")
    print(f"  saturation: {metadata['saturation_before']} -> {metadata['saturation_after']}")
    print(f"  output: {args.output}")
    if trajectory:
        print(f"  trajectory: {trajectory}")


if __name__ == "__main__":
    main()
