#!/usr/bin/env python3
"""Conservatively adapt one canonical HUG grasp for LinkerHand L25.

This offline tool keeps MANO geometry internally consistent. It does not infer
hidden object geometry or claim force closure. It only makes small pose changes
when they improve L25 retargeting feasibility while preserving the HUG grasp.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from anydexretarget.hand_representation import (
    BONE_EDGES,
    FINGERTIP_INDICES,
    PINCH_PAIRS,
    CanonicalGraspState,
    canonicalize_keypoints,
    load_canonical_grasp_state,
)
from hug.models.mano import MANO
from grasp_object import L25_JOINT_NAMES, L25_MODEL, _retarget_l25

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--robot", choices=("l25",), default="l25")
    parser.add_argument("--optimizer", choices=("vector", "adaptive"), default="vector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--pose-step-rad", type=float, default=0.04)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.12)
    args = parser.parse_args()
    if args.rounds <= 0 or args.frames <= 0 or args.fps <= 0:
        parser.error("--rounds, --frames, and --fps must be positive")
    if args.pose_step_rad <= 0 or args.max_pose_delta_rad <= 0:
        parser.error("--pose-step-rad and --max-pose-delta-rad must be positive")
    if args.pose_step_rad > args.max_pose_delta_rad:
        parser.error("--pose-step-rad must not exceed --max-pose-delta-rad")
    return args


def _pose_to_6d(pose_axis_angle: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(pose_axis_angle.reshape(-1, 3)).as_matrix()
    return rotations[:, :, :2].reshape(1, 15, 6).astype(np.float32)


def _rebuild_state(
    base: CanonicalGraspState,
    pose_axis_angle: np.ndarray,
    mano_model: MANO,
    device: torch.device,
) -> CanonicalGraspState:
    pose = np.asarray(pose_axis_angle, dtype=np.float32).reshape(15, 3)
    global_rotation = Rotation.from_matrix(
        base.mano_t_camera_wrist[:3, :3]
    ).as_rotvec().astype(np.float32)
    pose_coeffs = np.concatenate((global_rotation, pose.reshape(-1)))[None]
    shape = base.mano_shape.astype(np.float32)
    translation = base.mano_t_camera_wrist[:3, 3].astype(np.float32)

    with torch.no_grad():
        out = mano_model.mano_layer(
            torch.from_numpy(pose_coeffs).to(device),
            torch.from_numpy(shape).to(device),
        )
    keypoints_camera = out.joints[0].detach().cpu().numpy().astype(np.float32)
    keypoints_camera += translation[None]
    mesh_camera = out.verts[0].detach().cpu().numpy().astype(np.float32)
    mesh_camera += translation[None]

    canonical, basis = canonicalize_keypoints(keypoints_camera, base.handedness)
    wrist = keypoints_camera[0].copy()
    tips_camera = keypoints_camera[FINGERTIP_INDICES].copy()
    tips_canonical = canonical[FINGERTIP_INDICES].copy()
    bone_lengths = np.linalg.norm(
        canonical[BONE_EDGES[:, 1]] - canonical[BONE_EDGES[:, 0]], axis=1
    ).astype(np.float32)
    pinch_distances = np.linalg.norm(
        canonical[PINCH_PAIRS[:, 0]] - canonical[PINCH_PAIRS[:, 1]], axis=1
    ).astype(np.float32)

    object_canonical = np.full(3, np.nan, dtype=np.float32)
    tip_object_distance = np.full(len(FINGERTIP_INDICES), np.nan, dtype=np.float32)
    if np.isfinite(base.object_point_camera).all():
        object_canonical = (
            (base.object_point_camera - wrist) @ basis
        ).astype(np.float32)
        tip_object_distance = np.linalg.norm(
            tips_camera - base.object_point_camera[None], axis=1
        ).astype(np.float32)

    return replace(
        base,
        source=f"{base.source}+l25_morphology_adapted",
        keypoints_camera=keypoints_camera,
        keypoints_canonical=canonical,
        wrist_position_camera=wrist,
        canonical_basis_row=basis,
        bone_lengths=bone_lengths,
        fingertip_positions_camera=tips_camera,
        fingertip_positions_canonical=tips_canonical,
        pinch_distances=pinch_distances,
        mano_pose=pose[None],
        mano_pose_6d=_pose_to_6d(pose),
        mano_mesh_vertices_camera=mesh_camera,
        object_point_canonical=object_canonical,
        fingertip_to_object_distance=tip_object_distance,
    )


def _evaluate(
    state: CanonicalGraspState,
    optimizer: str,
    model: mujoco.MjModel,
) -> dict[str, object]:
    keypoints = state.keypoints_for_retargeting()
    records, metrics = _retarget_l25(keypoints, optimizer, 1, 30.0)
    qpos = np.asarray(metrics["qpos"], dtype=np.float64)
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    margin = np.minimum(
        (qpos - lower) / np.maximum(upper - lower, 1e-9),
        (upper - qpos) / np.maximum(upper - lower, 1e-9),
    )
    return {
        "state": state,
        "records": records,
        "metrics": metrics,
        "qpos": qpos,
        "saturation_count": int(np.count_nonzero(margin <= 0.05)),
        "min_normalized_margin": float(np.min(margin)),
    }


def _score(
    result: dict[str, object],
    base_cost: float,
    pose_delta: np.ndarray,
    max_delta: float,
) -> float:
    metrics = result["metrics"]
    saturation = float(result["saturation_count"]) / len(L25_JOINT_NAMES)
    violations = float(metrics["violations_before_clamp"]) / len(L25_JOINT_NAMES)
    pose_change = float(np.sqrt(np.mean(np.square(pose_delta)))) / max_delta
    return (
        0.65 * float(metrics["cost"]) / max(base_cost, 1e-8)
        + 0.15 * saturation
        + 0.15 * violations
        + 0.05 * pose_change
    )


def _trajectory_records(
    state: CanonicalGraspState,
    optimizer: str,
    frames: int,
    fps: float,
) -> tuple[list[dict], dict]:
    records, metrics = _retarget_l25(
        state.keypoints_for_retargeting(), optimizer, frames, fps
    )
    for record in records:
        record["human_representation"] = "robot_conditioned_canonical_grasp"
        record["human_keypoints_canonical"] = state.keypoints_canonical.copy()
        record["adaptation_source"] = "mano_pose_coordinate_descent"
    return records, metrics


def main() -> None:
    args = _parse_args()
    base = load_canonical_grasp_state(args.input)
    if base.handedness != "right":
        raise ValueError("Only right-hand L25 adaptation is currently verified")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano_model = MANO().to(device).eval()
    l25_model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    original_pose = base.mano_pose.reshape(15, 3).astype(np.float32)
    pose_norm = np.linalg.norm(original_pose, axis=1)
    directions = np.divide(
        original_pose,
        pose_norm[:, None],
        out=np.zeros_like(original_pose),
        where=pose_norm[:, None] > 1e-6,
    )
    active = np.flatnonzero(pose_norm > 1e-6)

    baseline = _evaluate(base, args.optimizer, l25_model)
    base_cost = float(baseline["metrics"]["cost"])
    best_delta = np.zeros(15, dtype=np.float32)
    best_score = _score(
        baseline, base_cost, best_delta, args.max_pose_delta_rad
    )
    best = baseline
    evaluations = 1

    for _round in range(args.rounds):
        improved = False
        for index in active:
            local_best = best
            local_score = best_score
            local_delta = best_delta
            for sign in (-1.0, 1.0):
                proposal = best_delta.copy()
                proposal[index] = np.clip(
                    proposal[index] + sign * args.pose_step_rad,
                    -args.max_pose_delta_rad,
                    args.max_pose_delta_rad,
                )
                if proposal[index] == best_delta[index]:
                    continue
                candidate_pose = original_pose + proposal[:, None] * directions
                candidate_state = _rebuild_state(
                    base, candidate_pose, mano_model, device
                )
                candidate = _evaluate(candidate_state, args.optimizer, l25_model)
                evaluations += 1
                candidate_score = _score(
                    candidate, base_cost, proposal, args.max_pose_delta_rad
                )
                if candidate_score + 1e-9 < local_score:
                    local_best = candidate
                    local_score = candidate_score
                    local_delta = proposal
            if local_score + 1e-9 < best_score:
                best, best_score, best_delta = local_best, local_score, local_delta
                improved = True
        if not improved:
            break

    adapted = best["state"]
    adapted.to_npz(args.output)
    trajectory_path = args.trajectory_output
    final_metrics = best["metrics"]
    if trajectory_path is not None:
        records, final_metrics = _trajectory_records(
            adapted, args.optimizer, args.frames, args.fps
        )
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("wb") as stream:
            pickle.dump(records, stream)

    metadata = {
        "source_canonical_grasp": str(args.input.resolve()),
        "output_canonical_grasp": str(args.output.resolve()),
        "robot": args.robot,
        "optimizer": args.optimizer,
        "method": "mano_pose_coordinate_descent",
        "contact_optimization": False,
        "hidden_surface_geometry_used": False,
        "evaluations": evaluations,
        "active_mano_joints": active.tolist(),
        "pose_delta_rad": best_delta.tolist(),
        "pose_delta_rms_rad": float(np.sqrt(np.mean(np.square(best_delta)))),
        "score_before": _score(
            baseline, base_cost, np.zeros(15, dtype=np.float32), args.max_pose_delta_rad
        ),
        "score_after": best_score,
        "solver_cost_before": float(baseline["metrics"]["cost"]),
        "solver_cost_after": float(final_metrics["cost"]),
        "saturation_before": int(baseline["saturation_count"]),
        "saturation_after": int(best["saturation_count"]),
        "limit_violations_before": int(
            baseline["metrics"]["violations_before_clamp"]
        ),
        "limit_violations_after": int(
            final_metrics["violations_before_clamp"]
        ),
        "trajectory": (
            None if trajectory_path is None else str(trajectory_path.resolve())
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("L25 morphology-aware canonical adaptation completed (offline)")
    print(f"  input: {args.input}")
    print(f"  adapted state: {args.output}")
    print(
        f"  score: {metadata['score_before']:.6f} -> {metadata['score_after']:.6f}; "
        f"solver cost: {metadata['solver_cost_before']:.6f} -> "
        f"{metadata['solver_cost_after']:.6f}"
    )
    print(
        f"  saturation: {metadata['saturation_before']} -> "
        f"{metadata['saturation_after']}; limit violations: "
        f"{metadata['limit_violations_before']} -> "
        f"{metadata['limit_violations_after']}"
    )
    print(f"  pose delta RMS: {metadata['pose_delta_rms_rad']:.6f} rad")
    if trajectory_path is not None:
        print(f"  trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
