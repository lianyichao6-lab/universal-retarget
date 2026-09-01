#!/usr/bin/env python3
"""Generate, score, and rank multiple HUG grasps for one RGB-D target point.

This command is offline-only. HUG weights are loaded once and reused across
all stochastic samples. Candidate selection preserves HUG's learned complete
grasp prior. A partial, single-view point cloud is used only to reject grasps
that are clearly detached from the visible object surface.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anydexretarget.hand_representation import (  # noqa: E402
    FINGERTIP_INDICES,
    canonical_grasp_from_hug,
)
from anydexretarget.hug_adapter import landmarks_from_prediction  # noqa: E402
from hug.dataloader.grasp_dataset import GraspDataset  # noqa: E402
from hug.inference import load_model  # noqa: E402
from hug.prepare_inputs import (  # noqa: E402
    _load_intrinsics,
    _read_depth_uint16,
    _read_rgb,
    prepare_pkl,
)
from tools.grasp_object import (  # noqa: E402
    L25_JOINT_NAMES,
    L25_MODEL,
    _map_original_point,
    _prediction_payload,
    _resolve_device,
    _retarget_l25,
    _run_hug,
    _save_target_preview,
    _write_pickle,
)


SCORE_VERSION = "hug_prior_l25_feasibility_v3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument(
        "--hug-pointcloud",
        type=Path,
        help=(
            "Experimental PointNeXt input. Must be a fused cloud expressed in the "
            "anchor RGB-D camera frame. Default: rebuild HUG PCL from --depth."
        ),
    )
    parser.add_argument(
        "--point", type=float, nargs=2, metavar=("U", "V"),
        help="Original RGB target pixel. Defaults to source object-mask metadata.",
    )
    parser.add_argument("--robot", choices=("l25",), default="l25")
    parser.add_argument("--optimizer", choices=("vector", "adaptive"), default="vector")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--contact-threshold-m", type=float, default=0.02)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "external/hug/checkpoints/hug_full.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("hardware execution is not implemented; use --dry-run")
    if args.candidates <= 0:
        parser.error("--candidates must be positive")
    if args.sampling_steps <= 0 or args.frames <= 0 or args.fps <= 0:
        parser.error("--sampling-steps, --frames, and --fps must be positive")
    if args.contact_threshold_m <= 0:
        parser.error("--contact-threshold-m must be positive")
    if args.hug_pointcloud is not None and not args.hug_pointcloud.is_file():
        parser.error(f"--hug-pointcloud does not exist: {args.hug_pointcloud}")
    return args


def _load_object_points(path: Path) -> tuple[np.ndarray, np.ndarray, Path | None]:
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points_camera"], dtype=np.float32)
        pixels = np.asarray(data["pixels_uv"], dtype=np.int32)
        source_mask = Path(str(data["source_mask"].item())) if "source_mask" in data else None
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Invalid points_camera in {path}: {points.shape}")
    if pixels.shape != (len(points), 2):
        raise ValueError(f"Invalid pixels_uv in {path}: {pixels.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Object point cloud contains NaN or Inf")
    return points, pixels, source_mask


def _point_from_mask_metadata(source_mask: Path | None) -> tuple[float, float]:
    if source_mask is None:
        raise ValueError("--point is required because pointcloud has no source_mask field")
    metadata_path = source_mask.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"--point is required because mask metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    point = metadata.get("foreground_point")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"Invalid foreground_point in {metadata_path}")
    return float(point[0]), float(point[1])


def _nearest_object_point(
    points: np.ndarray, pixels: np.ndarray, target_uv: tuple[float, float]
) -> np.ndarray:
    target = np.asarray(target_uv, dtype=np.float32)
    index = int(np.argmin(np.sum((pixels.astype(np.float32) - target) ** 2, axis=1)))
    return points[index].copy()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _geometry_metrics(
    prediction: dict[str, Any], tree: cKDTree, contact_threshold_m: float
) -> dict[str, Any]:
    grasp = prediction["grasp"]
    keypoints = np.asarray(grasp["landmarks_3d"], dtype=np.float32)
    tips = keypoints[FINGERTIP_INDICES]
    tip_distances = np.asarray(tree.query(tips, k=1)[0], dtype=np.float32)
    palm_points = keypoints[[0, 5, 9, 13, 17]]
    palm_distance = float(np.min(tree.query(palm_points, k=1)[0]))
    mesh = np.asarray(grasp["mesh_vertices"], dtype=np.float32)
    mesh_surface_min = float(np.min(tree.query(mesh, k=1)[0]))
    best_three = np.sort(tip_distances)[:3]
    return {
        "tip_surface_mean_m": float(np.mean(tip_distances)),
        "tip_surface_best3_mean_m": float(np.mean(best_three)),
        "tip_surface_min_m": float(np.min(tip_distances)),
        "tip_surface_max_m": float(np.max(tip_distances)),
        "tip_surface_distances_m": tip_distances.tolist(),
        "contact_finger_count": int(np.count_nonzero(tip_distances <= contact_threshold_m)),
        "palm_surface_min_m": palm_distance,
        "mesh_surface_min_m": mesh_surface_min,
        "thumb_index_distance_m": float(np.linalg.norm(keypoints[4] - keypoints[8])),
    }


def _l25_metrics(qpos: np.ndarray, solver_metrics: dict[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    qpos = np.asarray(qpos, dtype=np.float64)
    normalized_margin = np.minimum(
        (qpos - lower) / np.maximum(upper - lower, 1e-9),
        (upper - qpos) / np.maximum(upper - lower, 1e-9),
    )
    return {
        "l25_solver_cost": float(solver_metrics["cost"]),
        "l25_solve_ms": float(solver_metrics["solve_ms"]),
        "joint_limit_violations_before_clamp": int(
            solver_metrics["violations_before_clamp"]
        ),
        "joint_limit_violations_after_clamp": int(
            solver_metrics["violations_after_clamp"]
        ),
        "joint_saturation_count": int(np.count_nonzero(normalized_margin <= 0.05)),
        "joint_min_normalized_margin": float(np.min(normalized_margin)),
    }


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    span = float(np.max(values) - minimum)
    return np.zeros_like(values) if span < 1e-12 else (values - minimum) / span


def _rank(rows: list[dict[str, Any]], contact_threshold_m: float) -> None:
    """Rank HUG samples without treating a partial cloud as complete geometry.

    Every candidate is already sampled from the HUG learned grasp distribution.
    Visible-surface distances are gates for obviously detached hands, not rewards
    that pull every fingertip onto the camera-facing surface.
    """
    solver = _minmax(np.asarray([row["l25_solver_cost"] for row in rows]))
    mesh_clearance = np.asarray(
        [row["mesh_surface_min_m"] for row in rows], dtype=np.float64
    )
    tip_clearance = np.asarray(
        [row["tip_surface_best3_mean_m"] for row in rows], dtype=np.float64
    )

    # Do not reward extra visible contacts: they may all lie on the same side.
    mesh_detached = np.clip(
        (mesh_clearance - contact_threshold_m) / contact_threshold_m, 0.0, 1.0
    )
    # Fingertips can legitimately contact an occluded surface. Only penalize a
    # candidate when even its three closest tips are substantially far away.
    tip_detached = np.clip(
        (tip_clearance - 2.0 * contact_threshold_m)
        / (2.0 * contact_threshold_m),
        0.0,
        1.0,
    )
    visible_rejection = 0.7 * mesh_detached + 0.3 * tip_detached
    saturation = np.asarray(
        [row["joint_saturation_count"] / len(L25_JOINT_NAMES) for row in rows]
    )
    violations = np.asarray(
        [row["joint_limit_violations_before_clamp"] for row in rows], dtype=np.float64
    )
    violation_fraction = violations / len(L25_JOINT_NAMES)
    total = (
        0.15 * visible_rejection
        + 0.50 * solver
        + 0.25 * saturation
        + 0.10 * violation_fraction
    )
    order = np.argsort(total)
    for rank, index in enumerate(order, start=1):
        rows[int(index)]["visible_surface_rejection_penalty"] = float(
            visible_rejection[int(index)]
        )
        rows[int(index)]["total_score"] = float(total[int(index)])
        rows[int(index)]["rank"] = rank


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "candidate", "seed", "status", "total_score",
        "tip_surface_mean_m", "tip_surface_best3_mean_m", "tip_surface_min_m",
        "tip_surface_max_m", "contact_finger_count", "palm_surface_min_m",
        "mesh_surface_min_m", "visible_surface_rejection_penalty",
        "thumb_index_distance_m", "hug_inference_ms",
        "l25_solver_cost", "l25_solve_ms", "joint_limit_violations_before_clamp",
        "joint_limit_violations_after_clamp", "joint_saturation_count",
        "joint_min_normalized_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["rank"]))


def main() -> None:
    args = _parse_args()
    rgb = _read_rgb(args.rgb)
    depth = _read_depth_uint16(args.depth)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} != depth shape {depth.shape[:2]}")
    object_points, object_pixels, source_mask = _load_object_points(args.pointcloud)
    point = (
        (float(args.point[0]), float(args.point[1]))
        if args.point is not None
        else _point_from_mask_metadata(source_mask)
    )
    nearest_pixel_distance = float(np.sqrt(np.min(
        np.sum((object_pixels.astype(np.float32) - point) ** 2, axis=1)
    )))
    if nearest_pixel_distance > 20.0:
        raise ValueError(
            f"Target point {point} is {nearest_pixel_distance:.1f}px from the object point cloud. "
            "Use the point stored with this mask or regenerate matching mask/pointcloud data."
        )
    point_224, crop = _map_original_point(point, rgb.shape)
    object_tree = cKDTree(object_points)
    object_point = _nearest_object_point(object_points, object_pixels, point)

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. Pass --overwrite to replace artifacts."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    input_pkl = prepare_pkl(
        rgb,
        depth,
        _load_intrinsics(args.intrinsics),
        "input",
        args.output / "hug_input",
        object_name=args.output.name,
    )
    _save_target_preview(args.output / "target_point.png", rgb, point, crop)
    device = _resolve_device(args.device)
    model = load_model(args.checkpoint, use_ema=True, device=device)
    dataset = GraspDataset(
        str(input_pkl.parent),
        split="val",
        use_rgb=bool(getattr(model, "use_rgb", True)),
        use_depth=bool(getattr(model, "use_depth", False)),
    )

    rows: list[dict[str, Any]] = []
    for candidate_index in range(args.candidates):
        seed = args.seed_start + candidate_index
        _set_seed(seed)
        candidate_name = f"candidate_{candidate_index:03d}"
        candidate_dir = args.output / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        grasp, actual_uv, _depth_m, hug_ms = _run_hug(
            input_pkl,
            args.checkpoint,
            point_224,
            device,
            args.sampling_steps,
            model=model,
            dataset=dataset,
            pcl_seed=seed,
            hug_pointcloud=args.hug_pointcloud,
        )
        prediction = _prediction_payload(input_pkl, grasp, actual_uv)
        _write_pickle(candidate_dir / "prediction.pkl", prediction)
        keypoints = landmarks_from_prediction(prediction)
        canonical = canonical_grasp_from_hug(
            prediction,
            handedness="right",
            object_point_camera=object_point,
            condition_point_224=actual_uv,
        )
        canonical.to_npz(candidate_dir / "canonical_grasp.npz")
        retarget_keypoints = canonical.keypoints_for_retargeting()
        canonical_roundtrip_max_error = float(
            np.max(np.abs(retarget_keypoints - keypoints))
        )
        records, solver_metrics = _retarget_l25(
            retarget_keypoints, args.optimizer, args.frames, args.fps
        )
        _write_pickle(candidate_dir / "trajectory.pkl", records)
        qpos = np.asarray(solver_metrics["qpos"], dtype=np.float32)
        np.savez_compressed(
            candidate_dir / "trajectory.npz",
            source=np.asarray("hug_candidate"),
            human_representation=np.asarray("canonical_grasp_state"),
            human_keypoints_canonical=np.repeat(
                canonical.keypoints_canonical[None].astype(np.float32), args.frames, axis=0
            ),
            robot=np.asarray(args.robot),
            optimizer=np.asarray(args.optimizer),
            seed=np.asarray(seed, dtype=np.int64),
            timestamps=np.arange(args.frames, dtype=np.float64) / args.fps,
            human_keypoints=np.repeat(retarget_keypoints[None].astype(np.float32), args.frames, axis=0),
            robot_qpos=np.repeat(qpos[None], args.frames, axis=0),
            robot_joint_names=np.asarray(L25_JOINT_NAMES),
        )
        metrics = {
            "candidate": candidate_name,
            "seed": seed,
            "status": "success",
            "hug_inference_ms": float(hug_ms),
            "retarget_input_representation": "canonical_grasp_state",
            "hug_pointcloud_input": (
                str(args.hug_pointcloud.resolve())
                if args.hug_pointcloud is not None
                else "anchor_depth_rebuilt"
            ),
            "canonical_to_retarget_max_error": canonical_roundtrip_max_error,
            **_geometry_metrics(prediction, object_tree, args.contact_threshold_m),
            **_l25_metrics(qpos, solver_metrics),
        }
        (candidate_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(metrics)
        print(
            f"[{candidate_index + 1}/{args.candidates}] {candidate_name} seed={seed} "
            f"best3_tip={metrics['tip_surface_best3_mean_m']:.4f}m "
            f"contacts={metrics['contact_finger_count']} "
            f"solver={metrics['l25_solver_cost']:.4g}"
        )

    _rank(rows, args.contact_threshold_m)
    for row in rows:
        metrics_path = args.output / row["candidate"] / "metrics.json"
        metrics_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output / "candidates.csv", rows)
    best = min(rows, key=lambda row: row["rank"])
    summary = {
        "score_version": SCORE_VERSION,
        "score_interpretation": (
            "lower is better; provisional relative ranking of HUG samples for "
            "L25 feasibility"
        ),
        "candidate_semantics": (
            "Every candidate is an unmodified complete MANO grasp sampled by HUG. "
            "The visible point cloud only rejects clearly detached candidates."
        ),
        "hug_pointcloud_input": (
            str(args.hug_pointcloud.resolve())
            if args.hug_pointcloud is not None
            else "anchor_depth_rebuilt"
        ),
        "hug_pointcloud_experiment": (
            "Fused point cloud injected into the pretrained single-view PointNeXt "
            "encoder; compare against anchor_depth_rebuilt before drawing conclusions."
            if args.hug_pointcloud is not None
            else "single_view_baseline"
        ),
        "single_view_limit": (
            "The provisional best candidate is not validated for hidden-surface "
            "collision, force closure, or physical stability."
        ),
        "model_loaded_once": True,
        "candidate_count": args.candidates,
        "contact_threshold_m": args.contact_threshold_m,
        "pointcloud": str(args.pointcloud.resolve()),
        "target_point_original": list(point),
        "target_point_to_cloud_distance_px": nearest_pixel_distance,
        "selection_status": "provisional",
        "best_candidate": best,
        "best_candidate_dir": str((args.output / best["candidate"]).resolve()),
    }
    (args.output / "best_candidate.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    del dataset, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("HUG candidate ranking completed (offline dry-run)")
    print(
        f"  provisional best: {best['candidate']} seed={best['seed']} "
        f"score={best['total_score']:.4f}"
    )
    print(f"  ranking: {args.output / 'candidates.csv'}")
    print(f"  summary: {args.output / 'best_candidate.json'}")


if __name__ == "__main__":
    main()
