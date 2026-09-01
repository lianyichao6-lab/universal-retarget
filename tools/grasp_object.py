#!/usr/bin/env python3
"""Generate one HUG grasp at an RGB-D pixel and retarget it to LinkerHand L25.

This is an offline-only pipeline. It never imports or calls a hardware SDK.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Keep local HUG inference deterministic and independent of a broken shell proxy.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

import cv2
import mujoco
import numpy as np
import torch

from anydexretarget.hug_adapter import landmarks_from_prediction
from anydexretarget.hand_representation import canonical_grasp_from_hug
from anydexretarget.retarget import Retargeter
from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend
from hug.dataloader.data_classes import CameraIntrinsics, Grasp, GraspData
from hug.dataloader.grasp_dataset import GraspDataset
from hug.inference import load_model
from hug.models.mano import mano_params_to_grasp_dict
from hug.prepare_inputs import (
    TARGET_SIZE,
    _load_intrinsics,
    _read_depth_uint16,
    _read_rgb,
    prepare_pkl,
)
from hug.utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz, sample_fixed_n


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "l25": {
        "vector": ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml",
        "adaptive": ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml",
        **DEX_CONFIGS,
    }
}
L25_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
L25_JOINT_NAMES = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        required=True,
        metavar=("U", "V"),
        help="Target pixel in the original RGB image (not the 224x224 crop).",
    )
    parser.add_argument("--robot", choices=sorted(CONFIGS), default="l25")
    parser.add_argument("--optimizer", choices=("vector", "adaptive", "dexpilot", "joint_angle"), default="vector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "external/hug/checkpoints/hug_full.safetensors",
    )
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offline only. --no-dry-run is rejected because hardware sending is not implemented.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("hardware execution is not implemented; use --dry-run")
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be positive")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return requested


def _map_original_point(
    point: tuple[float, float], image_shape: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = image_shape[:2]
    side = min(h, w)
    x_off = (w - side) // 2
    y_off = (h - side) // 2
    u, v = point
    if not (x_off <= u < x_off + side and y_off <= v < y_off + side):
        raise ValueError(
            f"--point ({u:g}, {v:g}) is outside HUG's center crop. "
            f"For this {w}x{h} RGB image choose u in [{x_off}, {x_off + side - 1}] "
            f"and v in [{y_off}, {y_off + side - 1}]."
        )
    mapped = np.array(
        [(u - x_off) * TARGET_SIZE / side, (v - y_off) * TARGET_SIZE / side],
        dtype=np.float32,
    )
    mapped = np.clip(mapped, 0.0, TARGET_SIZE - 1.0)
    return mapped, {"x": x_off, "y": y_off, "size": side}


def _nearest_valid_depth(
    depth_mm: np.ndarray, uv: np.ndarray, max_radius: int = 12
) -> tuple[np.ndarray, float]:
    h, w = depth_mm.shape[:2]
    u0 = int(np.clip(round(float(uv[0])), 0, w - 1))
    v0 = int(np.clip(round(float(uv[1])), 0, h - 1))
    value = int(depth_mm[v0, u0])
    if 0 < value < 65535:
        return np.array([float(uv[0]), float(uv[1])], np.float32), value / 1000.0
    for radius in range(1, max_radius + 1):
        x0, x1 = max(0, u0 - radius), min(w, u0 + radius + 1)
        y0, y1 = max(0, v0 - radius), min(h, v0 + radius + 1)
        patch = depth_mm[y0:y1, x0:x1]
        ys, xs = np.where((patch > 0) & (patch < 65535))
        if len(xs):
            distances = (xs + x0 - u0) ** 2 + (ys + y0 - v0) ** 2
            index = int(np.argmin(distances))
            u1, v1 = int(xs[index] + x0), int(ys[index] + y0)
            return np.array([float(u1), float(v1)], np.float32), float(depth_mm[v1, u1]) / 1000.0
    raise ValueError(
        f"No valid depth within {max_radius} pixels of HUG point ({u0}, {v0})"
    )


def _write_pickle(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream)
    tmp.replace(path)


def _save_target_preview(
    path: Path,
    rgb: np.ndarray,
    point: tuple[float, float],
    crop: dict[str, int],
) -> None:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x0, y0, side = crop["x"], crop["y"], crop["size"]
    cv2.rectangle(image, (x0, y0), (x0 + side - 1, y0 + side - 1), (0, 220, 255), 2)
    cv2.drawMarker(
        image,
        (int(round(point[0])), int(round(point[1]))),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=3,
    )
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write target preview: {path}")


def _build_external_pcl_tensors(
    pointcloud: Path,
    center: np.ndarray,
    crop_radius: float | None,
    pcl_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an anchor-frame fused cloud in HUG's PointNeXt input format."""
    with np.load(pointcloud, allow_pickle=False) as data:
        if "points_camera" not in data or "colors_rgb" not in data:
            raise ValueError(
                f"--hug-pointcloud must contain points_camera and colors_rgb: {pointcloud}"
            )
        points = np.asarray(data["points_camera"], dtype=np.float32)
        colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(
            f"Invalid --hug-pointcloud arrays: points={points.shape}, colors={colors.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("--hug-pointcloud contains NaN or Inf")
    if crop_radius is not None:
        keep = np.linalg.norm(points - center[None], axis=1) <= crop_radius
        points, colors = points[keep], colors[keep]
    if len(points) < 32:
        raise ValueError(
            "--hug-pointcloud has fewer than 32 points near the selected object point; "
            "check that it is fused in the anchor camera frame and overlaps the click."
        )
    points, colors = sample_fixed_n(
        points, colors, 4096, rng=np.random.default_rng(pcl_seed)
    )
    return torch.from_numpy(points).float(), torch.from_numpy(colors).float() / 255.0


def _run_hug(
    input_pkl: Path,
    checkpoint: Path,
    point_224: np.ndarray,
    device: str,
    sampling_steps: int,
    *,
    model: Any | None = None,
    dataset: GraspDataset | None = None,
    pcl_seed: int | None = None,
    hug_pointcloud: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, float, float]:
    owns_model = model is None
    owns_dataset = dataset is None
    if model is None:
        model = load_model(checkpoint, use_ema=True, device=device)
    use_rgb = bool(getattr(model, "use_rgb", True))
    use_depth = bool(getattr(model, "use_depth", False))
    pcl_use_rgb = bool(getattr(model, "pcl_use_rgb", False))
    if dataset is None:
        dataset = GraspDataset(
            str(input_pkl.parent), split="val", use_rgb=use_rgb, use_depth=use_depth
        )
    sample = dataset.get_inference_data(input_pkl.stem)
    actual_uv, depth_m = _nearest_valid_depth(sample["depth_image"], point_224)
    camera_k_np = np.asarray(sample["camera_K"], dtype=np.float32)
    point_uv = torch.tensor(
        [[actual_uv[0], actual_uv[1], depth_m]], dtype=torch.float32, device=device
    )
    camera_k = torch.from_numpy(camera_k_np).unsqueeze(0).to(device)
    rgb_tensor = sample["rgb"].unsqueeze(0).to(device) if use_rgb else None

    pcl_xyz = None
    pcl_rgb = None
    if use_depth:
        crop_radius = getattr(model, "pcl_crop_radius", None)
        center = pixel_to_xyz(actual_uv[0], actual_uv[1], depth_m, camera_k_np)
        if hug_pointcloud is not None:
            pcl_xyz, pcl_rgb = _build_external_pcl_tensors(
                hug_pointcloud, center, crop_radius, pcl_seed
            )
            pcl_xyz = pcl_xyz.unsqueeze(0).to(device)
            pcl_rgb = pcl_rgb.unsqueeze(0).to(device) if pcl_use_rgb else None
        elif crop_radius is not None:
            depth_m_image = sample["depth_image"].astype(np.float32) / 1000.0
            xyz, colors = depth_to_pcl_tensors(
                depth_m_image,
                sample["rgb_original"],
                camera_k_np,
                center=center,
                crop_radius=crop_radius,
                rng=np.random.default_rng(pcl_seed),
            )
            pcl_xyz = xyz.unsqueeze(0).to(device)
            pcl_rgb = colors.unsqueeze(0).to(device) if pcl_use_rgb else None
        else:
            pcl_xyz = sample["pcl_xyz"].unsqueeze(0).to(device)
            pcl_rgb = sample["pcl_rgb"].unsqueeze(0).to(device) if pcl_use_rgb else None

    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        prediction = model.sample(
            point_uv,
            camera_k,
            steps=sampling_steps,
            rgb=rgb_tensor,
            pcl_xyz=pcl_xyz,
            pcl_rgb=pcl_rgb,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000.0
    grasp = mano_params_to_grasp_dict(
        prediction[0],
        model.fixed_betas.squeeze(0),
        model.mano,
        camera_k_np,
        model.mesh_faces,
    )
    del prediction
    if owns_model:
        del model
    if owns_dataset:
        del dataset
    if owns_model or owns_dataset:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return grasp, actual_uv, depth_m, inference_ms


def _prediction_payload(
    input_pkl: Path,
    grasp: dict[str, Any],
    point_224: np.ndarray,
) -> dict[str, Any]:
    with input_pkl.open("rb") as stream:
        source = pickle.load(stream)
    camera = source["camera"]
    camera_original = source["camera_original"]
    point_norm = np.asarray(point_224, dtype=np.float32) / float(TARGET_SIZE)
    return asdict(
        GraspData(
            object_name=source.get("object_name", ""),
            frame_index=int(source.get("frame_index", 0)),
            grasp_index=int(source.get("grasp_index", 0)),
            camera=CameraIntrinsics(**camera) if isinstance(camera, dict) else camera,
            camera_original=(
                CameraIntrinsics(**camera_original)
                if isinstance(camera_original, dict)
                else camera_original
            ),
            grasp=Grasp(**grasp),
            image=source.get("image", b""),
            depth=source.get("depth", b""),
            object_mask=point_norm.tobytes(),
            condition_point=np.asarray(point_224, dtype=np.float32),
        )
    )


def _retarget_l25(
    keypoints: np.ndarray,
    optimizer_name: str,
    frames: int,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray | float | int]]:
    started = time.perf_counter()
    if optimizer_name in DEX_CONFIGS:
        backend = DexRetargetBackend(optimizer_name, hand_side="right")
        qpos_backend, backend_verbose = backend.retarget(keypoints)
        qpos = qpos_backend
        verbose = {
            "mediapipe_kp": backend_verbose["mediapipe_kp"],
            "cost": float(backend.optimizer.opt.last_optimum_value()),
        }
        source_names = [str(name).lower() for name in backend.joint_names]
        retarget_robot = backend.optimizer.robot
        is_dex_backend = True
    else:
        retargeter = Retargeter.from_yaml(
            str(CONFIGS["l25"][optimizer_name]), hand_side="right"
        )
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        source_names = [str(name).lower() for name in retargeter.optimizer.robot.dof_joint_names]
        retarget_robot = retargeter.optimizer.robot
        is_dex_backend = False
    solve_ms = (time.perf_counter() - started) * 1000.0
    source_by_name = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in L25_JOINT_NAMES if name.lower() not in source_by_name]
    if missing:
        raise ValueError(f"Retargeter output is missing L25 joints: {missing}")
    target = np.asarray(
        [qpos[source_by_name[name.lower()]] for name in L25_JOINT_NAMES],
        dtype=np.float64,
    )

    model = mujoco.MjModel.from_xml_path(str(L25_MODEL))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if model_names != L25_JOINT_NAMES:
        raise ValueError("L25 MuJoCo joint order differs from the audited output order")
    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    # Pinocchio uses float64 while the MuJoCo XML range is represented at
    # float32 precision. Ignore sub-microradian boundary round-off.
    limit_tolerance = 1e-6
    violations_before = int(
        np.count_nonzero(
            (target < lower - limit_tolerance) | (target > upper + limit_tolerance)
        )
    )
    # Clamp inside the physical limits, then evaluate the serialized float32
    # values against the actual closed limits rather than the safety margin.
    target = np.clip(target, lower + 1e-6, upper - 1e-6).astype(np.float32)
    violations_after = int(np.count_nonzero((target < lower) | (target > upper)))
    if not np.isfinite(target).all():
        raise ValueError("Retargeted qpos contains NaN or Inf")

    if is_dex_backend:
        robot = retarget_robot
        tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        robot.compute_forward_kinematics(qpos)
        fingertips = np.asarray(
            [robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in tip_names],
            dtype=np.float32,
        )
    else:
        robot = retarget_robot
        tip_ids = [robot.get_link_index(name) for name in retargeter.optimizer.task_link_names]
        fingertips = robot.compute_points_batch(
            qpos, tip_ids, retargeter.optimizer.task_offsets
        ).astype(np.float32)
    transformed = np.asarray(verbose["mediapipe_kp"], dtype=np.float32)
    cost = float(verbose["cost"])
    records = [
        {
            "timestamp": index / fps,
            "target": target.copy(),
            "sim_qpos": target.copy(),
            "robot_joint_names": L25_JOINT_NAMES.copy(),
            "human_keypoints": keypoints.astype(np.float32, copy=True),
            "human_keypoints_retarget_frame": transformed.copy(),
            "fingertip_positions": fingertips.copy(),
            "solver_cost": cost,
            "solve_time_ms": solve_ms,
            "robot": "l25",
            "optimizer": optimizer_name,
            "dry_run": True,
        }
        for index in range(frames)
    ]
    metrics: dict[str, np.ndarray | float | int] = {
        "qpos": target,
        "fingertips": fingertips,
        "transformed_keypoints": transformed,
        "cost": cost,
        "solve_ms": solve_ms,
        "violations_before_clamp": violations_before,
        "violations_after_clamp": violations_after,
    }
    return records, metrics


def main() -> None:
    args = _parse_args()
    rgb = _read_rgb(args.rgb)
    depth = _read_depth_uint16(args.depth)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} != depth shape {depth.shape[:2]}")
    point = (float(args.point[0]), float(args.point[1]))
    point_224, crop = _map_original_point(point, rgb.shape)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. Pass --overwrite to replace artifacts."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir = args.output / "hug_input"
    input_pkl = prepare_pkl(
        rgb,
        depth,
        _load_intrinsics(args.intrinsics),
        "input",
        input_dir,
        object_name=args.output.name,
    )
    _save_target_preview(args.output / "target_point.png", rgb, point, crop)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
    grasp, actual_uv, depth_m, hug_ms = _run_hug(
        input_pkl,
        args.checkpoint,
        point_224,
        device,
        args.sampling_steps,
        pcl_seed=args.seed,
    )
    prediction = _prediction_payload(input_pkl, grasp, actual_uv)
    prediction_path = args.output / "prediction.pkl"
    _write_pickle(prediction_path, prediction)
    keypoints = landmarks_from_prediction(prediction)
    camera_k = np.asarray(prediction["camera"]["K"], dtype=np.float32)
    object_point_camera = np.array(
        [
            (actual_uv[0] - camera_k[0, 2]) * depth_m / camera_k[0, 0],
            (actual_uv[1] - camera_k[1, 2]) * depth_m / camera_k[1, 1],
            depth_m,
        ],
        dtype=np.float32,
    )
    canonical_state = canonical_grasp_from_hug(
        prediction,
        handedness="right",
        object_point_camera=object_point_camera,
        condition_point_224=actual_uv,
    )
    canonical_path = args.output / "canonical_grasp.npz"
    canonical_state.to_npz(canonical_path)
    retarget_keypoints = canonical_state.keypoints_for_retargeting()
    canonical_roundtrip_max_error = float(
        np.max(np.abs(retarget_keypoints - keypoints))
    )

    records, metrics = _retarget_l25(
        retarget_keypoints, args.optimizer, args.frames, args.fps
    )
    trajectory_path = args.output / "trajectory.pkl"
    _write_pickle(trajectory_path, records)
    timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    np.savez_compressed(
        args.output / "trajectory.npz",
        video=np.asarray(""),
        source=np.asarray("hug"),
        human_representation=np.asarray("canonical_grasp_state"),
        human_keypoints_canonical=np.repeat(
            canonical_state.keypoints_canonical[None].astype(np.float32), args.frames, axis=0
        ),
        robot=np.asarray(args.robot),
        optimizer=np.asarray(args.optimizer),
        timestamps=timestamps,
        human_keypoints=np.repeat(keypoints[None].astype(np.float32), args.frames, axis=0),
        robot_qpos=np.repeat(np.asarray(metrics["qpos"])[None], args.frames, axis=0),
        robot_joint_names=np.asarray(L25_JOINT_NAMES),
        fingertip_positions=np.repeat(
            np.asarray(metrics["fingertips"])[None], args.frames, axis=0
        ),
        solver_cost=np.full(args.frames, metrics["cost"], dtype=np.float64),
        solve_time_ms=np.full(args.frames, metrics["solve_ms"], dtype=np.float64),
    )
    metadata = {
        "source": "hug",
        "robot": args.robot,
        "optimizer": args.optimizer,
        "dry_run": True,
        "hardware_commands_sent": 0,
        "rgb": str(args.rgb.resolve()),
        "depth": str(args.depth.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "image_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "center_crop": crop,
        "requested_point_original": list(point),
        "requested_point_224": point_224.tolist(),
        "actual_point_224": actual_uv.tolist(),
        "condition_depth_m": depth_m,
        "canonical_grasp": str(canonical_path.resolve()),
        "canonical_grasp_schema_version": 1,
        "retarget_input_representation": "canonical_grasp_state",
        "canonical_to_retarget_max_error": canonical_roundtrip_max_error,
        "canonical_grasp_finite": bool(
            np.isfinite(canonical_state.keypoints_canonical).all()
            and np.isfinite(canonical_state.mano_mesh_vertices_camera).all()
        ),
        "sampling_steps": args.sampling_steps,
        "hug_inference_ms": hug_ms,
        "frames": args.frames,
        "fps": args.fps,
        "robot_dof": len(L25_JOINT_NAMES),
        "robot_joint_names": L25_JOINT_NAMES,
        "solver_cost": metrics["cost"],
        "solve_time_ms": metrics["solve_ms"],
        "joint_limit_violations_before_clamp": metrics["violations_before_clamp"],
        "joint_limit_violations_after_clamp": metrics["violations_after_clamp"],
        "human_keypoints_shape": list(keypoints.shape),
        "human_keypoints_finite": bool(np.isfinite(keypoints).all()),
        "robot_qpos_finite": bool(np.isfinite(metrics["qpos"]).all()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("HUG grasp -> L25 retargeting completed (offline dry-run)")
    print(f"  HUG keypoints: {keypoints.shape}, finite={np.isfinite(keypoints).all()}")
    print(f"  Robot qpos: {np.asarray(metrics['qpos']).shape}, finite={np.isfinite(metrics['qpos']).all()}")
    print(f"  HUG inference: {hug_ms:.1f} ms; retarget solve: {float(metrics['solve_ms']):.1f} ms")
    print(f"  Joint limit violations after clamp: {metrics['violations_after_clamp']}")
    print(f"  Prediction: {prediction_path}")
    print(f"  Canonical grasp state: {canonical_path}")
    print(f"  RViz trajectory: {trajectory_path}")
    print(f"  Structured trajectory: {args.output / 'trajectory.npz'}")
    print(f"  Metadata: {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
