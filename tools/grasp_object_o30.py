#!/usr/bin/env python3
"""Run RGB-D point -> HUG -> O30 Vector retargeting as an offline pipeline.

The output is using the portable O30 controller-order contract.
It performs perception and retargeting only; this command never controls hardware.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from anydexretarget.hand_representation import canonical_grasp_from_hug
from anydexretarget.hug_adapter import landmarks_from_prediction
from anydexretarget.hug_o30 import retarget_hug_o30
from anydexretarget.hand_contract import O30_ACTIVE_JOINT_NAMES, O30_QPOS_JOINT_NAMES
from hug.prepare_inputs import _load_intrinsics, _read_depth_uint16, _read_rgb, prepare_pkl
from grasp_object import (
    ROOT,
    _map_original_point,
    _prediction_payload,
    _resolve_device,
    _run_hug,
    _save_target_preview,
    _write_pickle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, required=True, metavar=("U", "V"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "external/hug/checkpoints/hug_full.safetensors",
    )
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sampling_steps <= 0 or args.frames <= 0 or args.fps <= 0:
        parser.error("--sampling-steps, --frames, and --fps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    rgb = _read_rgb(args.rgb)
    depth = _read_depth_uint16(args.depth)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} != depth shape {depth.shape[:2]}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output}; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    point = (float(args.point[0]), float(args.point[1]))
    point_224, crop = _map_original_point(point, rgb.shape)
    input_pkl = prepare_pkl(
        rgb, depth, _load_intrinsics(args.intrinsics), "input", args.output / "hug_input",
        object_name=args.output.name,
    )
    _save_target_preview(args.output / "target_point.png", rgb, point, crop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    grasp, actual_uv, depth_m, hug_ms = _run_hug(
        input_pkl, args.checkpoint, point_224, _resolve_device(args.device),
        args.sampling_steps, pcl_seed=args.seed,
    )
    prediction = _prediction_payload(input_pkl, grasp, actual_uv)
    prediction_path = args.output / "prediction.pkl"
    _write_pickle(prediction_path, prediction)
    keypoints = landmarks_from_prediction(prediction)
    camera_k = np.asarray(prediction["camera"]["K"], dtype=np.float32)
    object_point_camera = np.array([
        (actual_uv[0] - camera_k[0, 2]) * depth_m / camera_k[0, 0],
        (actual_uv[1] - camera_k[1, 2]) * depth_m / camera_k[1, 1],
        depth_m,
    ], dtype=np.float32)
    canonical = canonical_grasp_from_hug(
        prediction, handedness="right", object_point_camera=object_point_camera,
        condition_point_224=actual_uv,
    )
    canonical_path = args.output / "canonical_grasp.npz"
    canonical.to_npz(canonical_path)
    retarget_keypoints = canonical.keypoints_for_retargeting()
    result = retarget_hug_o30(retarget_keypoints)
    records = [
        {
            "timestamp": index / args.fps,
            "target": result.qpos.copy(),
            "sim_qpos": result.qpos.copy(),
            "robot_joint_names": list(O30_QPOS_JOINT_NAMES),
            "hand_command_positions": result.command_positions.copy(),
            "hand_command_joint_names": list(O30_ACTIVE_JOINT_NAMES),
            "human_keypoints": keypoints.astype(np.float32, copy=True),
            "human_keypoints_retarget_frame": result.transformed_keypoints.copy(),
            "solver_cost": result.cost,
            "robot": "o30",
            "optimizer": "vector",
            "dry_run": True,
        }
        for index in range(args.frames)
    ]
    trajectory_path = args.output / "trajectory.pkl"
    _write_pickle(trajectory_path, records)
    timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    np.savez_compressed(
        args.output / "trajectory.npz",
        source=np.asarray("hug"), robot=np.asarray("o30"), optimizer=np.asarray("vector"),
        timestamps=timestamps,
        human_keypoints=np.repeat(keypoints[None].astype(np.float32), args.frames, axis=0),
        human_keypoints_canonical=np.repeat(canonical.keypoints_canonical[None].astype(np.float32), args.frames, axis=0),
        robot_qpos=np.repeat(result.qpos[None], args.frames, axis=0),
        robot_joint_names=np.asarray(O30_QPOS_JOINT_NAMES),
        hand_command_positions=np.repeat(result.command_positions[None], args.frames, axis=0),
        hand_command_joint_names=np.asarray(O30_ACTIVE_JOINT_NAMES),
    )
    metadata = {
        "source": "hug", "robot": "o30", "optimizer": "vector", "dry_run": True,
        "hardware_commands_sent": 0, "rgb": str(args.rgb.resolve()),
        "depth": str(args.depth.resolve()), "intrinsics": str(args.intrinsics.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "requested_point_original": list(point),
        "actual_point_224": actual_uv.tolist(), "condition_depth_m": depth_m,
        "canonical_grasp": str(canonical_path.resolve()), "hug_inference_ms": hug_ms,
        "vector_solve_cost": result.cost, "robot_dof": 20,
        "robot_joint_names": list(O30_QPOS_JOINT_NAMES),
        "hand_command_joint_names": list(O30_ACTIVE_JOINT_NAMES),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("HUG RGB-D point -> O30 Vector completed (offline)")
    print(f"  HUG keypoints: {keypoints.shape}; O30 qpos: {result.qpos.shape}")
    print(f"  prediction: {prediction_path}")
    print(f"  canonical grasp: {canonical_path}")
    print(f"  O30 trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
