#!/usr/bin/env python3
"""Convert a MANUS recording to standard 21x3 and an L25 qpos trajectory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend
from anydexretarget.l25_retarget_backend import (
    ADAPTIVE_CONFIG,
    BACKENDS,
    VECTOR_CONFIG,
)
from anydexretarget.manus_adapter import canonical_hand_frame_from_manus
from anydexretarget.retarget import Retargeter


def _solver(backend: str, hand: str) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    if backend in ("vector", "adaptive"):
        config = VECTOR_CONFIG if backend == "vector" else ADAPTIVE_CONFIG
        retargeter = Retargeter.from_yaml(str(config), hand_side=hand)
        joint_names = [
            str(name) for name in retargeter.optimizer.robot.dof_joint_names
        ]

        def solve(points: np.ndarray) -> np.ndarray:
            return np.asarray(
                retargeter.retarget(points, apply_filter=True), dtype=np.float64
            )

        return solve, joint_names

    geometry = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side=hand)
    target_names = [str(name) for name in geometry.optimizer.robot.dof_joint_names]
    dex = DexRetargetBackend(backend, hand_side=hand)
    source = {
        str(name).lower(): index for index, name in enumerate(dex.joint_names)
    }
    missing = [name for name in target_names if name.lower() not in source]
    if missing:
        raise ValueError("Dex backend is missing L25 joints: " + ", ".join(missing))
    indices = np.asarray([source[name.lower()] for name in target_names], dtype=np.int64)

    def solve(points: np.ndarray) -> np.ndarray:
        qpos, _ = dex.retarget(points)
        return np.asarray(qpos, dtype=np.float64)[indices]

    return solve, target_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="vector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    with np.load(args.input, allow_pickle=False) as data:
        hand = str(data["handedness"].item())
        timestamps = data["timestamps_s"][:: args.stride]
        wrists = data["wrists"][:: args.stride]
        keypoints = data["keypoints_25"][:: args.stride]
        masks = data["keypoint_masks"][:: args.stride]
        ergonomics = data["ergonomics"][:: args.stride]
    if hand not in ("left", "right"):
        raise ValueError(f"Unsupported handedness: {hand!r}")

    solve, joint_names = _solver(args.backend, hand)
    records = []
    canonical_frames = []
    rejected = 0
    for timestamp, wrist, points, mask, ergo in zip(
        timestamps, wrists, keypoints, masks, ergonomics
    ):
        try:
            frame = canonical_hand_frame_from_manus(
                wrist,
                points,
                int(mask),
                handedness=hand,
                timestamp_s=float(timestamp),
                ergonomics=ergo if np.isfinite(ergo).all() else None,
            )
            human_points = frame.keypoints_for_retargeting()
            qpos = solve(human_points)
            if not np.isfinite(qpos).all():
                raise ValueError("backend produced NaN or Inf")
        except ValueError:
            rejected += 1
            continue
        canonical_frames.append(frame.keypoints_canonical)
        records.append(
            {
                "target": qpos,
                "timestamp_s": float(timestamp),
                "human_keypoints": human_points,
                "backend": args.backend,
                "joint_names": joint_names,
            }
        )
    if not records:
        raise RuntimeError("No MANUS frame could be retargeted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream)

    canonical_output = args.canonical_output
    if canonical_output is not None:
        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            canonical_output,
            schema_version=np.asarray(1, dtype=np.int64),
            source=np.asarray("manus"),
            handedness=np.asarray(hand),
            timestamps_s=np.asarray(
                [record["timestamp_s"] for record in records], dtype=np.float64
            ),
            keypoints_21=np.stack([record["human_keypoints"] for record in records]),
            keypoints_canonical=np.stack(canonical_frames),
        )
    print(
        f"MANUS -> L25 {args.backend} complete: "
        f"{len(records)} frames, {rejected} rejected"
    )
    print(f"  trajectory: {args.output}")
    if canonical_output is not None:
        print(f"  canonical frames: {canonical_output}")


if __name__ == "__main__":
    main()
