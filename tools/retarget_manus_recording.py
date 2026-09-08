#!/usr/bin/env python3
"""Convert a MANUS recording to standard 21x3 and a hand qpos trajectory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend
from anydexretarget.l25_retarget_backend import (
    ADAPTIVE_CONFIG,
    BACKENDS as L25_BACKENDS,
    VECTOR_CONFIG as L25_VECTOR_CONFIG,
)
from anydexretarget.manus_adapter import canonical_hand_frame_from_manus
from anydexretarget.o30_retarget_backend import VECTOR_CONFIG as O30_VECTOR_CONFIG
from anydexretarget.retarget import Retargeter


ROBOT_BACKENDS = {
    "l25": L25_BACKENDS,
    "o30": ("vector",),
}


def _solver(
    backend: str, hand: str, robot: str
) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    if backend not in ROBOT_BACKENDS[robot]:
        supported = ", ".join(ROBOT_BACKENDS[robot])
        raise ValueError(f"{robot.upper()} supports: {supported}; got {backend}")

    if robot == "o30":
        retargeter = Retargeter.from_yaml(str(O30_VECTOR_CONFIG), hand_side=hand)
        joint_names = [str(name) for name in retargeter.optimizer.robot.dof_joint_names]

        def solve_o30(points: np.ndarray) -> np.ndarray:
            return np.asarray(
                retargeter.retarget(points, apply_filter=True), dtype=np.float64
            )

        return solve_o30, joint_names

    if backend in ("vector", "adaptive"):
        config = L25_VECTOR_CONFIG if backend == "vector" else ADAPTIVE_CONFIG
        retargeter = Retargeter.from_yaml(str(config), hand_side=hand)
        joint_names = [str(name) for name in retargeter.optimizer.robot.dof_joint_names]

        def solve_l25(points: np.ndarray) -> np.ndarray:
            return np.asarray(
                retargeter.retarget(points, apply_filter=True), dtype=np.float64
            )

        return solve_l25, joint_names

    geometry = Retargeter.from_yaml(str(L25_VECTOR_CONFIG), hand_side=hand)
    target_names = [str(name) for name in geometry.optimizer.robot.dof_joint_names]
    dex = DexRetargetBackend(backend, hand_side=hand)
    source = {str(name).lower(): index for index, name in enumerate(dex.joint_names)}
    missing = [name for name in target_names if name.lower() not in source]
    if missing:
        raise ValueError("Dex backend is missing L25 joints: " + ", ".join(missing))
    indices = np.asarray([source[name.lower()] for name in target_names], dtype=np.int64)

    def solve_dex(points: np.ndarray) -> np.ndarray:
        qpos, _ = dex.retarget(points)
        return np.asarray(qpos, dtype=np.float64)[indices]

    return solve_dex, target_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--robot", choices=sorted(ROBOT_BACKENDS), default="l25")
    parser.add_argument("--backend", default="vector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.backend not in ROBOT_BACKENDS[args.robot]:
        supported = ", ".join(ROBOT_BACKENDS[args.robot])
        raise ValueError(f"{args.robot.upper()} supports: {supported}; got {args.backend}")
    with np.load(args.input, allow_pickle=False) as data:
        hand = str(data["handedness"].item())
        timestamps = data["timestamps_s"][:: args.stride]
        wrists = data["wrists"][:: args.stride]
        keypoints = data["keypoints_25"][:: args.stride]
        masks = data["keypoint_masks"][:: args.stride]
        ergonomics = data["ergonomics"][:: args.stride]
    if hand not in ("left", "right"):
        raise ValueError(f"Unsupported handedness: {hand!r}")

    solve, joint_names = _solver(args.backend, hand, args.robot)
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
                "robot": args.robot,
                "joint_names": joint_names,
            }
        )
    if not records:
        raise RuntimeError("No MANUS frame could be retargeted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream)

    if args.canonical_output is not None:
        args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.canonical_output,
            schema_version=np.asarray(1, dtype=np.int64),
            source=np.asarray("manus"),
            robot=np.asarray(args.robot),
            handedness=np.asarray(hand),
            timestamps_s=np.asarray([record["timestamp_s"] for record in records], dtype=np.float64),
            keypoints_21=np.stack([record["human_keypoints"] for record in records]),
            keypoints_canonical=np.stack(canonical_frames),
        )
    print(
        f"MANUS -> {args.robot.upper()} {args.backend} complete: "
        f"{len(records)} frames, {rejected} rejected"
    )
    print(f"  trajectory: {args.output}")
    if args.canonical_output is not None:
        print(f"  canonical frames: {args.canonical_output}")


if __name__ == "__main__":
    main()
