#!/usr/bin/env python3
"""Back-project a binary object mask and registered RGB-D into a point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hug.prepare_inputs import _load_intrinsics, _read_depth_uint16, _read_rgb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ply",
        type=Path,
        help="Optional PLY output. Defaults to <output stem>.ply.",
    )
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_depth_m <= 0:
        parser.error("--max-depth-m must be positive")
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    return args


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read binary object mask: {path}")
    return mask > 0


def _backproject(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth_mm.shape or rgb.shape[:2] != mask.shape:
        raise ValueError(
            "RGB, depth, and mask must share original registered resolution; "
            f"got rgb={rgb.shape[:2]}, depth={depth_mm.shape}, mask={mask.shape}"
        )
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = mask & (depth_m > 0) & (depth_m < max_depth_m)
    v, u = np.nonzero(valid)
    if u.size == 0:
        raise ValueError("Object mask contains no valid depth points")
    z = depth_m[v, u]
    xyz = np.column_stack(
        (
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
        )
    ).astype(np.float32)
    return xyz, rgb[v, u].astype(np.uint8), np.column_stack((u, v)).astype(np.int32)


def _subsample(
    xyz: np.ndarray, colors: np.ndarray, pixels: np.ndarray, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xyz) <= max_points:
        return xyz, colors, pixels
    indices = np.random.default_rng(seed).choice(len(xyz), size=max_points, replace=False)
    return xyz[indices], colors[indices], pixels[indices]


def _write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, colors):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    args = _parse_args()
    rgb = _read_rgb(args.rgb)
    depth_mm = _read_depth_uint16(args.depth)
    mask = _read_mask(args.mask)
    K = _load_intrinsics(args.intrinsics).astype(np.float32)
    xyz, colors, pixels = _backproject(rgb, depth_mm, mask, K, args.max_depth_m)
    source_count = len(xyz)
    xyz, colors, pixels = _subsample(xyz, colors, pixels, args.max_points, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_camera=xyz,
        colors_rgb=colors,
        pixels_uv=pixels,
        intrinsics=K,
        source_rgb=np.asarray(str(args.rgb.resolve())),
        source_depth=np.asarray(str(args.depth.resolve())),
        source_mask=np.asarray(str(args.mask.resolve())),
    )
    ply_path = args.ply or args.output.with_suffix(".ply")
    _write_ply(ply_path, xyz, colors)
    bounds_min = xyz.min(axis=0)
    bounds_max = xyz.max(axis=0)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "rgb": str(args.rgb.resolve()),
                "depth": str(args.depth.resolve()),
                "mask": str(args.mask.resolve()),
                "intrinsics": str(args.intrinsics.resolve()),
                "point_count_before_subsample": source_count,
                "point_count": int(len(xyz)),
                "max_depth_m": args.max_depth_m,
                "bounds_camera_m": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
                "ply": str(ply_path.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Object point cloud written")
    print(f"  points: {len(xyz)} (mask-valid before subsample: {source_count})")
    print(f"  camera bounds [m]: min={bounds_min.tolist()} max={bounds_max.tolist()}")
    print(f"  npz: {args.output}")
    print(f"  ply: {ply_path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
