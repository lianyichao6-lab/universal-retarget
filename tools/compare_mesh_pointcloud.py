#!/usr/bin/env python3
"""Compare an aligned generated mesh against the original RGB-D object cloud."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial import cKDTree


def load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(tuple(raw.geometry.values()))
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"Invalid mesh: {path}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pointcloud", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-pointcloud", type=Path, help="Optional mesh-sampled HUG input cloud.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--point-size", type=float, default=0.002)
    parser.add_argument("--distance-samples", type=int, default=100000)
    args = parser.parse_args()
    with np.load(args.raw_pointcloud, allow_pickle=False) as data:
        raw_points = np.asarray(data["points_camera"], dtype=np.float32)
        raw_colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ValueError("Raw point cloud must contain N x 3 points_camera")
    mesh = load_mesh(args.mesh)
    np.random.seed(0)
    reference_surface, _ = trimesh.sample.sample_surface(mesh, args.distance_samples)
    distances, _ = cKDTree(reference_surface).query(raw_points, k=1)
    report = {
        "raw_pointcloud": str(args.raw_pointcloud.resolve()),
        "mesh": str(args.mesh.resolve()),
        "distance_definition": "nearest distance to a 100k uniformly sampled mesh surface; approximate, unsigned",
        "raw_to_mesh_distance_m": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(np.max(distances)),
            "fraction_within_10mm": float(np.mean(distances <= 0.010)),
            "fraction_within_20mm": float(np.mean(distances <= 0.020)),
        },
        "raw_observed_bounds_m": raw_points.min(axis=0).tolist() + raw_points.max(axis=0).tolist(),
        "raw_observed_extents_m": np.ptp(raw_points, axis=0).tolist(),
        "mesh_bounds_m": mesh.bounds.tolist(),
        "mesh_extents_m": mesh.extents.tolist(),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "interpretation": "The raw cloud only covers visible surfaces, so its full-object extents are not a valid size ground truth. Use nearest-surface distance and visual overlap to judge alignment.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    mesh_points = None
    if args.mesh_pointcloud:
        with np.load(args.mesh_pointcloud, allow_pickle=False) as data:
            mesh_points = np.asarray(data["points_camera"], dtype=np.float32)
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.002)
    server.scene.add_point_cloud("/raw_rgbd_observed", points=raw_points, colors=raw_colors,
                                 point_size=args.point_size * 1.25, point_shape="rounded")
    handle = server.scene.add_mesh_simple("/aligned_hunyuan_mesh", vertices=np.asarray(mesh.vertices, dtype=np.float32),
                                           faces=np.asarray(mesh.faces, dtype=np.int32), color=(65, 205, 190))
    handle.opacity = 0.32
    if mesh_points is not None:
        server.scene.add_point_cloud("/mesh_sampled_hug_input", points=mesh_points,
                                     colors=np.tile(np.asarray((255, 174, 45), dtype=np.uint8), (len(mesh_points), 1)),
                                     point_size=args.point_size, point_shape="rounded")
    center = raw_points.mean(axis=0)
    server.scene.add_label("/labels/raw", text="raw RGB-D observed surface", position=center + np.array([0.0, -0.08, 0.05]), anchor="bottom-left")
    server.scene.add_label("/labels/mesh", text="aligned Hunyuan mesh", position=center + np.array([0.0, -0.08, 0.025]), anchor="bottom-left")
    server.scene.add_label("/labels/sample", text="orange: mesh samples used by HUG", position=center + np.array([0.0, -0.08, 0.0]), anchor="bottom-left")
    print(json.dumps(report["raw_to_mesh_distance_m"], indent=2))
    print(f"report: {args.report}")
    print(f"viewer: http://localhost:{args.port}")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
