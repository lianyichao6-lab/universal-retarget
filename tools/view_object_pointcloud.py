#!/usr/bin/env python3
"""Serve an object point cloud and optional surface-proxy mesh in Viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument(
        "--overlay-pointcloud", type=Path,
        help="Optional second point cloud, displayed in amber for alignment comparison.",
    )
    parser.add_argument(
        "--mesh", type=Path,
        help="Optional PLY/OBJ/STL mesh to overlay on the object point cloud.",
    )
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--point-size", type=float, default=0.004)
    args = parser.parse_args()
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    with np.load(args.pointcloud, allow_pickle=False) as data:
        points = data["points_camera"].copy()
        colors = data["colors_rgb"].copy()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Invalid points_camera in {args.pointcloud}: {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors_rgb shape {colors.shape} does not match {points.shape}")
    overlay_points = None
    if args.overlay_pointcloud is not None:
        with np.load(args.overlay_pointcloud, allow_pickle=False) as data:
            overlay_points = np.asarray(data["points_camera"], dtype=np.float32)
        if overlay_points.ndim != 2 or overlay_points.shape[1] != 3 or len(overlay_points) == 0:
            raise ValueError(
                f"Invalid overlay points_camera in {args.overlay_pointcloud}: {overlay_points.shape}"
            )

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.002)
    server.scene.add_point_cloud(
        "/object_points",
        points=points,
        colors=colors,
        point_size=args.point_size,
        point_shape="rounded",
    )
    if overlay_points is not None:
        server.scene.add_point_cloud(
            "/alignment_overlay",
            points=overlay_points,
            colors=np.tile(np.asarray((255, 181, 46), dtype=np.uint8), (len(overlay_points), 1)),
            point_size=args.point_size * 1.15,
            point_shape="rounded",
        )
    if args.mesh is not None:
        mesh = trimesh.load_mesh(args.mesh, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Invalid triangle mesh: {args.mesh}")
        mesh_handle = server.scene.add_mesh_simple(
            "/surface_proxy",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            color=(110, 190, 255),
        )
        mesh_handle.opacity = 0.38
    print(f"Serving {len(points)} object points at http://localhost:{args.port}")
    if args.mesh is not None:
        print(f"Overlaying {len(mesh.faces)} mesh faces from {args.mesh}")
    if args.overlay_pointcloud is not None:
        print(f"Overlaying CAD/reference points from {args.overlay_pointcloud}")
    print("Inspect for incorrect bridges or artificial caps. Stop with Ctrl-C.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
