#!/usr/bin/env python3
"""Show top-ranked HUG MANO candidates beside their object point cloud."""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from anydexretarget.hand_representation import BONE_EDGES


COLORS = (
    (77, 163, 255),
    (255, 157, 77),
    (99, 205, 125),
    (196, 122, 255),
    (240, 210, 85),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, help="Optional aligned object mesh overlaid beside every candidate.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--point-size", type=float, default=0.003)
    args = parser.parse_args()
    if args.top_k <= 0 or args.point_size <= 0:
        parser.error("--top-k and --point-size must be positive")
    return args


def _load_ranking(path: Path, top_k: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    successful = [row for row in rows if row.get("status") == "success"]
    successful.sort(key=lambda row: int(row["rank"]))
    if not successful:
        raise ValueError(f"No successful candidates in {path}")
    return successful[:top_k]


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    grasp = payload["grasp"]
    vertices = np.asarray(grasp["mesh_vertices"], dtype=np.float32)
    faces = np.asarray(grasp["mesh_faces"], dtype=np.int32)
    keypoints = np.asarray(grasp["landmarks_3d"], dtype=np.float32)
    if vertices.shape != (778, 3) or keypoints.shape != (21, 3):
        raise ValueError(f"Unexpected HUG geometry in {path}")
    return vertices, faces, keypoints


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(tuple(raw.geometry.values()))
    if not isinstance(raw, trimesh.Trimesh) or len(raw.vertices) == 0 or len(raw.faces) == 0:
        raise ValueError(f"Invalid object mesh: {path}")
    return np.asarray(raw.vertices, dtype=np.float32), np.asarray(raw.faces, dtype=np.int32)


def main() -> None:
    args = _parse_args()
    ranking = _load_ranking(args.candidates_dir / "candidates.csv", args.top_k)
    with np.load(args.pointcloud, allow_pickle=False) as data:
        object_points = np.asarray(data["points_camera"], dtype=np.float32)
        object_colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
    if object_points.ndim != 2 or object_points.shape[1] != 3 or len(object_points) == 0:
        raise ValueError(f"Invalid object point cloud: {object_points.shape}")

    object_mesh = _load_mesh(args.mesh) if args.mesh else None

    extent_x = float(np.ptp(object_points[:, 0]))
    spacing = max(extent_x + 0.15, 0.3)
    label_z = float(np.max(object_points[:, 2]) + 0.08)
    label_y = float(np.min(object_points[:, 1]))
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.002)

    for display_index, row in enumerate(ranking):
        candidate = row["candidate"]
        offset = np.array([display_index * spacing, 0.0, 0.0], dtype=np.float32)
        vertices, faces, keypoints = _load_prediction(
            args.candidates_dir / candidate / "prediction.pkl"
        )
        base = f"/rank_{row['rank']}_{candidate}"
        if object_mesh:
            mesh_vertices, mesh_faces = object_mesh
            object_mesh_handle = server.scene.add_mesh_simple(
                base + "/full_object_mesh",
                vertices=mesh_vertices + offset,
                faces=mesh_faces,
                color=(73, 196, 186),
            )
            object_mesh_handle.opacity = 0.32
        server.scene.add_point_cloud(
            base + "/object",
            points=object_points + offset,
            colors=object_colors,
            point_size=args.point_size,
            point_shape="rounded",
        )
        mesh = server.scene.add_mesh_simple(
            base + "/mano",
            vertices=vertices + offset,
            faces=faces,
            color=COLORS[display_index % len(COLORS)],
        )
        mesh.opacity = 0.65
        segments = np.stack(
            (keypoints[BONE_EDGES[:, 0]], keypoints[BONE_EDGES[:, 1]]), axis=1
        ) + offset
        server.scene.add_line_segments(
            base + "/skeleton",
            points=segments,
            colors=np.full((len(segments), 2, 3), 255, dtype=np.uint8),
            line_width=3.0,
        )
        server.scene.add_label(
            base + "/label",
            text=(
                f"rank {row['rank']} | {candidate} | seed {row['seed']} | "
                f"score {float(row['total_score']):.4f}"
            ),
            position=offset + np.array([0.0, label_y, label_z], dtype=np.float32),
            anchor="bottom-left",
        )

    print(f"Serving top {len(ranking)} candidates at http://localhost:{args.port}")
    print("Rotate/zoom and reject candidates with implausible hand-object geometry.")
    print("Stop with Ctrl-C.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
