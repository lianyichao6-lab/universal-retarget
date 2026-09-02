#!/usr/bin/env python3
"""Project an anchor-camera mesh onto its source RGB image for alignment checks.

The mesh must already be expressed in the RGB-D anchor camera frame.  This is
an inspection tool: it does not change the mesh or HUG input point cloud.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [mesh for mesh in raw.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No triangle mesh found: {path}")
    return raw


def _load_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 camera matrix, got {values.shape}: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="Anchor RGB image.")
    parser.add_argument("--intrinsics", type=Path, required=True, help="Anchor 3x3 K matrix.")
    parser.add_argument("--mesh", type=Path, required=True, help="Mesh already aligned to anchor camera coordinates.")
    parser.add_argument("--output", type=Path, required=True, help="Output RGB overlay PNG.")
    parser.add_argument("--silhouette-output", type=Path, help="Optional mesh silhouette PNG.")
    parser.add_argument("--alpha", type=float, default=0.28, help="Mesh fill opacity in [0, 1].")
    parser.add_argument("--stride", type=int, default=2, help="Draw every Nth triangle for a faster diagnostic preview.")
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0 or args.stride <= 0:
        raise ValueError("--alpha must be in (0, 1) and --stride must be positive")

    image = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {args.rgb}")
    height, width = image.shape[:2]
    mesh = _load_mesh(args.mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    K = _load_intrinsics(args.intrinsics)

    depth = vertices[:, 2]
    pixels = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-6
    pixels[valid, 0] = K[0, 0] * vertices[valid, 0] / depth[valid] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * vertices[valid, 1] / depth[valid] + K[1, 2]

    valid_faces = valid[faces].all(axis=1)
    visible_faces = faces[valid_faces][:: args.stride]
    face_depth = depth[visible_faces].mean(axis=1)
    # Paint far-to-near to make the projected outer boundary visually solid.
    order = np.argsort(face_depth)[::-1]
    fill = image.copy()
    silhouette = np.zeros((height, width), dtype=np.uint8)
    mesh_color_bgr = (190, 205, 65)  # cyan in RGB
    for face in visible_faces[order]:
        triangle = np.rint(pixels[face]).astype(np.int32)
        if ((triangle[:, 0] < -1).all() or (triangle[:, 0] >= width).all() or
                (triangle[:, 1] < -1).all() or (triangle[:, 1] >= height).all()):
            continue
        cv2.fillConvexPoly(fill, triangle, mesh_color_bgr, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(silhouette, triangle, 255, lineType=cv2.LINE_AA)

    overlay = cv2.addWeighted(fill, args.alpha, image, 1.0 - args.alpha, 0.0)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(overlay, "cyan: aligned reconstruction mesh", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"Failed to write {args.output}")
    if args.silhouette_output:
        args.silhouette_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.silhouette_output), silhouette):
            raise RuntimeError(f"Failed to write {args.silhouette_output}")
    coverage = float(np.mean(silhouette > 0))
    print(f"overlay: {args.output}")
    if args.silhouette_output:
        print(f"silhouette: {args.silhouette_output}")
    print(f"projected mesh coverage: {coverage:.2%} of image pixels")


if __name__ == "__main__":
    main()
