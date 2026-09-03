#!/usr/bin/env python3
"""Prepare a direct CAD-mesh input bundle for the existing HUG pipeline.

This renders one synthetic anchor RGB-D view of a meter-scale CAD mesh and
exports both point-cloud representations HUG needs:

* ``object_pointcloud.npz`` is the rendered visible surface and contains
  ``pixels_uv`` for the HUG target click.
* ``hug_pointcloud.npz`` is a uniformly sampled full CAD surface in the same
  anchor camera frame. Pass it as ``--hug-pointcloud`` to HUG.

The default 35 cm, three-quarter view is deliberately an initial simulation
pose, not a real-camera calibration. It is sufficient for direct CAD-to-HUG
experiments and can later be replaced by a measured model-to-camera transform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Invalid triangle mesh: {path}")
    return loaded


def _rotation_matrix(azimuth_deg: float, elevation_deg: float, roll_deg: float) -> np.ndarray:
    azimuth, elevation, roll = np.deg2rad([azimuth_deg, elevation_deg, roll_deg])
    cz, sz = np.cos(azimuth), np.sin(azimuth)
    cx, sx = np.cos(elevation), np.sin(elevation)
    cy, sy = np.cos(roll), np.sin(roll)
    rotate_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rotate_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    return rotate_z @ rotate_x @ rotate_y


def _find_blender(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    for candidate in (
        shutil.which("blender"),
        ROOT.parent / ".tools/blender-5.2.0-linux-x64/blender",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Blender not found; pass --blender /path/to/blender")


def _sample_full_surface(mesh: trimesh.Trimesh, transform: np.ndarray, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    points_model, face_index = trimesh.sample.sample_surface(mesh, samples)
    points_camera = (np.c_[points_model, np.ones(len(points_model))] @ transform.T)[:, :3]
    colors = getattr(mesh.visual, "face_colors", None)
    if colors is not None and len(colors) == len(mesh.faces):
        rgb = np.asarray(colors[face_index, :3], dtype=np.uint8)
    else:
        rgb = np.full((len(points_camera), 3), 145, dtype=np.uint8)
    return points_camera.astype(np.float32), rgb


def _visible_cloud(rgb: np.ndarray, depth_mm: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = depth_mm > 0
    rows, cols = np.nonzero(valid)
    if len(rows) < 32:
        raise RuntimeError("Synthetic depth contains too few object pixels")
    z = depth_mm[rows, cols].astype(np.float32) / 1000.0
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    points = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z)).astype(np.float32)
    pixels = np.column_stack((cols, rows)).astype(np.int32)
    return points, pixels, rgb[rows, cols, :3].astype(np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-unit", choices=tuple(UNIT_TO_METERS), default="m")
    parser.add_argument("--distance-m", type=float, default=0.35)
    parser.add_argument("--azimuth-deg", type=float, default=25.0)
    parser.add_argument("--elevation-deg", type=float, default=-12.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender", help="Blender executable. Defaults to PATH or local .tools installation.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.distance_m <= 0 or args.width < 32 or args.height < 32 or args.fx <= 0 or args.fy <= 0 or args.samples < 32:
        parser.error("distance, image dimensions, focal lengths, and samples must be positive")
    return args


def main() -> None:
    args = _parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    mesh = _as_mesh(mesh_path)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * UNIT_TO_METERS[args.model_unit]
    model_to_camera = np.eye(4, dtype=np.float64)
    model_to_camera[:3, :3] = _rotation_matrix(args.azimuth_deg, args.elevation_deg, args.roll_deg)
    model_to_camera[:3, 3] = (0.0, 0.0, args.distance_m)
    intrinsics = np.array(((args.fx, 0.0, (args.width - 1) * 0.5),
                           (0.0, args.fy, (args.height - 1) * 0.5),
                           (0.0, 0.0, 1.0)), dtype=np.float64)
    transform_path = output / "model_to_camera.txt"
    intrinsics_path = output / "intrinsics.txt"
    np.savetxt(transform_path, model_to_camera, fmt="%.10f")
    np.savetxt(intrinsics_path, intrinsics, fmt="%.10f")
    blender = _find_blender(args.blender)
    depth_npy = output / "_depth_m.npy"
    command = [
        str(blender), "--background", "--python", str(ROOT / "tools/_blender_render_cad_hug_inputs.py"), "--",
        "--mesh", str(mesh_path), "--model-to-camera", str(transform_path), "--intrinsics", str(intrinsics_path),
        "--width", str(args.width), "--height", str(args.height), "--rgb", str(output / "rgb.png"),
        "--depth-npy", str(depth_npy),
    ]
    subprocess.run(command, check=True)
    if not depth_npy.is_file():
        raise RuntimeError("Blender completed without writing depth; inspect Blender output above")
    depth_m = np.load(depth_npy)
    depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(str(output / "depth.png"), depth_mm):
        raise RuntimeError("Unable to write synthetic depth.png")
    depth_npy.unlink(missing_ok=True)
    rgb = cv2.imread(str(output / "rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError("Unable to read Blender RGB render")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    points_visible, pixels_uv, colors_visible = _visible_cloud(rgb, depth_mm, intrinsics)
    mask = (depth_mm > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(output / "object_mask.png"), mask):
        raise RuntimeError("Unable to write synthetic object mask")
    np.savez_compressed(
        output / "object_pointcloud.npz",
        points_camera=points_visible,
        pixels_uv=pixels_uv,
        colors_rgb=colors_visible,
        source_mask=np.asarray(str((output / "object_mask.png").resolve())),
        source_type=np.asarray("synthetic_cad_render_visible_surface"),
    )
    mesh_camera = mesh.copy()
    mesh_camera.apply_transform(model_to_camera)
    mesh_camera.export(output / "object_mesh_camera.ply")
    full_points, full_colors = _sample_full_surface(mesh, model_to_camera, args.samples, args.seed)
    if np.any(full_points[:, 2] <= 0):
        raise RuntimeError("CAD placement put part of the full surface behind the camera")
    np.savez_compressed(
        output / "hug_pointcloud.npz",
        points_camera=full_points,
        colors_rgb=full_colors,
        source_type=np.asarray("synthetic_cad_full_surface"),
        source_mesh=np.asarray(str(mesh_path)),
        model_to_camera=model_to_camera,
    )
    target_uv = np.rint(np.median(pixels_uv, axis=0)).astype(int)
    metadata = {
        "source_type": "synthetic_cad_hug_input",
        "source_mesh": str(mesh_path),
        "coordinate_frame": "synthetic_anchor_camera_opencv",
        "model_unit": args.model_unit,
        "model_to_camera": model_to_camera.tolist(),
        "intrinsics": intrinsics.tolist(),
        "visible_points": int(len(points_visible)),
        "full_surface_points": int(len(full_points)),
        "target_point_uv": target_uv.tolist(),
        "files": {
            "rgb": "rgb.png", "depth": "depth.png", "intrinsics": "intrinsics.txt",
            "visible_pointcloud": "object_pointcloud.npz", "full_surface_pointcloud": "hug_pointcloud.npz",
            "object_mesh": "object_mesh_camera.ply",
        },
    }
    (output / "cad_hug_input.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic CAD HUG inputs written: {output}")
    print(f"  visible surface: {len(points_visible)} points")
    print(f"  full CAD surface: {len(full_points)} points")
    print(f"  HUG target point: {target_uv[0]} {target_uv[1]}")
    print("  next: pass object_pointcloud.npz as --pointcloud and hug_pointcloud.npz as --hug-pointcloud")


if __name__ == "__main__":
    main()
