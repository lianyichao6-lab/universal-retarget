"""Blender-side renderer for ``render_cad_hug_inputs.py``.

The model-to-camera transform uses OpenCV camera coordinates (X right, Y down,
Z forward). Blender's camera frame differs by Y/Z sign, hence the fixed
conversion below. This script is intentionally Blender-only and is launched by
the regular Python wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


CAMERA_TO_BLENDER = Matrix(((1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0),
                            (0.0, 0.0, -1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--model-to-camera", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth-npy", type=Path, required=True)
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _camera(intrinsics: np.ndarray, width: int, height: int) -> None:
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if not np.isclose(fx, fy, rtol=1e-4):
        raise ValueError("Synthetic renderer currently requires fx == fy")
    data = bpy.data.cameras.new("HUG_Anchor_Camera")
    data.sensor_width = 36.0
    data.lens = float(fx * data.sensor_width / width)
    data.shift_x = float((width * 0.5 - cx) / width)
    data.shift_y = float((cy - height * 0.5) / width)
    data.lens_unit = "MILLIMETERS"
    camera = bpy.data.objects.new("HUG_Anchor_Camera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera


def _import_mesh(path: Path, transform: np.ndarray) -> None:
    bpy.ops.wm.obj_import(filepath=str(path))
    matrix = CAMERA_TO_BLENDER @ Matrix(transform.tolist())
    imported = list(bpy.context.selected_objects)
    if not imported:
        raise ValueError(f"No mesh imported from {path}")
    for object_ in imported:
        if object_.type != "MESH":
            continue
        object_.matrix_world = matrix @ object_.matrix_world
        if not object_.data.materials:
            material = bpy.data.materials.new("CAD_Default_Material")
            material.diffuse_color = (0.36, 0.38, 0.42, 1.0)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs["Base Color"].default_value = material.diffuse_color
            bsdf.inputs["Roughness"].default_value = 0.42
            object_.data.materials.append(material)


def _lighting() -> None:
    world = bpy.context.scene.world
    world.color = (0.035, 0.035, 0.035)
    for location, energy, size in (((0.18, -0.20, -0.16), 900.0, 0.18),
                                  ((-0.16, 0.08, -0.24), 600.0, 0.14)):
        data = bpy.data.lights.new("CAD_Softbox", type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new("CAD_Softbox", data)
        bpy.context.scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (0.0, 0.0, 0.0)


def _depth_from_raycast(scene: bpy.types.Scene, intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    depth = np.zeros((height, width), dtype=np.float32)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = Vector((0.0, 0.0, 0.0))
    for row in range(height):
        for col in range(width):
            direction = Vector(((col - cx) / fx, -(row - cy) / fy, -1.0)).normalized()
            hit, location, _, _, _, _ = scene.ray_cast(depsgraph, origin, direction)
            if hit:
                depth[row, col] = max(0.0, -float(location.z))
    return depth


def main() -> None:
    args = _args()
    transform = np.loadtxt(args.model_to_camera, dtype=np.float64)
    intrinsics = np.loadtxt(args.intrinsics, dtype=np.float64)
    if transform.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("Expected 4x4 model transform and 3x3 intrinsics")
    _clear_scene()
    _camera(intrinsics, args.width, args.height)
    _import_mesh(args.mesh.resolve(), transform)
    _lighting()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = True
    args.rgb.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.rgb.resolve())
    bpy.ops.render.render(write_still=True)
    depth = _depth_from_raycast(scene, intrinsics, args.width, args.height)
    args.depth_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.depth_npy, depth)
    print(f"CAD RGB render written: {args.rgb}")
    print(f"CAD depth array written: {args.depth_npy}")


if __name__ == "__main__":
    main()
