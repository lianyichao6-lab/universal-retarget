"""Build a Blender CAD workspace from an anchor RGB-D capture and observed geometry.

Run this file with Blender, not the project virtual environment.  The scene uses
the anchor camera coordinate frame: OpenCV/HUG points (X right, Y down, Z forward)
are converted into Blender camera coordinates (X right, Y up, camera looks -Z).
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


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--observed-pointcloud", type=Path, required=True)
    parser.add_argument("--surface-proxy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Alpha"].default_value = color[3]
    material.surface_render_method = "DITHERED"
    return material


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _configure_camera(rgb: Path, intrinsics: np.ndarray, resolution: tuple[int, int]) -> bpy.types.Object:
    width, height = resolution
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    camera_data = bpy.data.cameras.new("Anchor_RGBD_Camera")
    camera_data.lens_unit = "MILLIMETERS"
    camera_data.sensor_width = 36.0
    camera_data.lens = float(fx * camera_data.sensor_width / width)
    camera_data.shift_x = float((width * 0.5 - cx) / width)
    camera_data.shift_y = float((cy - height * 0.5) / width)
    camera = bpy.data.objects.new("Anchor_RGBD_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    camera_data.show_background_images = True
    background = camera_data.background_images.new()
    background.image = bpy.data.images.load(str(rgb), check_existing=True)
    background.frame_method = "FIT"
    background.alpha = 0.55
    return camera


def _observed_points(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points_camera"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Invalid observed points: {points.shape}")
    return points


def _make_point_cloud(points_camera: np.ndarray) -> bpy.types.Object:
    points = np.c_[points_camera, np.ones(len(points_camera))] @ np.asarray(CAMERA_TO_BLENDER).T
    mesh = bpy.data.meshes.new("Observed_RGBD_Points_Mesh")
    mesh.from_pydata(points[:, :3].tolist(), [], [])
    object_ = bpy.data.objects.new("Observed_RGBD_Points", mesh)
    bpy.context.scene.collection.objects.link(object_)
    modifier = object_.modifiers.new("Visible_Points", "NODES")
    tree = bpy.data.node_groups.new("Observed_RGBD_Point_Display", "GeometryNodeTree")
    modifier.node_group = tree
    input_node = tree.nodes.new("NodeGroupInput")
    output_node = tree.nodes.new("NodeGroupOutput")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    to_points = tree.nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    to_points.inputs["Radius"].default_value = 0.0018
    tree.links.new(input_node.outputs["Geometry"], to_points.inputs["Mesh"])
    tree.links.new(to_points.outputs["Points"], output_node.inputs["Geometry"])
    object_.data.materials.append(_material("Observed_Point_Material", (0.08, 0.42, 1.0, 1.0)))
    return object_


def _import_proxy(path: Path) -> bpy.types.Object:
    bpy.ops.wm.ply_import(filepath=str(path))
    imported = bpy.context.selected_objects[-1]
    imported.name = "Observed_Surface_Proxy"
    imported.matrix_world = CAMERA_TO_BLENDER @ imported.matrix_world
    imported.data.materials.clear()
    imported.data.materials.append(_material("Surface_Proxy_Material", (0.15, 0.75, 0.95, 0.30)))
    return imported


def _add_draft_bounds(points_camera: np.ndarray) -> bpy.types.Object:
    minimum, maximum = np.min(points_camera, axis=0), np.max(points_camera, axis=0)
    center_camera = (minimum + maximum) * 0.5
    extent = np.maximum(maximum - minimum, 0.002)
    center = CAMERA_TO_BLENDER @ Vector((float(center_camera[0]), float(center_camera[1]), float(center_camera[2]), 1.0))
    bpy.ops.mesh.primitive_cube_add(location=center[:3])
    cube = bpy.context.active_object
    cube.name = "CAD_DRAFT_BOUNDS_DELETE_OR_EDIT"
    cube.dimensions = tuple(float(value) for value in extent)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    wire = cube.modifiers.new("CAD_DRAFT_BOUNDS", "WIREFRAME")
    wire.thickness = 0.001
    cube.data.materials.append(_material("CAD_Draft_Material", (1.0, 0.38, 0.05, 1.0)))
    return cube


def _label(text: str, location: tuple[float, float, float]) -> None:
    bpy.ops.object.text_add(location=location)
    label = bpy.context.active_object
    label.name = text
    label.data.body = text
    label.data.size = 0.025
    label.data.extrude = 0.0002
    label.data.materials.append(_material(f"{text}_Material", (1.0, 1.0, 1.0, 1.0)))


def main() -> None:
    args = _arguments()
    rgb = args.rgb.resolve()
    intrinsics_path = args.intrinsics.resolve()
    observed_path = args.observed_pointcloud.resolve()
    if not rgb.is_file() or not intrinsics_path.is_file() or not observed_path.is_file():
        raise FileNotFoundError("RGB, intrinsics, and observed point cloud must exist")
    image = bpy.data.images.load(str(rgb), check_existing=True)
    width, height = image.size
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, got {intrinsics.shape}")
    points = _observed_points(observed_path)
    _clear_scene()
    _configure_camera(rgb, intrinsics, (width, height))
    _make_point_cloud(points)
    if args.surface_proxy is not None:
        proxy = args.surface_proxy.resolve()
        if not proxy.is_file():
            raise FileNotFoundError(proxy)
        _import_proxy(proxy)
    _add_draft_bounds(points)
    _label("Blue: observed RGB-D points", (0.0, 0.10, -0.25))
    _label("Orange: edit or replace this CAD draft", (0.0, 0.06, -0.25))
    bpy.context.scene.world.color = (0.025, 0.025, 0.025)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"Blender RGB-D CAD workspace written: {args.output}")


if __name__ == "__main__":
    main()
