from __future__ import annotations

import json

import numpy as np
import trimesh

from anydexretarget.reconstruction_adapter import adapt_reconstruction_mesh, load_mesh, load_transform


def test_explicit_metric_mesh_import_preserves_visible_points(tmp_path):
    mesh_path = tmp_path / "object.glb"
    anchor_path = tmp_path / "anchor.npz"
    transform_path = tmp_path / "pose.json"
    output_dir = tmp_path / "result"
    mesh = trimesh.creation.box(extents=(0.08, 0.06, 0.10))
    mesh.export(mesh_path)
    transform = np.eye(4)
    transform[:3, 3] = (0.03, -0.02, 0.5)
    transform_path.write_text(json.dumps({"T_anchor_object": transform.tolist()}), encoding="utf-8")
    np.random.seed(4)
    visible, _ = trimesh.sample.sample_surface(mesh, 500)
    visible = visible @ transform[:3, :3].T + transform[:3, 3]
    colors = np.tile(np.asarray((12, 34, 56), dtype=np.uint8), (len(visible), 1))
    np.savez_compressed(anchor_path, points_camera=visible.astype(np.float32), colors_rgb=colors)

    result = adapt_reconstruction_mesh(
        mesh_path=mesh_path,
        anchor_cloud_path=anchor_path,
        output_dir=output_dir,
        backend="3dgenerationpipeline-fine",
        anchor_frame="waist_camera_color_optical_frame",
        transform=load_transform(transform_path),
        auto_align=False,
        surface_samples=1200,
        alignment_samples=1000,
    )

    with np.load(result["outputs"]["surface"], allow_pickle=False) as data:
        assert np.allclose(data["points_camera"][: len(visible)], visible, atol=1e-6)
        assert np.array_equal(data["colors_rgb"][: len(visible)], colors)
        assert np.all(data["is_observed"][: len(visible)] == 1)
        assert np.all(data["confidence"][: len(visible)] == 1.0)
        assert str(data["anchor_frame"]) == "waist_camera_color_optical_frame"
    metadata = json.loads(result["outputs"]["metadata"].read_text(encoding="utf-8"))
    assert metadata["backend"] == "3dgenerationpipeline-fine"
    assert metadata["alignment"]["mode"] == "explicit_transform"
    assert np.allclose(metadata["alignment"]["aligned_extent_m"], (0.08, 0.06, 0.10), atol=1e-6)


def test_transform_rejects_scaled_rotation(tmp_path):
    path = tmp_path / "bad.npy"
    transform = np.eye(4)
    transform[:3, :3] *= 2.0
    np.save(path, transform)
    try:
        load_transform(path)
    except ValueError as error:
        assert "orthonormal" in str(error)
    else:
        raise AssertionError("scaled transform was accepted")


def test_glb_scene_node_transform_is_applied(tmp_path):
    path = tmp_path / "transformed.glb"
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)
    scene.add_geometry(
        trimesh.creation.box(extents=(0.1, 0.2, 0.3)),
        transform=transform,
    )
    path.write_bytes(scene.export(file_type="glb"))
    mesh = load_mesh(path)
    assert np.allclose(mesh.centroid, transform[:3, 3], atol=1e-6)
