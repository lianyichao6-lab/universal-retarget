#!/usr/bin/env python3
"""Inspect one HUG object-relative contact plan in Viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from anydexretarget.hand_representation import BONE_EDGES, load_canonical_grasp_state


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(tuple(raw.geometry.values()))
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"Invalid object mesh: {path}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-plan", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    with np.load(args.contact_plan, allow_pickle=False) as data:
        canonical_path = Path(str(data["source_canonical_grasp"].item()))
        mesh_path = Path(str(data["source_object_mesh"].item()))
        tips = np.asarray(data["fingertip_positions_camera"], dtype=np.float32)
        anchors = np.asarray(data["surface_anchor_camera"], dtype=np.float32)
        normals = np.asarray(data["surface_normal_camera"], dtype=np.float32)
        near = np.asarray(data["near_surface"], dtype=np.uint8).astype(bool)
        distances = np.asarray(data["tip_surface_distance_m"], dtype=np.float32)
        names = [str(value) for value in data["finger_names"]]
        object_to_camera = np.asarray(data["object_to_camera"], dtype=np.float32)
    state = load_canonical_grasp_state(canonical_path)
    object_mesh = _load_mesh(mesh_path)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame(
        "/object_frame",
        position=object_to_camera[:3, 3],
        axes_length=0.05,
        axes_radius=0.0015,
    )
    object_handle = server.scene.add_mesh_simple(
        "/object_mesh",
        vertices=np.asarray(object_mesh.vertices, dtype=np.float32),
        faces=np.asarray(object_mesh.faces, dtype=np.int32),
        color=(73, 196, 186),
    )
    object_handle.opacity = 0.30
    hand_handle = server.scene.add_mesh_simple(
        "/hug_mano_mesh",
        vertices=np.asarray(state.mano_mesh_vertices_camera, dtype=np.float32),
        faces=np.asarray(state.mano_mesh_faces, dtype=np.int32),
        color=(180, 120, 255),
    )
    hand_handle.opacity = 0.62
    keypoints = np.asarray(state.keypoints_camera, dtype=np.float32)
    segments = np.stack((keypoints[BONE_EDGES[:, 0]], keypoints[BONE_EDGES[:, 1]]), axis=1)
    server.scene.add_line_segments(
        "/mano_skeleton", points=segments,
        colors=np.full((len(segments), 2, 3), 255, dtype=np.uint8), line_width=2.5,
    )
    tip_colors = np.where(near[:, None], np.asarray((65, 220, 110), dtype=np.uint8), np.asarray((255, 166, 45), dtype=np.uint8))
    anchor_colors = np.where(near[:, None], np.asarray((20, 245, 245), dtype=np.uint8), np.asarray((255, 90, 90), dtype=np.uint8))
    server.scene.add_point_cloud("/fingertips", points=tips, colors=tip_colors, point_size=0.012, point_shape="rounded")
    server.scene.add_point_cloud("/surface_anchors", points=anchors, colors=anchor_colors, point_size=0.010, point_shape="rounded")
    line_colors = np.repeat(tip_colors[:, None, :], 2, axis=1)
    server.scene.add_line_segments("/tip_to_surface", points=np.stack((tips, anchors), axis=1), colors=line_colors, line_width=3.0)
    normal_endpoints = anchors + normals * 0.025
    server.scene.add_line_segments(
        "/surface_normals", points=np.stack((anchors, normal_endpoints), axis=1),
        colors=np.tile(np.asarray((255, 230, 65), dtype=np.uint8), (len(anchors), 2, 1)), line_width=2.0,
    )
    for name, tip, distance, active in zip(names, tips, distances, near):
        server.scene.add_label(
            f"/labels/{name}",
            text=f"{name}: {distance * 1000:.1f} mm {'near' if active else 'far'}",
            position=tip + np.array((0.0, -0.018, 0.0), dtype=np.float32),
            anchor="bottom-left",
        )
    print(f"Contact-plan viewer: http://localhost:{args.port}")
    print("green tip/cyan anchor = near surface; orange/red = not near; yellow = surface normal")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
