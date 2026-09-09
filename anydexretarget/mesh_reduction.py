"""Deterministic triangle reduction for offline collision proxies."""

from __future__ import annotations

import numpy as np
import trimesh


def reduce_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Voxel-cluster a mesh for collision while retaining its outer geometry."""
    if max_faces <= 0:
        raise ValueError("max_faces must be positive")
    if len(mesh.faces) <= max_faces:
        return mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lower = vertices.min(axis=0)
    extent = vertices.max(axis=0) - lower
    positive = extent[extent > 1e-9]
    if len(positive) == 0:
        raise ValueError("Cannot reduce a zero-size mesh")
    voxel = max(float(np.prod(positive) / max(1000, max_faces // 2)) ** (1.0 / 3.0), float(positive.min()) / 10000.0)
    for _ in range(24):
        _keys, inverse = np.unique(np.floor((vertices - lower) / voxel).astype(np.int64), axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        reduced_vertices = np.column_stack([np.bincount(inverse, weights=vertices[:, axis]) / counts for axis in range(3)])
        reduced_faces = inverse[faces]
        keep = (reduced_faces[:, 0] != reduced_faces[:, 1]) & (reduced_faces[:, 1] != reduced_faces[:, 2]) & (reduced_faces[:, 0] != reduced_faces[:, 2])
        reduced_faces = reduced_faces[keep]
        _unique, indices = np.unique(np.sort(reduced_faces, axis=1), axis=0, return_index=True)
        reduced_faces = reduced_faces[np.sort(indices)]
        if 0 < len(reduced_faces) <= max_faces:
            return trimesh.Trimesh(vertices=reduced_vertices, faces=reduced_faces, process=False)
        voxel *= 1.35
    raise RuntimeError("Unable to reduce collision mesh to the requested face budget")


__all__ = ["reduce_mesh"]
