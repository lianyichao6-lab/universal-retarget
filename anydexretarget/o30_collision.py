"""Conservative O30 mesh collision checks in the HUG retarget frame.

This is an offline candidate filter, not a replacement for force feedback or a
full rigid-body grasp simulator. It rejects O30 self collisions and object
surface intrusions by the palm/non-tip links, while allowing fingertip contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hppfcl
import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .o30_retarget_backend import VECTOR_CONFIG
from .retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
O30_URDF = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _link_name(geometry_name: str) -> str:
    return geometry_name.removesuffix("_0")


def _finger(link_name: str) -> str | None:
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        if link_name.startswith(name + "_"):
            return name
    return None


def _sample_vertices(mesh_path: str, count: int) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid O30 collision mesh: {mesh_path}")
    indices = np.linspace(0, len(vertices) - 1, min(count, len(vertices)), dtype=np.int64)
    return vertices[indices]


@dataclass(frozen=True)
class O30CollisionResult:
    """One static pose collision report."""

    self_collision_pairs: tuple[str, ...]
    forbidden_object_links: tuple[str, ...]
    link_object_min_distance_m: dict[str, float]
    fingertip_object_distance_m: dict[str, float]
    object_surface_min_distance_m: float

    @property
    def self_collision_count(self) -> int:
        return len(self.self_collision_pairs)

    @property
    def forbidden_object_collision_count(self) -> int:
        return len(self.forbidden_object_links)

    @property
    def fingertip_contact_count(self) -> int:
        return sum(distance <= 0.012 for distance in self.fingertip_object_distance_m.values())

    @property
    def safe(self) -> bool:
        return self.self_collision_count == 0 and self.forbidden_object_collision_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "self_collision_pairs": list(self.self_collision_pairs),
            "self_collision_count": self.self_collision_count,
            "forbidden_object_links": list(self.forbidden_object_links),
            "forbidden_object_collision_count": self.forbidden_object_collision_count,
            "link_object_min_distance_m": self.link_object_min_distance_m,
            "fingertip_object_distance_m": self.fingertip_object_distance_m,
            "fingertip_contact_count": self.fingertip_contact_count,
            "object_surface_min_distance_m": self.object_surface_min_distance_m,
            "safe": self.safe,
        }


class O30CollisionEvaluator:
    """Evaluate O30 mesh self collision and point-cloud object clearance."""

    def __init__(
        self,
        object_points_camera: np.ndarray,
        *,
        wrist_position_camera: np.ndarray,
        canonical_basis_row: np.ndarray,
        mesh_vertices_per_link: int = 384,
        forbidden_clearance_m: float = 0.003,
    ) -> None:
        if forbidden_clearance_m <= 0:
            raise ValueError("forbidden_clearance_m must be positive")
        points = np.asarray(object_points_camera, dtype=np.float64)
        wrist = np.asarray(wrist_position_camera, dtype=np.float64)
        basis = np.asarray(canonical_basis_row, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("object_points_camera must be finite N x 3")
        if wrist.shape != (3,) or basis.shape != (3, 3):
            raise ValueError("invalid canonical hand frame")

        retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
        rotation_cfg = retargeter.rotation_xyz or {}
        rotation = Rotation.from_euler(
            "xyz",
            [rotation_cfg.get(axis, 0.0) for axis in ("x", "y", "z")],
            degrees=True,
        ).as_matrix()
        object_points = (points - wrist[None]) @ basis @ rotation.T
        self.object_tree = cKDTree(object_points)
        self.forbidden_clearance_m = float(forbidden_clearance_m)

        self.model = pin.buildModelFromUrdf(str(O30_URDF))
        self.geometry_model = pin.buildGeomFromUrdf(
            self.model,
            str(O30_URDF),
            pin.GeometryType.COLLISION,
            [str(O30_URDF.parent)],
        )
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self.geometry_names = [_link_name(item.name) for item in self.geometry_model.geometryObjects]
        self.local_vertices = [
            _sample_vertices(item.meshPath, mesh_vertices_per_link)
            for item in self.geometry_model.geometryObjects
        ]
        self.self_pairs = self._build_self_pairs()

    def _build_self_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs: list[tuple[int, int]] = []
        for first, first_name in enumerate(self.geometry_names):
            first_finger = _finger(first_name)
            for second in range(first + 1, len(self.geometry_names)):
                second_name = self.geometry_names[second]
                second_finger = _finger(second_name)
                if first_finger is not None and first_finger == second_finger:
                    continue  # linked meshes in one finger are adjacent by design.
                if first_name == "hand_base_link" and second_name.endswith("metacarpals"):
                    continue
                if second_name == "hand_base_link" and first_name.endswith("metacarpals"):
                    continue
                pairs.append((first, second))
        return tuple(pairs)

    def evaluate(self, qpos: np.ndarray) -> O30CollisionResult:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"O30 qpos must be finite with shape ({self.model.nq},)")
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateGeometryPlacements(
            self.model, self.data, self.geometry_model, self.geometry_data
        )

        request = hppfcl.CollisionRequest()
        self_pairs: list[str] = []
        for first, second in self.self_pairs:
            result = hppfcl.CollisionResult()
            hppfcl.collide(
                self.geometry_model.geometryObjects[first].geometry,
                self.geometry_data.oMg[first],
                self.geometry_model.geometryObjects[second].geometry,
                self.geometry_data.oMg[second],
                request,
                result,
            )
            if result.isCollision():
                self_pairs.append(f"{self.geometry_names[first]}:{self.geometry_names[second]}")

        distances: dict[str, float] = {}
        tip_distances: dict[str, float] = {}
        forbidden: list[str] = []
        for index, name in enumerate(self.geometry_names):
            placement = self.geometry_data.oMg[index]
            vertices = self.local_vertices[index]
            world_vertices = vertices @ placement.rotation.T + placement.translation[None]
            min_distance = float(np.min(self.object_tree.query(world_vertices, k=1)[0]))
            distances[name] = min_distance
            if name.endswith("_distal"):
                tip_distances[name] = min_distance
            elif min_distance < self.forbidden_clearance_m:
                forbidden.append(name)
        return O30CollisionResult(
            self_collision_pairs=tuple(self_pairs),
            forbidden_object_links=tuple(forbidden),
            link_object_min_distance_m=distances,
            fingertip_object_distance_m=tip_distances,
            object_surface_min_distance_m=float(min(distances.values())),
        )

    def safe_closure(
        self, target_qpos: np.ndarray, *, steps: int = 16
    ) -> tuple[float, O30CollisionResult]:
        """Find the furthest safe linear close fraction from O30's open posture."""
        if steps < 2:
            raise ValueError("steps must be at least 2")
        target = np.asarray(target_qpos, dtype=np.float64)
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        target = np.clip(target, lower, upper)
        # Vector retargeting uses zero qpos as O30 neutral.  The all-lower
        # posture self-collides, so it is not a valid closure starting pose.
        open_qpos = np.clip(np.zeros_like(target), lower, upper)
        safe_fraction = 0.0
        safe_result = self.evaluate(open_qpos)
        for fraction in np.linspace(0.0, 1.0, steps):
            result = self.evaluate(open_qpos + fraction * (target - open_qpos))
            if not result.safe:
                break
            safe_fraction = float(fraction)
            safe_result = result
        return safe_fraction, safe_result


__all__ = ["O30CollisionEvaluator", "O30CollisionResult"]
