"""Strict O30 mesh checks against a metric reconstructed object mesh.

The existing point-cloud evaluator is useful for early candidate rejection.
This module is the final static gate: it uses all O30 collision triangles for
collision tests and dense full-STL surface samples for clearance/contact
classification.  It deliberately fails closed when geometry is unsafe.
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

from .mesh_reduction import reduce_mesh

from .o30_scale import O30ScaleProfile, retargeter_from_profile


ROOT = Path(__file__).resolve().parents[1]
O30_URDF = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _link_name(name: str) -> str:
    return name.removesuffix("_0")


def _finger(name: str) -> str | None:
    return next((finger for finger in FINGERS if name.startswith(finger + "_")), None)


def _distal_pad_mask(name: str, vertices: np.ndarray) -> np.ndarray:
    """Return a local distal-pad sample mask using O30 mesh axes.

    The thumb is built along local +Y, while the four long fingers are built
    along local +Z.  A common +Z quantile falsely labels the thumb sidewall
    as its pad and breaks both contact fitting and collision classification.
    """
    finger = _finger(name)
    longitudinal_axis, normal_axis = (1, 2) if finger == "thumb" else (2, 0)
    longitudinal = vertices[:, longitudinal_axis]
    normal = vertices[:, normal_axis]
    mask = (longitudinal >= np.quantile(longitudinal, 0.64)) & (normal >= np.quantile(normal, 0.55))
    if int(mask.sum()) < 16:
        long_span = max(float(np.ptp(longitudinal)), 1e-9)
        normal_span = max(float(np.ptp(normal)), 1e-9)
        score = (longitudinal - longitudinal.min()) / long_span + 0.35 * (normal - normal.min()) / normal_span
        mask[np.argsort(score)[-min(16, len(score)):]] = True
    return mask


def _mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Invalid triangle mesh: {path}")
    return loaded


def _bvh(mesh: trimesh.Trimesh) -> hppfcl.BVHModelOBBRSS:
    model = hppfcl.BVHModelOBBRSS()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    for face in np.asarray(mesh.faces, dtype=np.int64):
        model.addTriangle(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]])
    model.endModel()
    return model


@dataclass(frozen=True)
class O30MeshValidation:
    self_collision_pairs: tuple[str, ...]
    forbidden_mesh_collisions: tuple[str, ...]
    nonpad_clearance_violations: tuple[str, ...]
    fingertip_distances_m: dict[str, float]
    link_clearances_m: dict[str, float]
    contact_fingers: tuple[str, ...]
    minimum_clearance_m: float
    max_pad_penetration_m: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "self_collision_pairs": list(self.self_collision_pairs),
            "forbidden_mesh_collisions": list(self.forbidden_mesh_collisions),
            "nonpad_clearance_violations": list(self.nonpad_clearance_violations),
            "fingertip_distances_m": self.fingertip_distances_m,
            "link_clearances_m": self.link_clearances_m,
            "contact_fingers": list(self.contact_fingers),
            "minimum_clearance_m": self.minimum_clearance_m,
            "max_pad_penetration_m": self.max_pad_penetration_m,
            "passed": self.passed,
        }


class O30MeshCollisionEvaluator:
    """Validate a 20-axis O30 pose in the metric hand frame."""

    def __init__(
        self,
        object_mesh_camera: Path,
        *,
        wrist_position_camera: np.ndarray,
        canonical_basis_row: np.ndarray,
        scale_profile: O30ScaleProfile | None = None,
        forbidden_clearance_m: float = 0.001,
        contact_distance_m: float = 0.0015,
        max_pad_penetration_m: float = 0.0003,
        vertices_per_link: int = 4096,
        object_mesh_is_hand_frame: bool = False,
        max_object_faces: int = 8000,
    ) -> None:
        if min(forbidden_clearance_m, contact_distance_m, max_pad_penetration_m) <= 0:
            raise ValueError("O30 mesh collision tolerances must be positive")
        if vertices_per_link < 128:
            raise ValueError("vertices_per_link must be at least 128")
        wrist = np.asarray(wrist_position_camera, dtype=np.float64)
        basis = np.asarray(canonical_basis_row, dtype=np.float64)
        if wrist.shape != (3,) or basis.shape != (3, 3) or not np.isfinite(wrist).all() or not np.isfinite(basis).all():
            raise ValueError("Invalid candidate camera-to-hand transform")
        retargeter = retargeter_from_profile(scale_profile)
        rotation_cfg = retargeter.rotation_xyz or {}
        rotation = Rotation.from_euler(
            "xyz", [rotation_cfg.get(axis, 0.0) for axis in ("x", "y", "z")], degrees=True
        ).as_matrix()
        source = _mesh(object_mesh_camera).copy()
        if not object_mesh_is_hand_frame:
            source.vertices = (np.asarray(source.vertices, dtype=np.float64) - wrist[None]) @ basis @ rotation.T
        self.object_mesh = source
        self.collision_mesh = reduce_mesh(source, max_object_faces)
        self.object_bvh = _bvh(self.collision_mesh)
        object_vertices = np.asarray(source.vertices, dtype=np.float64)
        if len(object_vertices) > 100000:
            sample_indices = np.linspace(0, len(object_vertices) - 1, 100000, dtype=np.int64)
            object_vertices = object_vertices[sample_indices]
        self.object_tree = cKDTree(object_vertices)
        self.object_bounds_min = np.asarray(source.vertices, dtype=np.float64).min(axis=0)
        self.object_bounds_max = np.asarray(source.vertices, dtype=np.float64).max(axis=0)
        self.forbidden_clearance_m = float(forbidden_clearance_m)
        self.contact_distance_m = float(contact_distance_m)
        self.max_pad_penetration_m = float(max_pad_penetration_m)

        self.model = pin.buildModelFromUrdf(str(O30_URDF))
        self.geometry_model = pin.buildGeomFromUrdf(
            self.model, str(O30_URDF), pin.GeometryType.COLLISION, [str(O30_URDF.parent)]
        )
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self.names = [_link_name(item.name) for item in self.geometry_model.geometryObjects]
        self.local_vertices: list[np.ndarray] = []
        self.pad_masks: list[np.ndarray] = []
        self.collision_bvhs: list[hppfcl.BVHModelOBBRSS] = []
        for item, name in zip(self.geometry_model.geometryObjects, self.names, strict=True):
            link_mesh = _mesh(item.meshPath)
            self.collision_bvhs.append(_bvh(reduce_mesh(link_mesh, 750)))
            vertices = np.asarray(link_mesh.vertices, dtype=np.float64)
            if len(vertices) > vertices_per_link:
                indices = np.linspace(0, len(vertices) - 1, vertices_per_link, dtype=np.int64)
                vertices = vertices[indices]
            self.local_vertices.append(vertices)
            if name.endswith("_distal"):
                self.pad_masks.append(_distal_pad_mask(name, vertices))
            else:
                self.pad_masks.append(np.zeros(len(vertices), dtype=bool))
        self.self_pairs = self._self_pairs()

    def _self_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs: list[tuple[int, int]] = []
        for first, first_name in enumerate(self.names):
            for second in range(first + 1, len(self.names)):
                second_name = self.names[second]
                first_finger, second_finger = _finger(first_name), _finger(second_name)
                if first_finger is not None and first_finger == second_finger:
                    continue
                if {first_name, second_name} == {"hand_base_link", "thumb_metacarpals"}:
                    continue
                pairs.append((first, second))
        return tuple(pairs)

    @staticmethod
    def _collision_depth(first: object, first_pose: object, second: object, second_pose: object) -> float:
        request = hppfcl.CollisionRequest()
        request.enable_contact = True
        request.num_max_contacts = 16
        result = hppfcl.CollisionResult()
        hppfcl.collide(first, first_pose, second, second_pose, request, result)
        if not result.isCollision() or result.numContacts() == 0:
            return 0.0
        return float(max(result.getContact(index).penetration_depth for index in range(result.numContacts())))



    @staticmethod
    def _bounds_overlap(
        first_min: np.ndarray, first_max: np.ndarray, second_min: np.ndarray, second_max: np.ndarray, margin: float = 0.0
    ) -> bool:
        return bool(np.all(first_max + margin >= second_min) and np.all(second_max + margin >= first_min))


    @staticmethod
    def _bounds_distance(
        first_min: np.ndarray, first_max: np.ndarray, second_min: np.ndarray, second_max: np.ndarray
    ) -> float:
        gap = np.maximum(np.maximum(first_min - second_max, second_min - first_max), 0.0)
        return float(np.linalg.norm(gap))

    def evaluate(self, qpos: np.ndarray, *, exact_collision: bool = True) -> O30MeshValidation:
        q = np.asarray(qpos, dtype=np.float64)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(f"O30 qpos must be finite with shape ({self.model.nq},)")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.geometry_data)
        world_vertices = [
            vertices @ placement.rotation.T + placement.translation[None]
            for vertices, placement in zip(self.local_vertices, self.geometry_data.oMg, strict=True)
        ]
        world_trees = [cKDTree(vertices) for vertices in world_vertices]
        world_bounds = [(vertices.min(axis=0), vertices.max(axis=0)) for vertices in world_vertices]
        self_pairs: list[str] = []
        for first, second in self.self_pairs:
            if exact_collision:
                first_min, first_max = world_bounds[first]
                second_min, second_max = world_bounds[second]
                collided = self._bounds_overlap(first_min, first_max, second_min, second_max) and self._collision_depth(
                    self.collision_bvhs[first], self.geometry_data.oMg[first],
                    self.collision_bvhs[second], self.geometry_data.oMg[second],
                ) > 0.0
            else:
                # Conservatively prevents a closure path from passing through another finger.
                first_min, first_max = world_bounds[first]
                second_min, second_max = world_bounds[second]
                collided = self._bounds_overlap(
                    first_min, first_max, second_min, second_max, self.forbidden_clearance_m
                ) and bool(
                    world_trees[second].query(world_vertices[first], k=1)[0].min()
                    < self.forbidden_clearance_m
                )
            if collided:
                self_pairs.append(f"{self.names[first]}:{self.names[second]}")
        forbidden, nonpad = [], []
        clearances: dict[str, float] = {}
        tips: dict[str, float] = {}
        contacts: list[str] = []
        for index, name in enumerate(self.names):
            placement = self.geometry_data.oMg[index]
            vertices = world_vertices[index]
            link_min, link_max = world_bounds[index]
            near_object = self._bounds_overlap(
                link_min, link_max, self.object_bounds_min, self.object_bounds_max, self.contact_distance_m
            )
            if near_object:
                distances = np.asarray(self.object_tree.query(vertices, k=1)[0], dtype=np.float64)
            else:
                distances = np.full(
                    len(vertices),
                    self._bounds_distance(link_min, link_max, self.object_bounds_min, self.object_bounds_max),
                    dtype=np.float64,
                )
            minimum = float(distances.min(initial=np.inf))
            clearances[name] = minimum
            collision_depth = (
                self._collision_depth(
                    self.collision_bvhs[index], placement,
                    self.object_bvh, hppfcl.Transform3f(),
                ) if exact_collision and self._bounds_overlap(
                    world_bounds[index][0], world_bounds[index][1], self.object_bounds_min, self.object_bounds_max
                ) else 0.0
            )
            pad = self.pad_masks[index]
            if name.endswith("_distal"):
                pad_distance = float(distances[pad].min(initial=np.inf))
                tips[name] = pad_distance
                if pad_distance <= self.contact_distance_m:
                    contacts.append(name.removesuffix("_distal"))
                if np.any(~pad) and float(distances[~pad].min()) < self.forbidden_clearance_m:
                    nonpad.append(name)
                # A close non-pad sample is a warning.  It becomes a hard failure
                # only when the exact triangle check also reports penetration.
                if collision_depth > self.max_pad_penetration_m or (collision_depth > 0.0 and name in nonpad):
                    forbidden.append(name)
                # Full-triangle contact is allowed only at the calibrated pad and
                # only up to the explicitly bounded numerical penetration.
            elif collision_depth > 0.0:
                forbidden.append(name)
        min_clearance = float(min(clearances.values(), default=np.inf))
        passed = (
            not self_pairs and not forbidden and len(set(contacts)) >= 3 and "thumb" in contacts
        )
        return O30MeshValidation(
            self_collision_pairs=tuple(self_pairs),
            forbidden_mesh_collisions=tuple(forbidden),
            nonpad_clearance_violations=tuple(nonpad),
            fingertip_distances_m=tips,
            link_clearances_m=clearances,
            contact_fingers=tuple(sorted(set(contacts))),
            minimum_clearance_m=min_clearance,
            max_pad_penetration_m=max((self._collision_depth(
                self.collision_bvhs[index], self.geometry_data.oMg[index],
                self.object_bvh, hppfcl.Transform3f()
            ) for index, name in enumerate(self.names) if name.endswith("_distal")), default=0.0),
            passed=passed,
        )

    def validate_closure(self, target_qpos: np.ndarray, *, steps: int = 32) -> tuple[bool, list[O30MeshValidation]]:
        if steps < 2:
            raise ValueError("O30 closure validation needs at least two steps")
        target = np.asarray(target_qpos, dtype=np.float64)
        open_q = np.clip(np.zeros_like(target), self.model.lowerPositionLimit, self.model.upperPositionLimit)
        fractions = np.linspace(0.0, 1.0, steps)
        reports = [
            self.evaluate(open_q + fraction * (target - open_q), exact_collision=True)
            for index, fraction in enumerate(fractions)
        ]
        # Every closure frame uses exact triangle collision.  The fast proximity
        # proxy is useful during optimization but must not reject a hardware-gated
        # trajectory because adjacent fingers merely pass near one another.
        # Intermediate positions need only be collision free; final position also
        # needs the required thumb-plus-three-finger contact pattern.
        intermediate_safe = all(
            not item.self_collision_pairs and not item.forbidden_mesh_collisions
            for item in reports[:-1]
        )
        return bool(intermediate_safe and reports[-1].passed), reports


__all__ = ["O30MeshCollisionEvaluator", "O30MeshValidation"]
