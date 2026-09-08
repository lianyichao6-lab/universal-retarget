#!/usr/bin/env python3
"""Build an O30 MuJoCo scene containing the actual reconstructed object mesh."""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
O30_URDF = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(item for item in mesh.geometry.values() if isinstance(item, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"No valid triangle mesh: {path}")
    return mesh


def _reduce_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_faces:
        return mesh
    vertices, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    lower, extent = vertices.min(axis=0), vertices.max(axis=0) - vertices.min(axis=0)
    positive = extent[extent > 1e-9]
    if len(positive) == 0:
        raise ValueError("Cannot simplify zero-size mesh")
    voxel = max(float(np.prod(positive) / max(1000, max_faces // 2)) ** (1.0 / 3.0), float(positive.min()) / 10000.0)
    for _ in range(20):
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
    raise RuntimeError("Unable to simplify object mesh for MuJoCo")


def _build_urdf(mesh_path: Path, output: Path) -> None:
    tree = ET.parse(O30_URDF)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((O30_URDF.parent / filename).resolve()))
    link = ET.Element("link", {"name": "reconstructed_object"})
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": str(mesh_path.resolve())})
        if tag == "visual":
            material = ET.SubElement(element, "material", {"name": "reconstructed_object_material"})
            ET.SubElement(material, "color", {"rgba": "0.10 0.72 0.92 1"})
    root.append(link)
    joint = ET.Element("joint", {"name": "reconstructed_object_fixed", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "hand_base_link"})
    ET.SubElement(joint, "child", {"link": "reconstructed_object"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    root.append(joint)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _set_q(model: mujoco.MjModel, data: mujoco.MjData, values: np.ndarray, names: list[str]) -> None:
    by_name = {str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)).lower(): index for index in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)}
    data.qpos[:] = 0.0
    for value, name in zip(values, names):
        joint = by_name.get(name.lower())
        if joint is None:
            raise ValueError(f"O30 scene lacks joint {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)


def _contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    object_geoms = {index for index in range(model.ngeom) if "reconstructed_object" in str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")}
    object_pairs, self_pairs = [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        name1 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first) or f"geom_{first}")
        name2 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second) or f"geom_{second}")
        if first in object_geoms or second in object_geoms:
            object_pairs.append({"hand_geom": name2 if first in object_geoms else name1, "distance_m": float(contact.dist), "position_m": np.asarray(contact.pos).tolist()})
            continue
        finger1 = next((name for name in FINGERS if name1.startswith(name + "_")), None)
        finger2 = next((name for name in FINGERS if name2.startswith(name + "_")), None)
        if finger1 and finger2 and finger1 != finger2:
            self_pairs.append({"geom1": name1, "geom2": name2, "distance_m": float(contact.dist), "position_m": np.asarray(contact.pos).tolist()})
    return object_pairs, self_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-mesh-faces", type=int, default=180000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_mesh_faces < 200000:
        raise ValueError("--max-mesh-faces must be between 1 and 199999")
    with np.load(args.plan, allow_pickle=False) as data:
        plan = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {"camera_to_o30_rotation", "camera_to_o30_translation", "human_to_o30_uniform_scale", "qpos_vector_order", "vector_joint_names"}
    if missing := required - set(plan):
        raise ValueError("O30 plan missing: " + ", ".join(sorted(missing)))
    mesh = _load_mesh(args.object_mesh).copy()
    scale, rotation, translation = float(plan["human_to_o30_uniform_scale"].item()), np.asarray(plan["camera_to_o30_rotation"], dtype=np.float64), np.asarray(plan["camera_to_o30_translation"], dtype=np.float64)
    mesh.vertices = scale * np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation[None]
    original_faces = len(mesh.faces)
    proxy = _reduce_mesh(mesh, args.max_mesh_faces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_stl = args.output_dir / "object_in_o30_simulation_frame.stl"
    proxy.export(object_stl)
    scene_urdf = args.output_dir / "o30_object_relative_scene.urdf"
    _build_urdf(object_stl, scene_urdf)
    model, data = mujoco.MjModel.from_xml_path(str(scene_urdf)), None
    data = mujoco.MjData(model)
    _set_q(model, data, np.asarray(plan["qpos_vector_order"], dtype=np.float64), [str(item) for item in plan["vector_joint_names"]])
    object_pairs, self_pairs = _contacts(model, data)
    report = {
        "simulation_only": True, "hardware_command_generated": False, "plan": str(args.plan.resolve()), "source_object_mesh": str(args.object_mesh.resolve()),
        "transformed_object_mesh": str(object_stl.resolve()), "scene_urdf": str(scene_urdf.resolve()), "object_mesh_faces_before_simplification": original_faces,
        "object_mesh_faces": len(proxy.faces), "object_contact_pair_count": len(object_pairs), "object_contacts": object_pairs,
        "self_contact_pair_count": len(self_pairs), "self_contacts": self_pairs,
        "note": "The reconstructed mesh is in the O30 hand simulation frame. MuJoCo contacts are simulation diagnostics, not force closure or hardware validation.",
    }
    report_path = args.output_dir / "scene_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("O30 + reconstructed object MuJoCo scene built (simulation only)")
    print(f"  object contacts: {len(object_pairs)}; cross-finger contacts: {len(self_pairs)}")
    print(f"  scene: {scene_urdf}")
    print(f"  report: {report_path}")
    if args.show:
        print("Opening MuJoCo viewer; close its window to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
