#!/usr/bin/env python3
"""Build and optionally view an L25 plus reconstructed-object MuJoCo scene.

The object transform comes from an object-relative L25 plan and is valid only
in the plan's AnyDex-derived simulation frame.  This tool never connects to a
robot or produces vendor commands.
"""

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

from anydexretarget.retarget import Retargeter


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
VECTOR_CONFIG = ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=False)
    if isinstance(raw, trimesh.Scene):
        meshes = [item for item in raw.geometry.values() if isinstance(item, trimesh.Trimesh)]
        raw = trimesh.util.concatenate(meshes)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"No valid triangle mesh: {path}")
    return raw


def _make_mujoco_mesh(mesh: trimesh.Trimesh, max_faces: int) -> tuple[trimesh.Trimesh, int]:
    """Reduce only the MuJoCo mesh proxy below the STL face limit."""
    original_faces = len(mesh.faces)
    if original_faces <= max_faces:
        return mesh, original_faces

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lower = vertices.min(axis=0)
    extent = vertices.max(axis=0) - lower
    nonzero_extent = extent[extent > 1e-9]
    if len(nonzero_extent) == 0:
        raise ValueError("Cannot simplify a zero-size object mesh")

    # Vertex clustering retains a connected triangle surface without adding a
    # dependency to the HUG environment.  The full mesh remains used by HUG.
    target_vertices = max(1_000, max_faces // 2)
    voxel = max(float(np.prod(nonzero_extent) / target_vertices) ** (1.0 / 3.0), float(nonzero_extent.min()) / 10_000.0)
    for _ in range(20):
        cells = np.floor((vertices - lower) / voxel).astype(np.int64)
        _keys, inverse = np.unique(cells, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        reduced_vertices = np.column_stack([
            np.bincount(inverse, weights=vertices[:, axis]) / counts
            for axis in range(3)
        ])
        reduced_faces = inverse[faces]
        keep = ((reduced_faces[:, 0] != reduced_faces[:, 1])
                & (reduced_faces[:, 1] != reduced_faces[:, 2])
                & (reduced_faces[:, 0] != reduced_faces[:, 2]))
        reduced_faces = reduced_faces[keep]
        canonical_faces = np.sort(reduced_faces, axis=1)
        _unique, indices = np.unique(canonical_faces, axis=0, return_index=True)
        reduced_faces = reduced_faces[np.sort(indices)]
        if 0 < len(reduced_faces) <= max_faces:
            return trimesh.Trimesh(vertices=reduced_vertices, faces=reduced_faces, process=False), original_faces
        voxel *= 1.35
    raise RuntimeError(f"Unable to reduce object mesh below {max_faces} faces; final face count={len(reduced_faces)}")


def _source_mesh(plan_path: Path) -> Path:
    with np.load(plan_path, allow_pickle=False) as data:
        contact_path = Path(str(data["source_contact_plan"].item()))
    with np.load(contact_path, allow_pickle=False) as contact:
        mesh_path = Path(str(contact["source_object_mesh"].item()))
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    return mesh_path


def _transform_mesh(plan_path: Path, source: Path, output: Path, max_faces: int) -> tuple[trimesh.Trimesh, np.ndarray, int]:
    with np.load(plan_path, allow_pickle=False) as data:
        scale = float(np.asarray(data["human_to_l25_uniform_scale"]).item())
        rotation = np.asarray(data["camera_to_l25_rotation"], dtype=np.float64)
        translation = np.asarray(data["camera_to_l25_translation"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 0:
        raise ValueError("Invalid camera-to-L25 transform in plan")
    mesh = _load_mesh(source).copy()
    # MuJoCo places hand_base_link at z=0.05 m; Pinocchio FK uses root at zero.
    translation = translation + np.array((0.0, 0.0, 0.05), dtype=np.float64)
    mesh.vertices = scale * np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation[None]
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Transformed object mesh has NaN/Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco_mesh, original_faces = _make_mujoco_mesh(mesh, max_faces)
    mujoco_mesh.export(output)
    return mujoco_mesh, np.concatenate((rotation, translation[:, None]), axis=1), original_faces


def _build_xml(base: Path, mesh_path: Path, output: Path) -> None:
    tree = ET.parse(base)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("L25 MuJoCo model lacks <compiler>")
    compiler.set("meshdir", str((base.parent / "right").resolve()))
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("L25 MuJoCo model lacks <asset> or <worldbody>")
    asset.append(ET.Element("mesh", {"name": "reconstructed_object", "file": str(mesh_path.resolve())}))
    body = ET.Element("body", {"name": "reconstructed_object", "pos": "0 0 0"})
    body.append(ET.Element("geom", {
        "name": "reconstructed_object_geom", "type": "mesh", "mesh": "reconstructed_object",
        "rgba": "0.12 0.82 0.78 0.45", "contype": "1", "conaffinity": "1",
    }))
    worldbody.append(body)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)


def _contact_report(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reconstructed_object_geom")
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if object_id not in (contact.geom1, contact.geom2):
            continue
        other_id = contact.geom2 if contact.geom1 == object_id else contact.geom1
        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id) or f"geom_{other_id}"
        contacts.append({
            "hand_geom": str(other_name),
            "distance_m": float(contact.dist),
            "position_m": np.asarray(contact.pos).tolist(),
        })
    return contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-mesh-faces", type=int, default=180_000, help="Maximum faces in the MuJoCo STL proxy; must be below 200000.")
    parser.add_argument("--show", action="store_true", help="Open the MuJoCo viewer after static contact evaluation.")
    args = parser.parse_args()
    if not args.plan.is_file():
        raise FileNotFoundError(args.plan)
    if not 0 < args.max_mesh_faces < 200_000:
        raise ValueError("--max-mesh-faces must be between 1 and 199999")
    source_mesh = _source_mesh(args.plan)
    mesh_l25, transform, original_faces = _transform_mesh(
        args.plan, source_mesh, args.output_dir / "object_in_l25_simulation_frame.stl", args.max_mesh_faces
    )
    scene_xml = args.output_dir / "l25_object_relative_scene.xml"
    _build_xml(BASE_MODEL, args.output_dir / "object_in_l25_simulation_frame.stl", scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    with np.load(args.plan, allow_pickle=False) as data_file:
        qpos = np.asarray(data_file["qpos"], dtype=np.float64)
        names = [str(value) for value in data_file["robot_joint_names"]]
        target_key = "contact_target_positions_l25" if "contact_target_positions_l25" in data_file.files else "desired_fingertip_positions_l25"
        target_tips = np.asarray(data_file[target_key], dtype=np.float64)
        qpos_vector = np.asarray(data_file["qpos_vector_order"], dtype=np.float64)
        vector_joint_names = [str(value) for value in data_file["vector_joint_names"]]
        actual_tips = np.asarray(data_file["l25_fingertip_positions_optimized"], dtype=np.float64)
        active = np.asarray(data_file["active_contact_mask"], dtype=np.uint8).astype(bool)
    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    index = {str(name).lower(): i for i, name in enumerate(model_joint_names) if name is not None}
    if qpos.shape != (len(names),) or any(name.lower() not in index for name in names):
        raise ValueError("Plan qpos is incompatible with generated L25 scene")
    for value, name in zip(qpos, names):
        data.qpos[model.jnt_qposadr[index[name.lower()]]] = value
    mujoco.mj_forward(model, data)
    contacts = _contact_report(model, data)

    # Verify the exact five task-offset points used by Pinocchio against MuJoCo.
    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    robot = retargeter.optimizer.robot
    task_names = [str(name) for name in retargeter.optimizer.task_link_names]
    task_ids = [robot.get_link_index(name) for name in task_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    expected_vector_names = [str(name) for name in robot.dof_joint_names]
    if vector_joint_names != expected_vector_names:
        raise ValueError("Plan vector joint order disagrees with the current L25 Vector config")
    pinocchio_tips = robot.compute_points_batch(qpos_vector, task_ids, task_offsets)
    mujoco_tips = []
    for task_name, offset in zip(task_names, task_offsets):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model lacks task body: {task_name}")
        mujoco_tips.append(data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset)
    mujoco_tips = np.asarray(mujoco_tips, dtype=np.float64)
    root_offset = np.asarray((0.0, 0.0, 0.05), dtype=np.float64)
    fk_delta_mm = np.linalg.norm(pinocchio_tips - (mujoco_tips - root_offset), axis=1) * 1000.0
    report = {
        "simulation_only": True,
        "plan": str(args.plan.resolve()),
        "source_object_mesh": str(source_mesh.resolve()),
        "transformed_object_mesh": str((args.output_dir / "object_in_l25_simulation_frame.stl").resolve()),
        "scene_xml": str(scene_xml.resolve()),
        "object_mesh_vertices": int(len(mesh_l25.vertices)),
        "object_mesh_faces": int(len(mesh_l25.faces)),
        "object_mesh_faces_before_mujoco_simplification": int(original_faces),
        "contact_pair_count": len(contacts),
        "contacts": contacts,
        "active_contact_target_error_mm": (np.linalg.norm(actual_tips[active] - target_tips[active], axis=1) * 1000.0).tolist(),
        "pinocchio_mujoco_task_point_delta_mm": fk_delta_mm.tolist(),
        "pinocchio_mujoco_task_point_delta_max_mm": float(fk_delta_mm.max()),
        "collision_interpretation": "MuJoCo mesh contacts are a simulation diagnostic only. They do not prove force closure or validate the reconstructed hidden geometry.",
        "hardware_command_generated": False,
    }
    report_path = args.output_dir / "scene_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("L25 + reconstructed object MuJoCo scene built (simulation only)")
    if original_faces != len(mesh_l25.faces):
        print(f"  MuJoCo mesh proxy simplified: {original_faces} -> {len(mesh_l25.faces)} faces")
    print(f"  static mesh contact pairs: {len(contacts)}")
    print("  active contact target errors [mm]: " + ", ".join(f"{v:.1f}" for v in report["active_contact_target_error_mm"]))
    print(f"  Pinocchio-MuJoCo task-point max delta: {report['pinocchio_mujoco_task_point_delta_max_mm']:.3f} mm")
    print(f"  XML: {scene_xml}")
    print(f"  report: {report_path}")
    if args.show:
        print("Opening MuJoCo viewer; close its window to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
