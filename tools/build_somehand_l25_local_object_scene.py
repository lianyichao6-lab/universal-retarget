#!/usr/bin/env python3
"""Visualize an object-relative HUG grasp using SomeHand on the local L25 model."""

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

from somehand.api import RetargetingEngine


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/somehand_l25_local_right.yaml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation, target_center - rotation @ source_center


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"No triangle mesh: {path}")
    return mesh


def _build_scene(base_model: Path, object_mesh: Path, output: Path) -> None:
    tree = ET.parse(base_model)
    root = tree.getroot()
    compiler, asset, worldbody = root.find("compiler"), root.find("asset"), root.find("worldbody")
    if compiler is None or asset is None or worldbody is None:
        raise ValueError("Local L25 MJCF misses compiler, asset, or worldbody")
    meshdir = Path(compiler.get("meshdir", ""))
    if not meshdir.is_absolute():
        meshdir = (base_model.parent / meshdir).resolve()
    if not meshdir.exists():
        raise FileNotFoundError(f"Local L25 mesh directory does not exist: {meshdir}")
    compiler.set("meshdir", str(meshdir))
    asset.append(ET.Element("mesh", {"name": "hug_object", "file": str(object_mesh.resolve())}))
    object_body = ET.Element("body", {"name": "hug_object", "pos": "0 0 0"})
    object_body.append(ET.Element("geom", {
        "name": "hug_object_geom", "type": "mesh", "mesh": "hug_object",
        "rgba": "0.12 0.82 0.78 0.45", "contype": "1", "conaffinity": "1",
    }))
    worldbody.append(object_body)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hug_object_geom")
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if object_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == object_id else contact.geom1
        contacts.append({
            "hand_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}",
            "hand_link": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or "unknown",
            "distance_m": float(contact.dist),
        })
    return contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpos", type=Path, required=True, help="NPZ from retarget_canonical_somehand_l25_local.py")
    parser.add_argument("--contact-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    with np.load(args.qpos, allow_pickle=False) as source:
        if str(source["backend"].item()) != "somehand_local_l25" or str(source["robot"].item()) != "l25":
            raise ValueError("--qpos must be a somehand_local_l25 L25 output")
        qpos = np.asarray(source["robot_qpos"], dtype=np.float64)
        joint_names = [str(name) for name in source["robot_joint_names"]]
    with np.load(args.contact_plan, allow_pickle=False) as contact:
        human_tips = np.asarray(contact["fingertip_positions_camera"], dtype=np.float64)
        active = np.asarray(contact["near_surface"], dtype=np.uint8).astype(bool)
        mesh_path = Path(str(contact["source_object_mesh"].item()))
    if qpos.shape != (len(joint_names),) or human_tips.shape != (5, 3) or active.sum() < 3:
        raise ValueError("Invalid qpos or contact plan shape")

    engine = RetargetingEngine.from_config_path(str(CONFIG))
    if joint_names != engine.hand_model.get_joint_names():
        raise ValueError("Local L25 qpos joint order does not match the local L25 model")
    engine.hand_model.set_qpos(qpos)
    local_tips = np.asarray([engine.hand_model.get_site_position(f"{finger}_distal_tip") for finger in FINGERS])
    rotation, translation = _rigid(human_tips[active], local_tips[active])
    fitted_tips = human_tips @ rotation.T + translation[None]
    fit_error = np.linalg.norm(fitted_tips[active] - local_tips[active], axis=1)

    mesh = _load_mesh(mesh_path).copy()
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T + translation[None]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transformed_mesh = args.output_dir / "object_in_local_l25_frame.stl"
    mesh.export(transformed_mesh)
    scene_xml = args.output_dir / "somehand_local_l25_object_scene.xml"
    _build_scene(Path(engine.hand_model.mjcf_path), transformed_mesh, scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    current_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if current_names != joint_names:
        raise ValueError("Generated local L25 scene joint order differs from qpos source")
    violations = int(np.count_nonzero((qpos < model.jnt_range[:, 0]) | (qpos > model.jnt_range[:, 1])))
    if violations:
        raise ValueError(f"Local L25 qpos violates {violations} joint limits")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    contacts = _contacts(model, data)
    report = {
        "simulation_only": True,
        "hardware_command_generated": False,
        "backend": "somehand_local_l25",
        "qpos": str(args.qpos.resolve()),
        "contact_plan": str(args.contact_plan.resolve()),
        "source_object_mesh": str(mesh_path.resolve()),
        "scene_xml": str(scene_xml.resolve()),
        "rigid_fit_active_tip_error_mm": (fit_error * 1000.0).tolist(),
        "rigid_fit_active_tip_mean_error_mm": float(fit_error.mean() * 1000.0),
        "static_mesh_contact_pairs": len(contacts),
        "contacts": contacts,
        "joint_limit_violations": violations,
        "interpretation": "The local L25 model and qpos match. Mesh contacts diagnose intersection only; they do not establish force closure or physical stability.",
    }
    report_path = args.output_dir / "scene_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SomeHand local L25 + HUG object scene built (simulation only)")
    print(f"  active HUG-tip rigid-fit error: {report['rigid_fit_active_tip_mean_error_mm']:.2f} mm")
    print(f"  static mesh contact pairs: {len(contacts)}")
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
