#!/usr/bin/env python3
"""Rebase an L25 object-relative MuJoCo scene into an AR5 base frame."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from anydexretarget.luban_arm import homogeneous_transform


def rebase_scene(source: Path, calibration: Path, output_dir: Path) -> Path:
    """Preserve object-to-hand geometry while relocating the object mesh.

    The source scene stores object vertices in its world frame.  At frame zero
    its ``hand_base_link`` is the reference hand transform.  The output places
    those same vertices relative to the requested AR5/L25 hand target, so an
    FK-driven mocap hand has the same object-relative HUG grasp at approach.
    """
    with np.load(calibration, allow_pickle=False) as data:
        if not bool(np.asarray(data["simulation_only"]).item()):
            raise ValueError("only simulation-only calibrations may rebase a MuJoCo scene")
        target_hand = homogeneous_transform(
            data["T_robot_base_l25_hand_target"], "T_robot_base_l25_hand_target"
        )
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")
    if hand_body < 0:
        raise ValueError("source scene has no hand_base_link")
    source_hand = np.eye(4, dtype=np.float64)
    source_hand[:3, :3] = data.xmat[hand_body].reshape(3, 3)
    source_hand[:3, 3] = data.xpos[hand_body]

    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("source scene is missing asset")
    mesh = next((item for item in asset.findall("mesh") if item.get("name") == "reconstructed_object"), None)
    if mesh is None or not mesh.get("file"):
        raise ValueError("source scene is missing reconstructed_object mesh")
    mesh_path = Path(mesh.get("file"))
    if not mesh_path.is_absolute():
        mesh_path = (source.parent / mesh_path).resolve()
    raw = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(tuple(raw.geometry.values()))
    if not isinstance(raw, trimesh.Trimesh):
        raise ValueError("reconstructed object must be a triangle mesh")
    rebased = raw.copy()
    rebased.apply_transform(target_hand @ np.linalg.inv(source_hand))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh = output_dir / "reconstructed_object_ar5_base.stl"
    rebased.export(output_mesh)
    mesh.set("file", str(output_mesh.resolve()))
    output_scene = output_dir / "l25_object_relative_ar5_base.xml"
    ET.indent(tree, space="  ")
    tree.write(output_scene, encoding="utf-8", xml_declaration=False)
    return output_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = rebase_scene(args.scene, args.calibration, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
