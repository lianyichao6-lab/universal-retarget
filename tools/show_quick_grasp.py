#!/usr/bin/env python3
"""Visualize a quick_grasp result in the MuJoCo viewer.

Builds an L25 hand + object scene from quick_grasp output (trajectory.npz +
canonical_grasp.npz) and opens the interactive MuJoCo viewer, or renders a PNG.

Usage:
    # Interactive viewer
    DISPLAY=:0 .venv/bin/python tools/show_quick_grasp.py \
      --grasp-dir outputs/perception_bridge/run_motor_quick/quick_grasp \
      --object-mesh outputs/perception_bridge/run_motor_quick/object_mesh_camera.ply \
      --show

    # Render PNG (headless)
    .venv/bin/python tools/show_quick_grasp.py \
      --grasp-dir outputs/perception_bridge/run_motor_quick/quick_grasp \
      --object-mesh outputs/perception_bridge/run_motor_quick/object_mesh_camera.ply \
      --output grasp.png
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_XML = ROOT / "assets" / "linkerhand_l25" / "linkerhand_l25_right_mujoco.xml"
VECTOR_CONFIG = (
    ROOT / "example" / "config" / "vector" / "mediapipe"
    / "mediapipe_linkerhand_l25.yaml"
)


def _similarity(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Degenerate hand keypoints")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def _compute_similarity(canonical_npz: Path):
    from anydexretarget.hand_representation import load_canonical_grasp_state
    from anydexretarget.retarget import Retargeter

    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    state = load_canonical_grasp_state(canonical_npz)
    source_hand = state.keypoints_for_retargeting().astype(np.float64)
    qpos_vec, _ = retargeter.retarget_verbose(source_hand, apply_filter=False)

    robot = retargeter.optimizer.robot
    task_names = [str(n) for n in retargeter.optimizer.task_link_names]
    task_ids = [robot.get_link_index(n) for n in task_names]
    task_offsets = np.asarray(retargeter.optimizer.task_offsets, dtype=np.float64)
    l25_tips = robot.compute_points_batch(
        np.asarray(qpos_vec, dtype=np.float64), task_ids, task_offsets,
    )

    hug_tips = np.asarray(
        np.load(canonical_npz, allow_pickle=False)["fingertip_positions_camera"],
        dtype=np.float64,
    )
    return _similarity(hug_tips, l25_tips)


def _build_scene(object_stl: Path) -> str:
    tree = ET.parse(MODEL_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((MODEL_XML.parent / "right").resolve()))
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Model XML missing <asset> or <worldbody>")
    asset.append(ET.Element("mesh", {"name": "object", "file": str(object_stl.resolve())}))
    body = ET.Element("body", {"name": "object", "pos": "0 0 0"})
    body.append(ET.Element("geom", {
        "name": "object_geom", "type": "mesh", "mesh": "object",
        "rgba": "0.12 0.82 0.78 0.45", "contype": "0", "conaffinity": "0",
    }))
    worldbody.append(body)
    ET.indent(tree, space="  ")
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        tree.write(f, encoding="unicode", xml_declaration=False)
        return f.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-dir", type=Path, required=True,
                        help="quick_grasp output directory")
    parser.add_argument("--object-mesh", type=Path, default=None,
                        help="Object mesh in camera frame (.ply)")
    parser.add_argument("--frame", type=int, default=-1,
                        help="Trajectory frame; -1 = last")
    parser.add_argument("--output", type=Path, default=None,
                        help="Render to PNG (headless)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--show", action="store_true",
                        help="Open interactive MuJoCo viewer")
    args = parser.parse_args()

    if not args.show and args.output is None:
        parser.error("需要 --show (交互查看) 或 --output (渲染 PNG)")

    grasp_dir = args.grasp_dir
    traj_npz = grasp_dir / "trajectory.npz"
    canon_npz = grasp_dir / "canonical_grasp.npz"
    if not traj_npz.exists():
        raise FileNotFoundError(f"未找到 {traj_npz}")
    if not canon_npz.exists():
        raise FileNotFoundError(f"未找到 {canon_npz}")

    with np.load(traj_npz, allow_pickle=False) as npz:
        robot_qpos = np.asarray(npz["robot_qpos"], dtype=np.float64)
        joint_names = [str(n) for n in npz["robot_joint_names"]]

    idx = args.frame if args.frame >= 0 else len(robot_qpos) - 1
    idx = min(idx, len(robot_qpos) - 1)
    print(f"Frame {idx}/{len(robot_qpos)-1}")

    # Auto-detect object mesh from parent scene dir
    object_mesh_path = args.object_mesh
    if object_mesh_path is None:
        candidate = grasp_dir.parent / "object_mesh_camera.ply"
        if candidate.is_file():
            object_mesh_path = candidate
            print(f"自动检测物体网格: {object_mesh_path}")

    scene_xml = str(MODEL_XML)
    tmp_stl = None
    if object_mesh_path is not None and object_mesh_path.is_file():
        print("计算 camera→L25 相似变换...")
        scale, rotation, translation = _compute_similarity(canon_npz)
        print(f"  scale={scale:.4f}, |t|={np.linalg.norm(translation):.4f} m")

        mesh = trimesh.load_mesh(object_mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(parts)

        verts = scale * np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation[None]
        verts[:, 2] += 0.05
        transformed = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

        tmp_stl = Path(tempfile.mktemp(suffix=".stl", prefix="quick_grasp_obj_"))
        transformed.export(tmp_stl)
        scene_xml = _build_scene(tmp_stl)
        print(f"物体网格: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    if args.show:
        os.environ.pop("MUJOCO_GL", None)
        import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    joint_index = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
        for i in range(model.njnt)
    }
    for name, value in zip(joint_names, robot_qpos[idx]):
        key = name.lower()
        if key in joint_index:
            data.qpos[model.jnt_qposadr[joint_index[key]]] = value
    mujoco.mj_forward(model, data)

    if args.output is not None:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        renderer.update_scene(data, camera=-1)
        image = renderer.render()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(image)).save(str(args.output))
        print(f"渲染完成: {args.output}")

    if args.show:
        print("打开 MuJoCo 查看器; 关闭窗口退出.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    if tmp_stl is not None:
        tmp_stl.unlink(missing_ok=True)
    if scene_xml != str(MODEL_XML):
        Path(scene_xml).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
