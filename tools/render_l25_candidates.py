#!/usr/bin/env python3
"""Render L25 hand + object mesh from HUG candidates (no benchmark needed).

Computes the camera-to-L25 similarity transform from canonical_grasp.npz via
the same retargeter used during candidate generation, transforms the object
mesh into the L25 simulation frame, and renders each candidate as a PNG.

Usage:
    .venv/bin/python tools/render_l25_candidates.py \
      --candidates-dir outputs/perception_bridge/run_motor/hug_candidates_50 \
      --object-mesh outputs/perception_bridge/run_motor/object_mesh_camera.ply \
      --output-dir outputs/perception_bridge/run_motor/hand_renders
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_XML = ROOT / "assets" / "linkerhand_l25" / "linkerhand_l25_right_mujoco.xml"
VECTOR_CONFIG = ROOT / "example" / "config" / "vector" / "mediapipe" / "mediapipe_linkerhand_l25.yaml"


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation for target = scale * source @ R.T + t."""
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


def _transform_mesh(
    mesh: trimesh.Trimesh, scale: float, rotation: np.ndarray, translation: np.ndarray,
) -> trimesh.Trimesh:
    verts = scale * np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation[None]
    # MuJoCo places hand_base_link at z=0.05 m
    verts[:, 2] += 0.05
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def _build_scene_xml(base_xml: Path, object_stl: Path) -> str:
    tree = ET.parse(base_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((base_xml.parent / "right").resolve()))
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


def _set_qpos(model: mujoco.MjModel, data: mujoco.MjData, names: list[str], qpos: np.ndarray) -> None:
    joint_index = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
        for i in range(model.njnt)
    }
    for name, value in zip(names, qpos):
        key = name.lower()
        if key in joint_index:
            data.qpos[model.jnt_qposadr[joint_index[key]]] = value


def render_candidate(
    scene_xml: str,
    trajectory_npz: Path,
    output_png: Path,
    width: int,
    height: int,
    frame: int,
) -> None:
    with np.load(trajectory_npz, allow_pickle=False) as npz:
        robot_qpos = np.asarray(npz["robot_qpos"], dtype=np.float64)
        joint_names = [str(n) for n in npz["robot_joint_names"]]

    idx = frame if frame >= 0 else len(robot_qpos) - 1
    idx = min(idx, len(robot_qpos) - 1)
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    _set_qpos(model, data, joint_names, robot_qpos[idx])
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=-1)
    image = renderer.render()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image)).save(str(output_png))


def _compute_similarity_for_candidate(
    canonical_npz: Path, retargeter=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity from HUG fingertips (camera) to L25 FK fingertips (sim frame).

    Using fingertips instead of all 21 keypoints avoids the mismatch between
    MANO bone lengths and L25 link lengths, which otherwise places the object
    in the palm instead of at the fingertips.
    """
    from anydexretarget.hand_representation import load_canonical_grasp_state

    if retargeter is None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, default=None,
                        help="Object mesh in camera frame (e.g. object_mesh_camera.ply). "
                             "Omit to render hand only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame", type=int, default=-1, help="Frame index; -1 means last frame.")
    parser.add_argument("--tile-columns", type=int, default=10)
    args = parser.parse_args()

    candidates = sorted(args.candidates_dir.glob("candidate_*/trajectory.npz"))
    if not candidates:
        raise FileNotFoundError(f"No candidate_*/trajectory.npz found in {args.candidates_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_mesh = None
    if args.object_mesh is not None and args.object_mesh.is_file():
        source_mesh = trimesh.load_mesh(args.object_mesh, process=False)
        if isinstance(source_mesh, trimesh.Scene):
            parts = [g for g in source_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            source_mesh = trimesh.util.concatenate(parts)
        print(f"Object mesh loaded: {len(source_mesh.vertices)} verts, {len(source_mesh.faces)} faces")

    retargeter = None
    if source_mesh is not None:
        from anydexretarget.retarget import Retargeter
        retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")

    tmp_dir = Path(tempfile.mkdtemp(prefix="l25_render_"))
    pngs: list[Path] = []
    for i, npz_path in enumerate(candidates):
        name = npz_path.parent.name
        png = args.output_dir / f"{name}.png"
        canonical_npz = npz_path.parent / "canonical_grasp.npz"

        if source_mesh is not None and canonical_npz.is_file():
            scale, rotation, translation = _compute_similarity_for_candidate(
                canonical_npz, retargeter=retargeter,
            )
            transformed = _transform_mesh(source_mesh, scale, rotation, translation)
            stl_path = tmp_dir / f"{name}_object.stl"
            transformed.export(stl_path)
            scene_xml = _build_scene_xml(MODEL_XML, stl_path)
        else:
            scene_xml = str(MODEL_XML)

        render_candidate(scene_xml, npz_path, png, args.width, args.height, args.frame)

        if scene_xml != str(MODEL_XML):
            os.unlink(scene_xml)

        pngs.append(png)
        print(f"[{i + 1}/{len(candidates)}] {name} -> {png}")

    # Build contact sheet with labels
    if pngs:
        images = [Image.open(p) for p in pngs]
        cols = min(args.tile_columns, len(images))
        rows = (len(images) + cols - 1) // cols
        w, h = images[0].size
        label_h = 20
        pad = 4
        sheet = Image.new(
            "RGB",
            (cols * (w + pad) + pad, rows * (h + label_h + pad) + pad),
            (26, 26, 26),
        )
        draw = ImageDraw.Draw(sheet)
        for idx, (img, png) in enumerate(zip(images, pngs)):
            r, c = divmod(idx, cols)
            x = c * (w + pad) + pad
            y = r * (h + label_h + pad) + pad
            sheet.paste(img, (x, y))
            label = png.stem
            draw.text((x + 4, y + h + 2), label, fill=(200, 200, 200))
        sheet_path = args.output_dir / "contact_sheet.png"
        sheet.save(str(sheet_path))
        print(f"Contact sheet ({cols}x{rows}): {sheet_path}")

    # Cleanup temp STLs
    for f in tmp_dir.glob("*.stl"):
        f.unlink()
    tmp_dir.rmdir()

    print(f"Done: {len(pngs)} candidates rendered to {args.output_dir}")


if __name__ == "__main__":
    main()
