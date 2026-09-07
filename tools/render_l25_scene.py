#!/usr/bin/env python3
"""Render a built L25 MuJoCo scene (hand + object) to a PNG, offscreen.

The benchmark's ``build_l25_object_relative_scene.py`` writes an XML scene with
the L25 hand and the object mesh together in the simulation frame. This tool
loads that XML and renders it headlessly so the grasp can be inspected without
a display or RViz.

Usage:
    .venv/bin/python tools/render_l25_scene.py \
      --scene <scene>/l25_object_relative_scene.xml \
      --output grasp.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Headless rendering: avoid GLFW (needs a display) and use EGL/OSMesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


def _set_qpos_from_plan(model: mujoco.MjModel, data: mujoco.MjData, plan_path: Path) -> None:
    with np.load(plan_path, allow_pickle=False) as plan:
        qpos = np.asarray(plan["qpos"], dtype=np.float64)
        names = [str(n) for n in plan["robot_joint_names"]]
    joint_index = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is not None:
            joint_index[name.lower()] = i
    for name, value in zip(names, qpos):
        key = name.lower()
        if key in joint_index:
            data.qpos[model.jnt_qposadr[joint_index[key]]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None,
                        help="L25 plan .npz with qpos and robot_joint_names.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=0.45)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if args.plan is not None:
        _set_qpos_from_plan(model, data, args.plan)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    # Render with the free camera; the scene XML's <visual><global> sets azimuth/elevation.
    renderer.update_scene(data, camera=-1)
    image = renderer.render()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image)).save(str(args.output))
    print(f"Rendered {args.width}x{args.height} to {args.output}")


if __name__ == "__main__":
    main()
