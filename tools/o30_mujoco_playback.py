#!/usr/bin/env python3
"""Play an offline O30 trajectory directly in MuJoCo, standalone."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from anydexretarget.hand_contract import O30_QPOS_JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"


def _model_joint_names(model: mujoco.MjModel) -> dict[str, int]:
    names: dict[str, int] = {}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None:
            raise ValueError(f"MuJoCo joint {index} has no name")
        names[name] = index
    return names


def _load_frames(path: Path, model: mujoco.MjModel) -> np.ndarray:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    joint_ids = _model_joint_names(model)
    frames: list[np.ndarray] = []
    for record in records:
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError("each trajectory frame must contain target")
        target = np.asarray(record["target"], dtype=np.float64).reshape(-1)
        source_names = tuple(record.get("robot_joint_names", O30_QPOS_JOINT_NAMES))
        if target.shape != (20,) or tuple(source_names) != O30_QPOS_JOINT_NAMES:
            raise ValueError("trajectory is not an O30 physical-qpos trajectory")
        qpos = np.zeros(model.nq, dtype=np.float64)
        for source_index, name in enumerate(source_names):
            joint_id = joint_ids.get(name)
            if joint_id is None:
                raise ValueError(f"MuJoCo model lacks O30 joint {name}")
            qpos_address = model.jnt_qposadr[joint_id]
            qpos[qpos_address] = target[source_index]
        frames.append(qpos)
    return np.asarray(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--scene-xml", type=Path, help="Optional O30 object-relative URDF/XML built by build_o30_object_relative_scene.py.")
    parser.add_argument("--final-pose", action="store_true", help="Display only the final grasp pose; do not replay the close trajectory.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    model_path = args.scene_xml if args.scene_xml is not None else args.model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    frames = _load_frames(args.trajectory, model)
    if args.dry_run:
        print(json.dumps({
            "model": str(model_path),
            "model_nq": model.nq,
            "frame_count": len(frames),
            "final_pose": bool(args.final_pose),
            "first_qpos": frames[0].tolist(),
            "last_qpos": frames[-1].tolist(),
        }, indent=2))
        return
    data = mujoco.MjData(model)
    if args.final_pose:
        data.qpos[:] = frames[-1]
        mujoco.mj_forward(model, data)
        print("Displaying the final O30 grasp pose in MuJoCo")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
        return
    period = 1.0 / args.fps
    print(f"Playing {len(frames)} O30 frames in MuJoCo at {args.fps:.1f} Hz")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            started = time.perf_counter()
            data.qpos[:] = frames[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame += 1
            if frame >= len(frames):
                if args.no_loop:
                    break
                frame = 0
            time.sleep(max(0.0, period - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
