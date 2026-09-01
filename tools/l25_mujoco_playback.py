#!/usr/bin/env python3
"""Play an offline L25 qpos pickle directly in MuJoCo."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml"
L25_QPOS_NAMES = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
]


def model_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None:
            raise ValueError(f"MuJoCo joint {index} has no name")
        names.append(name)
    return names


def load_qpos(path: Path, model: mujoco.MjModel) -> np.ndarray:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Trajectory must be a non-empty pickle list")
    names = model_joint_names(model)
    by_name = {name.lower(): index for index, name in enumerate(names)}
    frames = []
    for frame in records:
        if not isinstance(frame, dict) or "target" not in frame:
            raise ValueError("Each trajectory frame must contain target")
        target = np.asarray(frame["target"], dtype=np.float64).reshape(-1)
        if target.size != 21 or not np.isfinite(target).all():
            raise ValueError(f"Expected finite 21-DoF target, got {target.shape}")
        qpos = np.zeros(model.nq, dtype=np.float64)
        for source_index, name in enumerate(L25_QPOS_NAMES):
            target_index = by_name.get(name.lower())
            if target_index is None:
                raise ValueError(f"Missing MuJoCo joint {name}")
            qpos[target_index] = target[source_index]
        frames.append(qpos)
    return np.asarray(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    frames = load_qpos(args.trajectory, model)
    data = mujoco.MjData(model)
    period = 1.0 / args.fps
    print(f"Playing {len(frames)} L25 frames in MuJoCo at {args.fps:.1f} Hz")
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
