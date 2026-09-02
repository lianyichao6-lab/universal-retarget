#!/usr/bin/env python3
"""Replay an existing L25 MuJoCo scene and record five-finger tactile data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.mujoco_tactile import fingertip_geom_ids, tactile_state_from_mujoco


L25_QPOS_NAMES = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
)


def _load_targets(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    targets = []
    for record in records:
        if not isinstance(record, dict) or "target" not in record:
            raise ValueError("each trajectory frame must contain target")
        target = np.asarray(record["target"], dtype=np.float64).reshape(-1)
        if target.shape != (21,) or not np.isfinite(target).all():
            raise ValueError("each target must be finite with shape (21,)")
        targets.append(target)
    return np.asarray(targets)


def _joint_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    result = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            result[name.lower()] = int(model.jnt_qposadr[joint_id])
    missing = [name for name in L25_QPOS_NAMES if name.lower() not in result]
    if missing:
        raise ValueError(f"scene is missing L25 joints: {missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    targets = _load_targets(args.trajectory)
    addresses = _joint_qpos_addresses(model)
    fingertip_ids = fingertip_geom_ids(model)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reconstructed_object_geom")
    object_ids = None if object_id < 0 else [object_id]

    qpos = np.zeros((len(targets), 21), dtype=np.float32)
    contact = np.zeros((len(targets), 5), dtype=bool)
    wrenches = np.zeros((len(targets), 5, 6), dtype=np.float32)
    timestamps = np.arange(len(targets), dtype=np.float64) / args.fps
    for frame_index, target in enumerate(targets):
        for value, name in zip(target, L25_QPOS_NAMES):
            data.qpos[addresses[name.lower()]] = value
        mujoco.mj_forward(model, data)
        state = tactile_state_from_mujoco(
            model,
            data,
            fingertip_ids,
            object_geom_ids=object_ids,
            timestamp_s=float(timestamps[frame_index]),
            contact_force_threshold=args.contact_force_threshold,
        )
        qpos[frame_index] = target.astype(np.float32)
        contact[frame_index] = state.contact
        wrenches[frame_index] = state.wrenches

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=timestamps,
        qpos=qpos,
        tactile_contact=contact,
        tactile_wrench=wrenches,
        finger_names=np.asarray(("thumb", "index", "middle", "ring", "pinky")),
        scene=np.asarray(str(args.scene.resolve())),
        trajectory=np.asarray(str(args.trajectory.resolve())),
    )
    report = {
        "scene": str(args.scene.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "frames": int(len(targets)),
        "fps": float(args.fps),
        "contact_force_threshold": float(args.contact_force_threshold),
        "frames_with_contact": int(np.count_nonzero(contact.any(axis=1))),
        "max_contact_count": int(contact.sum(axis=1).max()),
        "contact_count_per_finger": contact.sum(axis=0).astype(int).tolist(),
        "grasp_contact_observed": bool(np.any(contact.sum(axis=1) >= 2)),
        "lift_evaluated": False,
        "hardware_command_generated": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Replayed {len(targets)} L25 frames")
    print(f"  max contact count: {report['max_contact_count']}")
    print(f"  grasp contact observed: {report['grasp_contact_observed']}")
    print(f"  tactile output: {args.output}")


if __name__ == "__main__":
    main()
