#!/usr/bin/env python3
"""Replay an L25 scene with a kinematic hand lift and tactile logging."""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.luban_arm import arm_flange_pose_xyzw, homogeneous_transform
from anydexretarget.mujoco_tactile import fingertip_geom_ids, tactile_state_from_mujoco


L25_QPOS_NAMES = (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load_targets(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must be a non-empty pickle list")
    values = [np.asarray(record["target"], dtype=np.float64).reshape(-1) for record in records]
    result = np.asarray(values)
    if result.shape != (len(records), 21) or not np.isfinite(result).all():
        raise ValueError("trajectory targets must be finite with shape (N, 21)")
    return result


def _make_scene(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene is missing worldbody")
    hand = worldbody.find("body[@name='hand_base_link']")
    obj = worldbody.find("body[@name='reconstructed_object']")
    if hand is None or obj is None:
        raise ValueError("scene must contain hand_base_link and reconstructed_object")
    hand.set("mocap", "true")
    if obj.find("freejoint") is None:
        obj.insert(0, ET.Element("freejoint", {"name": "reconstructed_object_freejoint"}))
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)


def _joint_addresses(model: mujoco.MjModel) -> dict[str, int]:
    addresses = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            addresses[name.lower()] = int(model.jnt_qposadr[joint_id])
    missing = [name for name in L25_QPOS_NAMES if name.lower() not in addresses]
    if missing:
        raise ValueError(f"scene is missing joints: {missing}")
    return addresses


def _arm_targets(path: Path | None, frames: int) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["T_robot_base_arm_flange"], dtype=np.float64)
    if values.shape == (4, 4):
        values = np.repeat(values[None], frames, axis=0)
    if values.shape != (frames, 4, 4):
        raise ValueError(f"arm target must have shape (4,4) or ({frames},4,4), got {values.shape}")
    return np.asarray([homogeneous_transform(value, "arm target") for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    parser.add_argument("--arm-target", type=Path, help="NPZ with T_robot_base_arm_flange, shape (N,4,4).")
    parser.add_argument("--flange-hand", type=Path, help="NPY 4x4 T_arm_flange_l25_hand.")
    parser.add_argument("--lift-m", type=float, default=0.0)
    parser.add_argument("--kinematic-hold", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.lift_m < 0:
        parser.error("--fps must be positive and --lift-m must be non-negative")

    targets = _load_targets(args.trajectory)
    with tempfile.TemporaryDirectory(prefix="anydex_l25_lift_") as temp_dir:
        replay_scene = Path(temp_dir) / "replay_scene.xml"
        _make_scene(args.scene, replay_scene)
        model = mujoco.MjModel.from_xml_path(str(replay_scene))
        data = mujoco.MjData(model)
        addresses = _joint_addresses(model)
        fingertip_ids = fingertip_geom_ids(model)
        object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "reconstructed_object_geom")
        object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reconstructed_object")
        object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "reconstructed_object_freejoint")
        hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")
        mocap_id = int(model.body_mocapid[hand_body_id])
        object_qpos = int(model.jnt_qposadr[object_joint_id])
        if min(object_geom_id, object_body_id, object_joint_id, hand_body_id, mocap_id) < 0:
            raise ValueError("generated scene lacks required hand/object bodies")
        arm_targets = _arm_targets(args.arm_target, len(targets))
        flange_hand = np.eye(4, dtype=np.float64)
        if args.flange_hand is not None:
            flange_hand = homogeneous_transform(np.load(args.flange_hand, allow_pickle=False), "T_arm_flange_l25_hand")
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        initial_hand_pos = data.mocap_pos[mocap_id].copy()
        initial_hand_quat = data.mocap_quat[mocap_id].copy()
        initial_object_quat = data.qpos[object_qpos + 3:object_qpos + 7].copy()

        count = len(targets)
        timestamps = np.arange(count, dtype=np.float64) / args.fps
        contact = np.zeros((count, 5), dtype=bool)
        wrenches = np.zeros((count, 5, 6), dtype=np.float32)
        hand_positions = np.zeros((count, 3), dtype=np.float32)
        object_positions = np.zeros((count, 3), dtype=np.float32)
        hold_offset = None
        grasp_frame = None
        for index, target in enumerate(targets):
            if arm_targets is None:
                hand_pos = initial_hand_pos.copy()
                hand_quat = initial_hand_quat.copy()
                if args.lift_m and index >= count // 2:
                    fraction = (index - count // 2 + 1) / max(1, count - count // 2)
                    hand_pos[2] += args.lift_m * min(1.0, fraction)
            else:
                hand_transform = arm_targets[index] @ flange_hand
                hand_pos, quaternion_xyzw = arm_flange_pose_xyzw(hand_transform)
                hand_quat = np.asarray([quaternion_xyzw[3], *quaternion_xyzw[:3]])
            data.mocap_pos[mocap_id] = hand_pos
            data.mocap_quat[mocap_id] = hand_quat
            for value, name in zip(target, L25_QPOS_NAMES):
                data.qpos[addresses[name.lower()]] = value
            mujoco.mj_forward(model, data)
            state = tactile_state_from_mujoco(
                model, data, fingertip_ids,
                object_geom_ids=None if object_geom_id < 0 else [object_geom_id],
                timestamp_s=float(timestamps[index]),
                contact_force_threshold=args.contact_force_threshold,
            )
            if args.kinematic_hold and hold_offset is None and state.contact.sum() >= 2:
                hold_offset = data.xpos[object_body_id].copy() - hand_pos
                grasp_frame = index
            if hold_offset is not None:
                data.qpos[object_qpos:object_qpos + 3] = hand_pos + hold_offset
                data.qpos[object_qpos + 3:object_qpos + 7] = initial_object_quat
                mujoco.mj_forward(model, data)
                state = tactile_state_from_mujoco(
                    model, data, fingertip_ids,
                    object_geom_ids=None if object_geom_id < 0 else [object_geom_id],
                    timestamp_s=float(timestamps[index]),
                    contact_force_threshold=args.contact_force_threshold,
                )
            contact[index] = state.contact
            wrenches[index] = state.wrenches
            hand_positions[index] = hand_pos
            object_positions[index] = data.xpos[object_body_id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=timestamps,
        qpos=targets.astype(np.float32),
        tactile_contact=contact,
        tactile_wrench=wrenches,
        hand_position=hand_positions,
        object_position=object_positions,
        finger_names=np.asarray(FINGERS),
    )
    report = {
        "frames": count,
        "fps": args.fps,
        "grasp_frame": grasp_frame,
        "max_contact_count": int(contact.sum(axis=1).max()),
        "grasp_contact_observed": bool(np.any(contact.sum(axis=1) >= 2)),
        "object_height_delta_m": float(object_positions[-1, 2] - object_positions[0, 2]),
        "lift_evaluated": bool(args.lift_m > 0),
        "kinematic_hold": bool(args.kinematic_hold),
        "lift_success": bool(args.kinematic_hold and grasp_frame is not None and object_positions[-1, 2] - object_positions[0, 2] >= 0.05),
        "physics_interpretation": "kinematic object hold; not a force-closure or hardware validation",
        "hardware_command_generated": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Replayed {count} frames; max contact count={report['max_contact_count']}")
    print(f"grasp_contact_observed={report['grasp_contact_observed']} lift_success={report['lift_success']}")
    print(f"tactile output: {args.output}")


if __name__ == "__main__":
    main()
