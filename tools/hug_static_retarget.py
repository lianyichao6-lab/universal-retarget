#!/usr/bin/env python3
"""Retarget one saved HUG grasp prediction to an offline robot trajectory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import mujoco

from anydexretarget.hug_adapter import load_prediction
from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter
from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "vector": ROOT / "example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml",
    "adaptive": ROOT / "example/config/adaptive/mediapipe/mediapipe_linkerhand_l25.yaml",
    **DEX_CONFIGS,
}
L25_JOINT_NAMES = [
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prediction", type=Path)
    input_group.add_argument("--canonical-grasp", type=Path)
    parser.add_argument("--optimizer", choices=sorted(CONFIGS), default="vector")
    parser.add_argument("--config", type=Path, help="Optional native optimizer YAML; cannot be used with dex backends.")
    parser.add_argument("--dex-scaling", type=float, help="DexPilot/Joint-Angle global scaling override.")
    parser.add_argument("--dex-project-dist", type=float, help="DexPilot pinch projection distance in meters.")
    parser.add_argument("--dex-escape-dist", type=float, help="DexPilot pinch escape distance in meters.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--hand", choices=["right"], default="right")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.config is not None and args.optimizer in DEX_CONFIGS:
        parser.error("--config is only supported for native vector/adaptive optimizers")
    if any(v is not None and v <= 0 for v in (args.dex_scaling, args.dex_project_dist, args.dex_escape_dist)):
        parser.error("DexPilot overrides must be positive")
    if args.dex_project_dist is not None and args.dex_escape_dist is not None and args.dex_escape_dist < args.dex_project_dist:
        parser.error("--dex-escape-dist must be >= --dex-project-dist")

    if args.canonical_grasp is not None:
        canonical_state = load_canonical_grasp_state(args.canonical_grasp)
        if canonical_state.handedness != args.hand:
            raise ValueError(
                f"Canonical grasp handedness {canonical_state.handedness!r} does not match --hand {args.hand!r}"
            )
        keypoints = canonical_state.keypoints_for_retargeting()
        canonical_keypoints = canonical_state.keypoints_canonical.copy()
        input_representation = "canonical_grasp_state"
    else:
        frame = load_prediction(args.prediction)
        if frame.handedness != args.hand:
            raise ValueError(
                f"HUG prediction handedness {frame.handedness!r} does not match --hand {args.hand!r}"
            )
        keypoints = frame.keypoints_3d
        canonical_keypoints = None
        input_representation = "hug_prediction"

    if args.optimizer in DEX_CONFIGS:
        backend = DexRetargetBackend(args.optimizer, hand_side=args.hand, scaling_factor=args.dex_scaling, project_dist=args.dex_project_dist, escape_dist=args.dex_escape_dist)
        qpos, backend_verbose = backend.retarget(keypoints)
        verbose = {"mediapipe_kp": backend_verbose["mediapipe_kp"]}
        source_names = [name.lower() for name in backend.joint_names]
    else:
        config_path = args.config if args.config is not None else CONFIGS[args.optimizer]
        retargeter = Retargeter.from_yaml(str(config_path), hand_side=args.hand)
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        source_names = [str(name).lower() for name in retargeter.optimizer.robot.dof_joint_names]
    source_by_name = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in L25_JOINT_NAMES if name.lower() not in source_by_name]
    if missing:
        raise ValueError(f"Retargeter output is missing L25 joints: {missing}")
    target = np.asarray(
        [qpos[source_by_name[name.lower()]] for name in L25_JOINT_NAMES],
        dtype=np.float32,
    )
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml")
    )
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    # Leave a tiny margin so float32 serialization cannot cross a limit by 1 ulp.
    lower = lower + 1e-6
    upper = upper - 1e-6
    unclamped = target.copy()
    target = np.clip(target, lower, upper).astype(np.float32)
    clamp_count = int(np.count_nonzero(unclamped != target))
    records = [
        {
            "target": target.copy(),
            "sim_qpos": target.copy(),
            "human_keypoints": keypoints.copy(),
            "human_representation": input_representation,
            "human_keypoints_canonical": (
                None if canonical_keypoints is None else canonical_keypoints.copy()
            ),
            "human_keypoints_retarget_frame": verbose["mediapipe_kp"].copy(),
            "robot": "l25",
            "optimizer": args.optimizer,
        }
        for _ in range(args.frames)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream)
    print(
        f"saved {len(records)} frames to {args.output} "
        f"(input={input_representation}, qpos={target.shape}, finite={np.isfinite(target).all()}, clamped_joints={clamp_count})"
    )


if __name__ == "__main__":
    main()
