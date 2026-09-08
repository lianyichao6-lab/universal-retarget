#!/usr/bin/env python3
"""Retarget one saved HUG grasp prediction to an offline robot trajectory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.dex_backend import DEX_CONFIGS, DexRetargetBackend
from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.hug_adapter import load_prediction
from anydexretarget.hug_o30 import retarget_hug_o30
from anydexretarget.hand_contract import O30_ACTIVE_JOINT_NAMES, O30_QPOS_JOINT_NAMES
from anydexretarget.retarget import Retargeter


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


def _load_keypoints(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, str]:
    if args.canonical_grasp is not None:
        canonical_state = load_canonical_grasp_state(args.canonical_grasp)
        if canonical_state.handedness != args.hand:
            raise ValueError(
                f"Canonical grasp handedness {canonical_state.handedness!r} does not match --hand {args.hand!r}"
            )
        return (
            canonical_state.keypoints_for_retargeting(),
            canonical_state.keypoints_canonical.copy(),
            "canonical_grasp_state",
        )
    frame = load_prediction(args.prediction)
    if frame.handedness != args.hand:
        raise ValueError(
            f"HUG prediction handedness {frame.handedness!r} does not match --hand {args.hand!r}"
        )
    return frame.keypoints_3d, None, "hug_prediction"


def _retarget_l25(keypoints: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float | None]:
    if args.optimizer in DEX_CONFIGS:
        backend = DexRetargetBackend(
            args.optimizer,
            hand_side=args.hand,
            scaling_factor=args.dex_scaling,
            project_dist=args.dex_project_dist,
            escape_dist=args.dex_escape_dist,
        )
        qpos, backend_verbose = backend.retarget(keypoints)
        transformed = backend_verbose["mediapipe_kp"]
        source_names = [name.lower() for name in backend.joint_names]
        cost = None
    else:
        config_path = args.config if args.config is not None else CONFIGS[args.optimizer]
        retargeter = Retargeter.from_yaml(str(config_path), hand_side=args.hand)
        qpos, verbose = retargeter.retarget_verbose(keypoints, apply_filter=False)
        transformed = verbose["mediapipe_kp"]
        source_names = [str(name).lower() for name in retargeter.optimizer.robot.dof_joint_names]
        cost = float(verbose["cost"])

    source_by_name = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in L25_JOINT_NAMES if name.lower() not in source_by_name]
    if missing:
        raise ValueError(f"Retargeter output is missing L25 joints: {missing}")
    target = np.asarray(
        [qpos[source_by_name[name.lower()]] for name in L25_JOINT_NAMES], dtype=np.float32
    )
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml")
    )
    lower, upper = model.jnt_range[:, 0] + 1e-6, model.jnt_range[:, 1] - 1e-6
    return np.clip(target, lower, upper).astype(np.float32), np.asarray(transformed, dtype=np.float32), cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prediction", type=Path)
    input_group.add_argument("--canonical-grasp", type=Path)
    parser.add_argument("--robot", choices=("l25", "o30"), default="l25")
    parser.add_argument("--optimizer", choices=sorted(CONFIGS), default="vector")
    parser.add_argument("--config", type=Path, help="Optional native L25 optimizer YAML; cannot be used with dex or O30.")
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
    if args.robot == "o30" and args.optimizer != "vector":
        parser.error("O30 currently supports --optimizer vector only")
    if args.robot == "o30" and args.config is not None:
        parser.error("O30 uses its audited Vector configuration; do not pass --config")

    keypoints, canonical_keypoints, input_representation = _load_keypoints(args)
    if args.robot == "o30":
        result = retarget_hug_o30(keypoints)
        target = result.qpos
        transformed = result.transformed_keypoints
        joint_names = list(O30_QPOS_JOINT_NAMES)
        extra = {
            "hand_command_positions": result.command_positions.copy(),
            "hand_command_joint_names": list(O30_ACTIVE_JOINT_NAMES),
            "solver_cost": result.cost,
        }
    else:
        target, transformed, cost = _retarget_l25(keypoints, args)
        joint_names = L25_JOINT_NAMES.copy()
        extra = {} if cost is None else {"solver_cost": cost}

    records = [
        {
            "target": target.copy(),
            "sim_qpos": target.copy(),
            "robot_joint_names": joint_names,
            "human_keypoints": keypoints.copy(),
            "human_representation": input_representation,
            "human_keypoints_canonical": None if canonical_keypoints is None else canonical_keypoints.copy(),
            "human_keypoints_retarget_frame": transformed.copy(),
            "robot": args.robot,
            "optimizer": args.optimizer,
            **extra,
        }
        for _ in range(args.frames)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(records, stream)
    print(
        f"saved {len(records)} frames to {args.output} "
        f"(robot={args.robot}, input={input_representation}, qpos={target.shape}, finite={np.isfinite(target).all()})"
    )


if __name__ == "__main__":
    main()
