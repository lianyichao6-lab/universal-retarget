#!/usr/bin/env python3
"""Export a validated right-hand L25 RGB motion reference for W6.

The exporter deliberately rejects missing MediaPipe frames instead of carrying
forward stale landmarks.  The output is an auditable retargeting artifact, not
a simulation playback log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anydexretarget.hardware_adapter import L25_QPOS_JOINTS
from anydexretarget.luban_contract import L25_ACTIVE_JOINT_NAMES
from anydexretarget.retarget import Retargeter
from example.input.landmark_utils import landmarks_to_array, process_landmarks

TARGET_ANGLE = np.pi / 2.0
FULL_NAMES = tuple(L25_QPOS_JOINTS)
INDEPENDENT_NAMES = tuple(L25_ACTIVE_JOINT_NAMES)
MIMIC = {
    "thumb_ip": ("thumb_mcp", 1.03),
    "index_dip": ("index_pip", 0.89),
    "middle_dip": ("middle_pip", 0.89),
    "ring_dip": ("ring_pip", 0.89),
    "pinky_dip": ("pinky_pip", 0.89),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml"),
    )
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument("--end-time-s", type=float, default=None)
    parser.add_argument("--output-hz", type=float, default=20.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _canonical_full(qpos: np.ndarray, source_names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(source_names),) or not np.isfinite(values).all():
        raise ValueError("Retargeter produced an invalid L25 qpos vector")
    source = {name: index for index, name in enumerate(source_names)}
    if set(source) == set(FULL_NAMES):
        return np.asarray([values[source[name]] for name in FULL_NAMES], dtype=np.float64)
    if set(source) != set(INDEPENDENT_NAMES):
        raise ValueError("L25 retargeter joint names do not match full or independent contract")
    full = np.zeros(len(FULL_NAMES), dtype=np.float64)
    full_index = {name: index for index, name in enumerate(FULL_NAMES)}
    for name in INDEPENDENT_NAMES:
        full[full_index[name]] = values[source[name]]
    for name, (parent, ratio) in MIMIC.items():
        full[full_index[name]] = ratio * full[full_index[parent]]
    return full


def _resample(times: np.ndarray, values: np.ndarray, output_hz: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1])
    output_times = np.arange(0.0, duration + 1e-9, 1.0 / output_hz)
    flat = values.reshape(len(times), -1)
    sampled = np.stack(
        [np.interp(output_times, times, flat[:, index]) for index in range(flat.shape[1])],
        axis=-1,
    )
    return output_times, sampled.reshape((len(output_times), *values.shape[1:]))


def main() -> None:
    args = _parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.start_time_s < 0.0 or args.output_hz <= 0.0:
        raise ValueError("start-time-s must be non-negative and output-hz positive")

    with args.config.open("r", encoding="utf-8") as stream:
        retarget_config = yaml.safe_load(stream)
    retargeter = Retargeter.from_yaml(str(args.config), hand_side="right")
    source_names = tuple(str(name) for name in retargeter.optimizer.robot.dof_joint_names)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_time_s = args.end_time_s
    raw_times: list[float] = []
    raw_keypoints: list[np.ndarray] = []
    raw_q_full: list[np.ndarray] = []
    expected_label = "Left"  # MediaPipe labels unmirrored camera-view right hands as Left.

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frame_index / fps
            frame_index += 1
            if time_s < args.start_time_s:
                continue
            if end_time_s is not None and time_s > end_time_s:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            selected = None
            if result.multi_hand_landmarks and result.multi_handedness:
                for landmarks, handedness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    if handedness.classification[0].label == expected_label:
                        selected = landmarks
                        break
            if selected is None:
                raise RuntimeError(f"MediaPipe missed the right hand at {time_s:.3f}s")
            keypoints = process_landmarks(
                landmarks_to_array(selected, width, height), depth_scale=args.depth_scale
            )
            q_full = _canonical_full(retargeter.retarget(keypoints, apply_filter=True), source_names)
            raw_times.append(time_s)
            raw_keypoints.append(keypoints)
            raw_q_full.append(q_full)
    finally:
        capture.release()
        hands.close()

    if len(raw_times) < 2:
        raise RuntimeError("Selected video interval has fewer than two valid frames")
    times = np.asarray(raw_times, dtype=np.float64)
    source_start_time_s = float(times[0])
    source_end_time_s = float(times[-1])
    times -= source_start_time_s
    keypoints = np.stack(raw_keypoints)
    q_full = np.stack(raw_q_full)
    timestamps, keypoints = _resample(times, keypoints, args.output_hz)
    _, q_full = _resample(times, q_full, args.output_hz)
    independent_index = [FULL_NAMES.index(name) for name in INDEPENDENT_NAMES]
    q_independent = q_full[:, independent_index]

    root = REPO_ROOT
    metadata = {
        "source_video_sha256": _sha256(args.video),
        "source_fps": fps,
        "source_start_time_s": source_start_time_s,
        "source_end_time_s": source_end_time_s,
        "mediapipe_version": mp.__version__,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "temporal_filter_type": "Retargeter.LPFilter",
        "temporal_filter_params": retarget_config.get("retarget", {}),
        "anydex_git_commit": _git_commit(root),
        "projection_version": "l25-full-to-independent-v1",
        "output_hz": args.output_hz,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int64),
        timestamps_s=timestamps,
        human_keypoints=keypoints,
        q_ref_full=q_full,
        q_ref_independent=q_independent,
        joint_names_full=np.asarray(FULL_NAMES),
        joint_names_independent=np.asarray(INDEPENDENT_NAMES),
        hand_side=np.asarray("right"),
        object_name=np.asarray("cylinder_medium"),
        target_angle=np.asarray(TARGET_ANGLE, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved {len(timestamps)} W6 reference frames to {args.output}")


if __name__ == "__main__":
    main()
