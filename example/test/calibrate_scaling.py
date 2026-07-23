"""Calibrate segment_scaling for any robot hand and input source.

Supports two calibration modes:

* ``open``: robust open-hand morphology calibration.  This is the legacy
  robot-distance / human-distance estimate, but uses medians instead of means.
* ``functional``: starts from the open-hand estimate and then captures four
  thumb-to-finger pinch poses.  Fingertip scales are jointly refined so the
  scaled green skeleton preserves opposition contacts instead of optimizing
  neutral/open-hand lengths only.

Outputs recommended segment_scaling values and optionally writes them back to
both Adaptive and Vector config YAML files.

Supported inputs: mediapipe (video/camera), noitom, quest3, avp, pico4

Usage:
    python test/calibrate_scaling.py --robot sharpa --input mediapipe --video data/right.mp4
    python test/calibrate_scaling.py --robot wuji --input mediapipe
    python test/calibrate_scaling.py --robot wuji --input noitom
    python test/calibrate_scaling.py --robot wuji --input quest3
    python test/calibrate_scaling.py --config config/adaptive/avp/avp_wuji_hand.yaml --input avp
    python test/calibrate_scaling.py --robot wuji --input mediapipe --optimizer both --write
    python test/calibrate_scaling.py --robot gaia --input pico4 --hand right --calibration-mode functional
    python test/calibrate_scaling.py --robot gaia --input pico4 --hand right --calibration-mode functional --optimizer both --write
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

ROBOT_NAME_MAP = {
    "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
    "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
    "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
    "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand",
    "gaia": "gaia_hand20",
}

INPUT_TO_CONFIG_DIR = {
    "mediapipe": "mediapipe",
    "noitom": "noitom",
    "quest3": "quest3",
    "avp": "avp",
    "pico4": "pico4",
}

# MediaPipe task_kp → (finger, level): level 0=PIP, 1=DIP, 2=TIP
_KP_TO_FINGER_LEVEL = {
    2: ("thumb",  0), 3: ("thumb",  1), 4: ("thumb",  2),
    6: ("index",  0), 7: ("index",  1), 8: ("index",  2),
    10: ("middle", 0), 11: ("middle", 1), 12: ("middle", 2),
    14: ("ring",   0), 15: ("ring",   1), 16: ("ring",   2),
    18: ("pinky",  0), 19: ("pinky",  1), 20: ("pinky",  2),
}


def _resolve_config_paths(args, robot_file):
    """Return (adaptive_path, vector_path) as Path objects (may not exist)."""
    config_dir = INPUT_TO_CONFIG_DIR[args.input]
    adaptive = EXAMPLE_ROOT / f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"
    vector   = EXAMPLE_ROOT / f"config/vector/{config_dir}/{config_dir}_{robot_file}.yaml"
    return adaptive, vector


def _backup_config(config_path: Path):
    """Create a timestamped backup before changing a tuned configuration."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup_path)
    print(f"  已备份: {backup_path}")
    return backup_path


def _write_adaptive(config_path: Path, result: dict, backup: bool = True):
    """Update retarget.segment_scaling in an adaptive config YAML."""
    if backup:
        _backup_config(config_path)
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        doc = ryaml.load(f)

    seg = doc["retarget"]["segment_scaling"]
    for fname, vals in result.items():
        if fname not in seg:
            continue
        # Three-value format is [PIP, DIP, TIP].  Four-value format is
        # [MCP, PIP, DIP, TIP], so preserve the independently configured MCP
        # scale (Linker L20) and write calibration values into slots 1:4.
        dst = seg[fname]
        start = 1 if len(dst) == 4 else 0
        if len(dst) - start < len(vals):
            raise ValueError(
                f"segment_scaling.{fname} has unsupported length {len(dst)}"
            )
        for i, v in enumerate(vals):
            dst[start + i] = v

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)


def _write_vector(config_path: Path, result: dict, backup: bool = True):
    """Update key_vectors[*].scale in a vector config YAML."""
    if backup:
        _backup_config(config_path)
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        doc = ryaml.load(f)

    kv_list = doc["retarget"]["key_vectors"]
    for entry in kv_list:
        task_kp = int(entry.get("task_kp", -1))
        mapping = _KP_TO_FINGER_LEVEL.get(task_kp)
        if mapping is None:
            continue
        fname, level = mapping
        if fname in result:
            entry["scale"] = float(result[fname][level])

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)


def write_configs(args, robot_file, result, explicit_config_path=None):
    """Write calibration result to adaptive and/or vector config files.

    If --config was given, only that single file is written regardless of
    --optimizer.  Otherwise both adaptive/vector paths are resolved
    automatically and written according to --optimizer.
    """
    optimizer_mode = getattr(args, "optimizer", "both")

    # Single explicit config path
    if explicit_config_path:
        config_path = Path(explicit_config_path)
        if not config_path.is_absolute():
            config_path = EXAMPLE_ROOT / config_path
        _write_single(config_path, result, args)
        return

    adaptive_path, vector_path = _resolve_config_paths(args, robot_file)
    targets = []
    if optimizer_mode in ("adaptive", "both") and adaptive_path.exists():
        targets.append(("adaptive", adaptive_path))
    if optimizer_mode in ("vector", "both") and vector_path.exists():
        targets.append(("vector", vector_path))

    if not targets:
        print("  未找到对应配置文件，跳过写入。")
        return

    for opt_type, path in targets:
        try:
            if opt_type == "adaptive":
                _write_adaptive(path, result, backup=True)
            else:
                _write_vector(path, result, backup=True)
            print(f"  已写入 ({opt_type}): {path}")
        except Exception as e:
            print(f"  写入失败 ({path}): {e}")


def _write_single(config_path: Path, result: dict, args):
    """Write to a single explicitly-specified config file, auto-detecting type."""
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        opt_type_str = raw.get("optimizer", {}).get("type", "")
        if "KeyVector" in opt_type_str:
            _write_vector(config_path, result, backup=True)
            print(f"  已写入 (vector): {config_path}")
        else:
            _write_adaptive(config_path, result, backup=True)
            print(f"  已写入 (adaptive): {config_path}")
    except Exception as e:
        print(f"  写入失败 ({config_path}): {e}")


def _get_link_pos(robot, link_name, offset=None):
    lid = robot.get_link_index(link_name)
    pose = robot.get_link_pose(lid)
    pos = pose[:3, 3].copy()
    if offset is not None:
        pos += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
    return pos


def get_robot_distances(optimizer):
    """Compute robot FK distances at neutral pose.

    Returns:
        cumulative: (pip_dists, dip_dists, tip_dists) — origin→link3/link4/tip
        segment: (seg_pip, seg_dip, seg_tip) — origin→link3, link3→link4, link4→tip
    """
    robot = optimizer.robot
    nf = optimizer.num_fingers

    qpos = optimizer.neutral_qpos.copy() if optimizer.neutral_qpos is not None \
        else np.zeros(robot.model.nq)
    robot.compute_forward_kinematics(qpos)

    origin_pos = _get_link_pos(robot, optimizer.origin_link_name)

    def pos_for(link_names, offsets):
        result = []
        for fi in range(nf):
            if fi >= len(link_names):
                result.append(None)
                continue
            off = offsets[fi] if offsets is not None else None
            result.append(_get_link_pos(robot, link_names[fi], off))
        return result

    pip_pos = pos_for(
        getattr(optimizer, 'link3_names', []),
        getattr(optimizer, 'link3_offsets', None),
    )
    dip_pos = pos_for(
        getattr(optimizer, 'link4_names', []),
        getattr(optimizer, 'link4_offsets', None),
    )
    tip_pos = pos_for(
        optimizer.task_link_names,
        getattr(optimizer, 'task_offsets', None),
    )

    # Cumulative: origin → each link
    pip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in pip_pos]
    dip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in dip_pos]
    tip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in tip_pos]

    # Per-segment: link3→link4, link4→tip
    seg_pip = pip_dists  # origin→link3
    seg_dip = []
    seg_tip = []
    for fi in range(nf):
        if pip_pos[fi] is not None and dip_pos[fi] is not None:
            seg_dip.append(float(np.linalg.norm(dip_pos[fi] - pip_pos[fi])))
        else:
            seg_dip.append(None)
        if dip_pos[fi] is not None and tip_pos[fi] is not None:
            seg_tip.append(float(np.linalg.norm(tip_pos[fi] - dip_pos[fi])))
        else:
            seg_tip.append(None)

    return (pip_dists, dip_dists, tip_dists), (seg_pip, seg_dip, seg_tip)


def _transform_input_keypoints(raw_kp, retargeter, hand):
    """Return the exact transformed keypoints passed into optimizer.solve()."""
    kp = apply_mediapipe_transformations(raw_kp, hand)
    if retargeter.rotation_xyz:
        kp = retargeter._apply_rotation(kp)
    kp = retargeter.preprocess_transformed_keypoints(kp)
    return np.asarray(kp, dtype=np.float64)


def _robust_center(values):
    """Robust scalar/vector center used to suppress Pico tracking spikes."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return np.median(arr, axis=0)


def _wait_for_capture_start(input_device, message):
    """Preview input until Enter/space/s is pressed."""
    print(message)
    import select
    import sys as _sys
    while True:
        input_device.get_fingers_data()
        ready, _, _ = select.select([_sys.stdin], [], [], 0.0)
        if ready:
            _sys.stdin.readline()
            return
        try:
            import cv2
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), 13, ord(' ')):
                return
        except Exception:
            pass


def _pose_countdown(input_device, delay):
    """Keep refreshing the input and give the user time to form the pose."""
    delay = float(delay)
    if delay <= 0:
        return
    start = time.monotonic()
    last_second = None
    while True:
        elapsed = time.monotonic() - start
        remaining = delay - elapsed
        if remaining <= 0:
            break
        second = int(np.ceil(remaining))
        if second != last_second:
            print(f"  请调整姿势，{second} 秒后开始采集...", flush=True)
            last_second = second
        input_device.get_fingers_data()
        time.sleep(0.005)
    print("  开始采集。", flush=True)


def collect_pinch_vectors(
    input_device,
    retargeter,
    hand,
    finger_index,
    duration,
    max_gap_m,
):
    """Collect wrist→thumb-tip and wrist→finger-tip vectors during a pinch.

    Only frames whose unscaled thumb/finger tip gap is below ``max_gap_m`` are
    accepted.  Returning vectors (rather than lengths) lets the functional
    calibration directly minimize the green skeleton's opposition gap.
    """
    optimizer = retargeter.optimizer
    mp_thumb = optimizer.mp_finger_indices[0]
    mp_finger = optimizer.mp_finger_indices[finger_index]
    thumb_tip_idx = optimizer.MP_TIP_INDICES[mp_thumb]
    finger_tip_idx = optimizer.MP_TIP_INDICES[mp_finger]

    thumb_vectors = []
    finger_vectors = []
    raw_gaps = []
    seen = 0
    start = time.time()
    last_print = 0.0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue
        kp = _transform_input_keypoints(raw_kp, retargeter, hand)
        wrist = kp[0]
        thumb_tip = kp[thumb_tip_idx]
        finger_tip = kp[finger_tip_idx]
        gap = float(np.linalg.norm(thumb_tip - finger_tip))
        seen += 1
        if np.isfinite(gap) and gap <= max_gap_m:
            thumb_vectors.append(thumb_tip - wrist)
            finger_vectors.append(finger_tip - wrist)
            raw_gaps.append(gap)

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(
                f"  采集中... {elapsed:.0f}/{duration:.0f}s  "
                f"(有效 {len(raw_gaps)}/{seen} 帧, "
                f"阈值 {max_gap_m * 100:.1f}cm)",
                flush=True,
            )

    if not raw_gaps:
        return None
    return {
        "thumb_vector": _robust_center(thumb_vectors),
        "finger_vector": _robust_center(finger_vectors),
        "raw_gap": float(_robust_center(raw_gaps)),
        "valid_frames": len(raw_gaps),
        "seen_frames": seen,
    }


def refine_tip_scales_for_pinches(
    initial_tip_scales,
    pinch_samples,
    prior_weight=1.0,
    pinch_weight=4.0,
    max_relative_change=0.25,
):
    """Jointly refine fingertip scales using captured opposition contacts.

    The least-squares objective is

        prior_weight * Σ ((s_i - s0_i) / s0_i)^2
        + pinch_weight * Σ ||s_thumb*v_thumb - s_i*v_i||^2 / L_i^2

    where ``s0`` is the open-hand morphology estimate.  The first term keeps
    robot/human reach matching; the second makes the actual scaled green tips
    agree during opposition.  Bounds prevent a bad capture from producing an
    unsafe jump in scale.
    """
    initial = np.asarray(initial_tip_scales, dtype=np.float64)
    if initial.ndim != 1 or np.any(initial <= 0):
        raise ValueError("initial_tip_scales must be a positive 1-D array")
    if prior_weight <= 0 or pinch_weight < 0:
        raise ValueError("prior_weight must be > 0 and pinch_weight must be >= 0")
    if not 0 <= max_relative_change < 1:
        raise ValueError("max_relative_change must be in [0, 1)")

    n = len(initial)
    rows = []
    rhs = []

    # Relative prior: (s_i - s0_i) / s0_i = 0.
    prior_scale = np.sqrt(prior_weight)
    for i in range(n):
        row = np.zeros(n, dtype=np.float64)
        row[i] = prior_scale / initial[i]
        rows.append(row)
        rhs.append(prior_scale)

    contact_scale = np.sqrt(pinch_weight)
    for finger_index, sample in sorted(pinch_samples.items()):
        if sample is None or not (0 < finger_index < n):
            continue
        vt = np.asarray(sample["thumb_vector"], dtype=np.float64)
        vf = np.asarray(sample["finger_vector"], dtype=np.float64)
        length_ref = max(0.5 * (np.linalg.norm(vt) + np.linalg.norm(vf)), 1e-6)
        for axis in range(3):
            row = np.zeros(n, dtype=np.float64)
            row[0] = contact_scale * vt[axis] / length_ref
            row[finger_index] = -contact_scale * vf[axis] / length_ref
            rows.append(row)
            rhs.append(0.0)

    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)

    lower = initial * (1.0 - max_relative_change)
    upper = initial * (1.0 + max_relative_change)
    return np.clip(solution, lower, upper)


def _scaled_pinch_gap(sample, tip_scales, finger_index):
    if sample is None:
        return None
    vt = np.asarray(sample["thumb_vector"], dtype=np.float64)
    vf = np.asarray(sample["finger_vector"], dtype=np.float64)
    return float(np.linalg.norm(tip_scales[0] * vt - tip_scales[finger_index] * vf))


def collect_human_distances(input_device, retargeter, hand, duration):
    """Collect input data for `duration` seconds and compute robust median distances.

    Returns:
        frames: number of valid frames
        cumulative: (pip_median, dip_median, tip_median) — wrist→PIP/DIP/TIP
        segment: (seg_pip, seg_dip, seg_tip) — wrist→PIP, PIP→DIP, DIP→TIP
    """
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    pip_idx = [optimizer.MP_PIP_INDICES[i] for i in optimizer.mp_finger_indices]
    dip_idx = [optimizer.MP_DIP_INDICES[i] for i in optimizer.mp_finger_indices]
    tip_idx = [optimizer.MP_TIP_INDICES[i] for i in optimizer.mp_finger_indices]

    pip_buf = [[] for _ in range(nf)]
    dip_buf = [[] for _ in range(nf)]
    tip_buf = [[] for _ in range(nf)]
    seg_pip_buf = [[] for _ in range(nf)]
    seg_dip_buf = [[] for _ in range(nf)]
    seg_tip_buf = [[] for _ in range(nf)]

    start = time.time()
    frames = 0
    last_print = 0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue

        kp = _transform_input_keypoints(raw_kp, retargeter, hand)

        wrist = kp[0]
        for i in range(nf):
            p_pip = kp[pip_idx[i]]
            p_dip = kp[dip_idx[i]]
            p_tip = kp[tip_idx[i]]
            # Cumulative
            pip_buf[i].append(float(np.linalg.norm(p_pip - wrist)))
            dip_buf[i].append(float(np.linalg.norm(p_dip - wrist)))
            tip_buf[i].append(float(np.linalg.norm(p_tip - wrist)))
            # Per-segment
            seg_pip_buf[i].append(float(np.linalg.norm(p_pip - wrist)))
            seg_dip_buf[i].append(float(np.linalg.norm(p_dip - p_pip)))
            seg_tip_buf[i].append(float(np.linalg.norm(p_tip - p_dip)))
        frames += 1

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(f"  采集中... {elapsed:.0f}/{duration:.0f}s  ({frames} 帧)", flush=True)

    if frames == 0:
        return frames, None, None

    def robust_list(bufs):
        return [float(_robust_center(b)) if b else 0.0 for b in bufs]

    cumulative = (robust_list(pip_buf), robust_list(dip_buf), robust_list(tip_buf))
    segment = (robust_list(seg_pip_buf), robust_list(seg_dip_buf), robust_list(seg_tip_buf))
    return frames, cumulative, segment


def create_input_device(args):
    """Create input device based on --input type."""
    if args.input == "mediapipe":
        if args.video:
            from input.video import Video
            return Video(
                video_path=args.video,
                hand_side=args.hand,
                show_video=args.show_video,
                loop=True,
            )
        else:
            from input.realsense import Realsense
            return Realsense(hand_side=args.hand, show_video=True)
    elif args.input == "noitom":
        from input.noitom import NoitomInput
        return NoitomInput(
            local_ip=args.noitom_local_ip,
            local_port=args.noitom_local_port,
            server_ip=args.noitom_server_ip,
            server_port=args.noitom_server_port,
        )
    elif args.input == "quest3":
        from input.quest3 import Quest3
        return Quest3(port=args.quest3_port, protocol=args.quest3_protocol)
    elif args.input == "avp":
        from input.visionpro import VisionPro
        return VisionPro(ip=args.avp_ip)
    elif args.input == "pico4":
        from input.pico4 import Pico4
        return Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
    else:
        raise ValueError(f"Unknown input: {args.input}")


def fmt_cm(v):
    return f"{v*100:.2f}cm" if v else "   N/A"


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate segment_scaling for any robot hand and input source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Config YAML path (overrides --robot and --input for config lookup)")
    parser.add_argument("--robot", default="wuji",
                        choices=list(ROBOT_NAME_MAP.keys()),
                        help="Robot hand type (default: wuji)")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="mediapipe",
                        choices=["mediapipe", "noitom", "quest3", "avp", "pico4"],
                        help="Input source (default: mediapipe)")
    parser.add_argument("--video", default=None,
                        help="Video file for mediapipe input (omit to use camera)")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="采集时长（秒），默认 5")
    parser.add_argument(
        "--calibration-mode",
        default="open",
        choices=["open", "functional"],
        help=("open=仅张手长度标定；functional=张手初值 + 四次对指任务标定 "
              "（Gaia/Pico 推荐）"),
    )
    parser.add_argument("--pinch-duration", type=float, default=3.0,
                        help="functional 模式下每个对指动作的采集时长（秒），默认 3")
    parser.add_argument("--pose-delay", type=float, default=2.0,
                        help="按 Enter/空格/s 后留给调整姿势的时间（秒），默认 2")
    parser.add_argument("--pinch-max-gap", type=float, default=3.5,
                        help="接受对指帧的原始指尖距离阈值（厘米），默认 3.5")
    parser.add_argument("--pinch-weight", type=float, default=4.0,
                        help="对指接触相对于张手长度初值的权重，默认 4.0")
    parser.add_argument("--scale-prior-weight", type=float, default=1.0,
                        help="保留张手长度初值的权重，默认 1.0")
    parser.add_argument("--functional-base", default="current", choices=["current", "open"],
                        help="functional 初值：current=当前配置（推荐），open=张手距离比")
    parser.add_argument("--max-tip-scale-change", type=float, default=0.10,
                        help="对指阶段允许 TIP scale 相对初值最大变化，默认 0.10")
    parser.add_argument("--min-distal-ratio", type=float, default=0.6,
                        help="绿色 DIP→TIP 至少保留机器人末节长度的比例，默认 0.6")
    # Noitom
    parser.add_argument("--noitom-local-ip", default="0.0.0.0")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    # Quest3
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", default="udp", choices=["udp", "tcp"])
    # Pico4
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"],
                        help="Pico 4 input mode: relay daemon (default) or direct TCP server")
    parser.add_argument("--pico4-relay-host", default="127.0.0.1",
                        help="Pico 4 relay daemon host (default: 127.0.0.1)")
    parser.add_argument("--pico4-relay-port", type=int, default=63902,
                        help="Pico 4 relay daemon port (default: 63902)")
    parser.add_argument("--pico4-port", type=int, default=63901,
                        help="Pico 4 direct-mode TCP listen port (default: 63901)")
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888,
                        help="Pico 4 direct-mode UDP broadcast port (default: 29888)")
    # AVP
    parser.add_argument("--avp-ip", default="192.168.50.127")
    # Write-back options
    parser.add_argument("--optimizer", default="both",
                        choices=["adaptive", "vector", "both"],
                        help="写入哪种配置文件（默认: both）")
    parser.add_argument("--write", action="store_true",
                        help="标定后直接写入配置文件，不询问")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示建议值，绝不写入配置文件")
    args = parser.parse_args()
    if args.write and args.dry_run:
        parser.error("--write and --dry-run cannot be used together")
    if args.duration <= 0 or args.pinch_duration <= 0:
        parser.error("--duration and --pinch-duration must be positive")
    if args.pose_delay < 0:
        parser.error("--pose-delay must be >= 0")
    if args.pinch_max_gap <= 0:
        parser.error("--pinch-max-gap must be positive")
    if args.pinch_weight < 0 or args.scale_prior_weight <= 0:
        parser.error("--pinch-weight must be >= 0 and --scale-prior-weight must be > 0")
    if not 0 <= args.max_tip_scale_change < 1:
        parser.error("--max-tip-scale-change must be in [0, 1)")
    if not 0 <= args.min_distal_ratio <= 1:
        parser.error("--min-distal-ratio must be in [0, 1]")

    # Resolve config path
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    config_dir = INPUT_TO_CONFIG_DIR[args.input]
    config_path = args.config if args.config \
        else f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path

    print(f"配置文件: {config_file}")
    print(f"输入源:   {args.input}")
    print(f"机器人:   {robot_file}")
    print(f"标定模式: {args.calibration_mode}")
    if args.input == "pico4":
        if args.pico4_mode == "direct":
            print(f"Pico4模式: direct (tcp={args.pico4_port}, udp_broadcast={args.pico4_broadcast_port})")
        else:
            print(f"Pico4模式: relay ({args.pico4_relay_host}:{args.pico4_relay_port})")

    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    fi_names = [FINGER_NAMES[i] for i in optimizer.mp_finger_indices]
    configured_scaling = None
    if hasattr(optimizer, "segment_scaling"):
        configured_scaling = np.asarray(optimizer.segment_scaling, dtype=np.float64).copy()
    elif hasattr(optimizer, "_task_kp_indices") and hasattr(optimizer, "_vector_scalings"):
        configured_scaling = np.ones((nf, 3), dtype=np.float64)
        for task_kp, scale in zip(optimizer._task_kp_indices, optimizer._vector_scalings):
            mapping = _KP_TO_FINGER_LEVEL.get(int(task_kp))
            if mapping is None:
                continue
            finger_name, level = mapping
            if finger_name in fi_names:
                configured_scaling[fi_names.index(finger_name), level] = float(scale)

    # ── Robot FK distances ───────────────────────────────────────────────────
    print("\n正在计算机器人 FK 距离（中性位姿）...")
    (pip_robot, dip_robot, tip_robot), (rseg_pip, rseg_dip, rseg_tip) = \
        get_robot_distances(optimizer)

    print(f"\n{'='*68}")
    print(f"机器人 FK 距离（中性位姿）")
    print(f"{'='*68}")
    print(f"  累计距离（origin→各关节）:")
    print(f"  {'手指':8s}  {'→link3':>8s}  {'→link4':>8s}  {'→tip':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(pip_robot[i]):>8s}  "
              f"{fmt_cm(dip_robot[i]):>8s}  {fmt_cm(tip_robot[i]):>8s}")

    print(f"\n  逐段距离:")
    print(f"  {'手指':8s}  {'o→L3':>8s}  {'L3→L4':>8s}  {'L4→tip':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(rseg_pip[i]):>8s}  "
              f"{fmt_cm(rseg_dip[i]):>8s}  {fmt_cm(rseg_tip[i]):>8s}")

    # ── Create input device ──────────────────────────────────────────────────
    input_device = create_input_device(args)

    # ── Preview & wait for user ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    _wait_for_capture_start(
        input_device,
        "请自然伸直并张开所有手指。不要握拳；按 Enter/空格/s 后有 "
        f"{args.pose_delay:g} 秒调整姿势，再采集 {args.duration:.0f} 秒...",
    )
    _pose_countdown(input_device, args.pose_delay)
    print("开始采集张手姿态...")

    # ── Collect ──────────────────────────────────────────────────────────────
    frames, cumulative, segment = collect_human_distances(
        input_device, retargeter, args.hand, args.duration
    )

    if frames == 0:
        print("未收到有效数据，请检查输入设备。")
        return

    pip_human, dip_human, tip_human = cumulative
    hseg_pip, hseg_dip, hseg_tip = segment

    print(f"张手采集完成，共 {frames} 帧")

    pinch_samples = {}
    if args.calibration_mode == "functional":
        print(f"\n{'='*68}")
        print("开始任务标定：依次用拇指与四指指腹自然对指。")
        print("这里标定的是绿色目标骨架的接触保持，不要求握紧或用力挤压。")
        max_gap_m = args.pinch_max_gap / 100.0
        for finger_index in range(1, nf):
            finger_name = fi_names[finger_index]
            _wait_for_capture_start(
                input_device,
                f"请准备拇指-{finger_name}自然对指；按 Enter/空格/s 后有 "
                f"{args.pose_delay:g} 秒调整姿势，再采集 {args.pinch_duration:.0f} 秒...",
            )
            _pose_countdown(input_device, args.pose_delay)
            sample = collect_pinch_vectors(
                input_device,
                retargeter,
                args.hand,
                finger_index,
                args.pinch_duration,
                max_gap_m,
            )
            pinch_samples[finger_index] = sample
            if sample is None:
                print(f"  警告：{finger_name} 没有满足距离阈值的有效帧，将保留张手初值。")
            else:
                print(
                    f"  {finger_name}: 有效 {sample['valid_frames']}/{sample['seen_frames']} 帧, "
                    f"原始 gap 中位数 {sample['raw_gap'] * 100:.2f}cm"
                )

    print(f"\n{'='*68}")
    print(f"人手输入距离（变换后中位数）")
    print(f"{'='*68}")
    print(f"  累计距离（wrist→各关节）:")
    print(f"  {'手指':8s}  {'→PIP':>8s}  {'→DIP':>8s}  {'→TIP':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(pip_human[i]):>8s}  "
              f"{fmt_cm(dip_human[i]):>8s}  {fmt_cm(tip_human[i]):>8s}")

    print(f"\n  逐段距离:")
    print(f"  {'手指':8s}  {'w→PIP':>8s}  {'PIP→DIP':>8s}  {'DIP→TIP':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(hseg_pip[i]):>8s}  "
              f"{fmt_cm(hseg_dip[i]):>8s}  {fmt_cm(hseg_tip[i]):>8s}")

    # ── Compute scaling ──────────────────────────────────────────────────────
    def ratio(robot_d, human_d):
        if robot_d and human_d and human_d > 1e-4:
            return round(robot_d / human_d, 3)
        return 1.0

    open_result = {}
    for i, fname in enumerate(fi_names):
        open_result[fname] = [
            ratio(pip_robot[i], pip_human[i]),
            ratio(dip_robot[i], dip_human[i]),
            ratio(tip_robot[i], tip_human[i]),
        ]

    # Raw open-hand ratios are useful diagnostics, but Linker-style fixed palm
    # mounts and the movable human thumb CMC make them unsafe as an automatic
    # overwrite.  Functional mode therefore refines the current tuned config by
    # default; --functional-base open explicitly opts into the raw ratios.
    result = {name: list(vals) for name, vals in open_result.items()}
    if (
        args.calibration_mode == "functional"
        and args.functional_base == "current"
        and configured_scaling is not None
    ):
        for i, name in enumerate(fi_names):
            result[name] = [float(v) for v in configured_scaling[i]]
        print(f"\n{'='*68}")
        print("functional 初值使用当前配置（不会用原始张手 ratio 覆盖 PIP/DIP）")
        print(f"{'='*68}")
        for name in fi_names:
            print(f"  {name:8s}: {result[name]}")

    # Functional refinement only changes TIP scales.  PIP/DIP remain tied to
    # the robust open-hand morphology estimate, while TIP scales are solved
    # jointly so the scaled green skeleton preserves opposition contacts.
    initial_tip_scales = np.array([result[name][2] for name in fi_names], dtype=np.float64)
    if args.calibration_mode == "functional" and any(
        sample is not None for sample in pinch_samples.values()
    ):
        refined_tip_scales = refine_tip_scales_for_pinches(
            initial_tip_scales,
            pinch_samples,
            prior_weight=args.scale_prior_weight,
            pinch_weight=args.pinch_weight,
            max_relative_change=args.max_tip_scale_change,
        )

        # Structural guard: never let contact fitting pull a green fingertip
        # behind its DIP in the captured open pose.  Keep at least a fraction of
        # the robot's physical distal-link length in radial reach.
        for i, name in enumerate(fi_names):
            if not tip_human[i] or not dip_human[i]:
                continue
            dip_scale = float(result[name][1])
            min_tip_reach = (
                dip_scale * dip_human[i]
                + args.min_distal_ratio * float(rseg_tip[i] or 0.0)
            )
            min_tip_scale = min_tip_reach / tip_human[i]
            if refined_tip_scales[i] < min_tip_scale:
                print(
                    f"  结构约束: {name} TIP {refined_tip_scales[i]:.4f} → "
                    f"{min_tip_scale:.4f}（防止 TIP 缩到 DIP 后方）"
                )
                refined_tip_scales[i] = min_tip_scale

        print(f"\n{'='*68}")
        print("对指任务修正（仅 TIP scale）")
        print(f"{'='*68}")
        print(f"  {'手指':8s}  {'张手初值':>9s}  {'任务修正':>9s}  {'变化':>9s}")
        for i, name in enumerate(fi_names):
            change = 100.0 * (refined_tip_scales[i] / initial_tip_scales[i] - 1.0)
            print(
                f"  {name:8s}  {initial_tip_scales[i]:>9.4f}  "
                f"{refined_tip_scales[i]:>9.4f}  {change:>+8.1f}%"
            )
            result[name][2] = round(float(refined_tip_scales[i]), 4)

        print("\n  绿色 TIP 对指 gap（修正前 → 修正后）:")
        for finger_index, sample in sorted(pinch_samples.items()):
            if sample is None:
                continue
            before = _scaled_pinch_gap(sample, initial_tip_scales, finger_index)
            after = _scaled_pinch_gap(sample, refined_tip_scales, finger_index)
            print(
                f"  thumb-{fi_names[finger_index]:7s}: "
                f"{before * 100:6.2f}cm → {after * 100:6.2f}cm"
            )

    # Per-segment ratios (for reference)
    seg_result = {}
    for i, fname in enumerate(fi_names):
        seg_result[fname] = [
            ratio(rseg_pip[i], hseg_pip[i]),
            ratio(rseg_dip[i], hseg_dip[i]),
            ratio(rseg_tip[i], hseg_tip[i]),
        ]

    print(f"\n{'='*68}")
    print(f"标定结果")
    print(f"{'='*68}")

    label = "任务约束后的建议值" if args.calibration_mode == "functional" else "累计距离 ratio"
    print(f"\n  segment_scaling（{label}，用于配置文件）:")
    print(f"  {'手指':8s}  {'PIP':>6s}  {'DIP':>6s}  {'TIP':>6s}")
    for fname, vals in result.items():
        print(f"  {fname:8s}  {vals[0]:>6.3f}  {vals[1]:>6.3f}  {vals[2]:>6.3f}")

    print(f"\n  逐段 ratio（参考）:")
    print(f"  {'手指':8s}  {'o→L3':>6s}  {'L3→L4':>6s}  {'L4→tip':>6s}")
    for fname, vals in seg_result.items():
        print(f"  {fname:8s}  {vals[0]:>6.3f}  {vals[1]:>6.3f}  {vals[2]:>6.3f}")

    print(f"\n复制以下内容到配置文件的 segment_scaling 部分:")
    print("  segment_scaling:")
    for fname, vals in result.items():
        print(f"    {fname}: {vals}")

    # ── Write back to config ─────────────────────────────────────────────────
    print(f"\n{'='*68}")
    if args.write:
        print("写入配置文件...")
        write_configs(args, robot_file, result, explicit_config_path=args.config)
    else:
        # Show which files would be written
        if args.config:
            targets_info = [args.config]
        else:
            adaptive_path, vector_path = _resolve_config_paths(args, robot_file)
            targets_info = []
            if args.optimizer in ("adaptive", "both") and adaptive_path.exists():
                targets_info.append(str(adaptive_path))
            if args.optimizer in ("vector", "both") and vector_path.exists():
                targets_info.append(str(vector_path))

        if targets_info:
            print("目标配置文件:")
            for p in targets_info:
                print(f"  {p}")
            if args.dry_run:
                print("dry-run：仅显示建议值，未写入任何配置。")
            else:
                answer = input("\n是否写入？[y=写入 / n=跳过]: ").strip().lower()
                if answer == "y":
                    write_configs(args, robot_file, result, explicit_config_path=args.config)
                else:
                    print("已跳过写入。")
        else:
            print("未找到对应配置文件，跳过写入。")


if __name__ == "__main__":
    main()
