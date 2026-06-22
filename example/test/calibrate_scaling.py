"""Calibrate segment_scaling for any robot hand and input source.

Collects a few seconds of hand data while the user holds their hand flat
(fingers fully extended), then computes the ratio between robot FK link
distances and measured human bone distances. Outputs recommended
segment_scaling values and optionally writes them back to the config YAML.

Supported inputs: mediapipe (video/camera), noitom, quest3, avp

Usage:
    python test/calibrate_scaling.py --robot sharpa --input mediapipe --video data/right.mp4
    python test/calibrate_scaling.py --robot wuji --input mediapipe
    python test/calibrate_scaling.py --robot wuji --input noitom
    python test/calibrate_scaling.py --robot wuji --input quest3
    python test/calibrate_scaling.py --config config/adaptive/avp/avp_wuji_hand.yaml --input avp
    python test/calibrate_scaling.py --robot wuji --input mediapipe --optimizer both --write
"""

import argparse
import shutil
import sys
import time
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


def _write_adaptive(config_path: Path, result: dict, backup: bool):
    """Update retarget.segment_scaling in an adaptive config YAML."""
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        doc = ryaml.load(f)

    seg = doc["retarget"]["segment_scaling"]
    for fname, vals in result.items():
        if fname in seg:
            # preserve ruamel sequence type to keep inline flow style
            for i, v in enumerate(vals):
                seg[fname][i] = v

    if backup:
        shutil.copy2(config_path, str(config_path) + ".bak")
    with open(config_path, "w") as f:
        ryaml.dump(doc, f)


def _write_vector(config_path: Path, result: dict, backup: bool):
    """Update key_vectors[*].scale in a vector config YAML."""
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

    if backup:
        shutil.copy2(config_path, str(config_path) + ".bak")
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
            print(f"  备份已保存: {path}.bak")
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
        print(f"  备份已保存: {config_path}.bak")
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


def collect_human_distances(input_device, retargeter, hand, duration):
    """Collect input data for `duration` seconds and compute average distances.

    Returns:
        frames: number of valid frames
        cumulative: (pip_mean, dip_mean, tip_mean) — wrist→PIP/DIP/TIP
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

        kp = apply_mediapipe_transformations(raw_kp, hand)
        if retargeter.rotation_xyz:
            kp = retargeter._apply_rotation(kp)

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

    def mean_list(bufs):
        return [float(np.mean(b)) if b else 0.0 for b in bufs]

    cumulative = (mean_list(pip_buf), mean_list(dip_buf), mean_list(tip_buf))
    segment = (mean_list(seg_pip_buf), mean_list(seg_dip_buf), mean_list(seg_tip_buf))
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
        return Pico4()
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
    # Noitom
    parser.add_argument("--noitom-local-ip", default="0.0.0.0")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    # Quest3
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", default="udp", choices=["udp", "tcp"])
    # AVP
    parser.add_argument("--avp-ip", default="192.168.50.127")
    # Write-back options
    parser.add_argument("--optimizer", default="both",
                        choices=["adaptive", "vector", "both"],
                        help="写入哪种配置文件（默认: both）")
    parser.add_argument("--write", action="store_true",
                        help="标定后直接写入配置文件，不询问")
    args = parser.parse_args()

    # Resolve config path
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    config_dir = INPUT_TO_CONFIG_DIR[args.input]
    config_path = args.config if args.config \
        else f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path

    print(f"配置文件: {config_file}")
    print(f"输入源:   {args.input}")
    print(f"机器人:   {robot_file}")

    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    fi_names = [FINGER_NAMES[i] for i in optimizer.mp_finger_indices]

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
    print("请伸直所有手指（手掌朝下，五指张开、伸直）")
    print(f"准备好后按 's' 键开始采集 {args.duration:.0f} 秒（在终端或可视化窗口均可）...")

    # Preview in main thread, poll for keyboard input
    import select
    import sys as _sys
    while True:
        input_device.get_fingers_data()
        # Check terminal stdin (non-blocking)
        ready, _, _ = select.select([_sys.stdin], [], [], 0.0)
        if ready:
            _sys.stdin.readline()
            break
        # Check OpenCV key (works if show_video window is active)
        try:
            import cv2
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == 13 or key == ord(' '):
                break
        except Exception:
            pass
    print("开始采集...")

    # ── Collect ──────────────────────────────────────────────────────────────
    frames, cumulative, segment = collect_human_distances(
        input_device, retargeter, args.hand, args.duration
    )

    if frames == 0:
        print("未收到有效数据，请检查输入设备。")
        return

    pip_human, dip_human, tip_human = cumulative
    hseg_pip, hseg_dip, hseg_tip = segment

    print(f"采集完成，共 {frames} 帧")

    print(f"\n{'='*68}")
    print(f"人手输入距离（变换后平均值）")
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

    result = {}
    for i, fname in enumerate(fi_names):
        result[fname] = [
            ratio(pip_robot[i], pip_human[i]),
            ratio(dip_robot[i], dip_human[i]),
            ratio(tip_robot[i], tip_human[i]),
        ]

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

    print(f"\n  segment_scaling（累计距离 ratio，用于配置文件）:")
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
            print("将写入以下配置文件（含自动备份 .bak）:")
            for p in targets_info:
                print(f"  {p}")
            answer = input("\n是否写入？[y=写入 / n=跳过]: ").strip().lower()
            if answer == "y":
                write_configs(args, robot_file, result, explicit_config_path=args.config)
            else:
                print("已跳过写入。")
        else:
            print("未找到对应配置文件，跳过写入。")


if __name__ == "__main__":
    main()
