"""对指定帧运行 retargeting，计算 loss 并渲染 MuJoCo 仿真图片。

用法:
    # 默认使用 shadow_hand 配置，测试 7 个帧
    python test/benchmark_frames.py

    # 指定配置 & 帧号
    python test/benchmark_frames.py --config config/shadow_hand.yaml
    python test/benchmark_frames.py --config config/wuji_hand.yaml
    python test/benchmark_frames.py --frames 40 80 215

    # 同时跑多个配置做对比
    python test/benchmark_frames.py --configs config/shadow_hand.yaml config/wuji_hand.yaml

输出目录结构 (example/output/benchmark/):
    shadow_hand/
        frame_0040_sim.png      MuJoCo 仿真渲染
        frame_0040_input.png    原始视频帧 + MediaPipe 骨架
        ...
    results.pkl                 完整结果 (landmarks, qpos, loss, timing)
    summary.txt                 汇总表格
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# 设置 headless 渲染
if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mediapipe as mp
import mujoco
import numpy as np
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from qsq_retargeting import Retargeter
from qsq_retargeting.mediapipe import apply_mediapipe_transformations

# 默认测试帧
DEFAULT_FRAMES = [40, 80, 215, 304, 417, 582, 642]

# ── 与 video.py 一致的 landmark 预处理 ──────────────────────────────
_REFERENCE_WRIST_TO_MIDDLE_MCP = 0.092
_REFERENCE_SEGMENT_LENGTHS = {
    "thumb": [0.0505, 0.0318, 0.0302],
    "index": [0.0418, 0.0243, 0.0223],
    "middle": [0.0489, 0.0289, 0.0227],
    "ring": [0.0422, 0.0274, 0.0227],
    "pinky": [0.0343, 0.0195, 0.0201],
}
_FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _correct_segment_lengths(kp):
    kp_c = kp.copy()
    for name, idx in _FINGER_INDICES.items():
        ref = _REFERENCE_SEGMENT_LENGTHS[name]
        mcp, pip, dip, tip = idx
        base = kp_c[mcp].copy()
        for i, (a, b, rl) in enumerate([(mcp, pip, ref[0]), (pip, dip, ref[1]), (dip, tip, ref[2])]):
            seg = kp[b] - kp[a]
            n = np.linalg.norm(seg)
            if n > 1e-6:
                kp_c[b] = (base if i == 0 else kp_c[a]) + seg / n * rl
    return kp_c


def process_landmarks(kp, depth_scale=1.25):
    kp = kp - kp[0:1]
    d = np.linalg.norm(kp[9])
    if d < 1e-6:
        return kp
    kp = kp * (_REFERENCE_WRIST_TO_MIDDLE_MCP / d)
    kp = _correct_segment_lengths(kp)
    kp[:, 2] *= depth_scale
    return kp


def landmarks_to_array(hand_landmarks, w, h):
    kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    kp[:, 0] *= w
    kp[:, 1] *= h
    kp[:, 2] *= w * 2.5
    return kp


# ── 与 teleop_sim.py 保持一致的机器人手配置 ──────────────────────────

ROBOT_HAND_CONFIGS = {
    "shadow_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "shadow_hand" / f"scene_{side}.xml"),
        "needs_menagerie_mapping": True,
    },
    "wuji_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "wuji_hand" / "right.xml"),
    },
    "allegro_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "allegro_hand" / "scene_right.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "inspire_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "inspire_hand" / "inspire_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "ability_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "ability_hand" / "ability_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "leap_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "leap_hand" / "leap_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "svh_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "schunk_hand" / "schunk_svh_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 13, 14, 15, 16, 9, 10, 11, 12, 4, 5, 6, 7, 17, 18, 19],
    },
    "linkerhand_l21": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linkerhand_l21" / f"linkerhand_l21_{side}_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16],
        "qpos_servo_alpha": 0.2,
    },
    "rohand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "rohand" / f"rohand_{side}_mujoco.xml"),
        "qpos_mapping": [3, 4, 1, 2, 0, 13, 14, 11, 12, 10, 18, 19, 16, 17, 15, 8, 9, 6, 7, 5, 20, 21, 23, 24, 22],
        "qpos_servo_alpha": 0.18,
        "base_quat": (0.7071068, 0, 0.7071068, 0),
    },
    "unitree_dex5_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "unitree_dex5_hand" / f"unitree_dex5_hand_{side}_mujoco.xml"),
        "qpos_mapping": [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "qpos_servo_alpha": 0.2,
        "base_quat": (0.7071068, 0.7071068, 0, 0),
    },
}


def map_urdf_to_mujoco_menagerie(qpos):
    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[0], ctrl[1] = 0.0, 0.0
    ctrl[2] = qpos[17]; ctrl[3] = qpos[18]; ctrl[4] = qpos[19]
    ctrl[5] = qpos[20]; ctrl[6] = qpos[21]
    ctrl[7] = qpos[0]; ctrl[8] = qpos[1]; ctrl[9] = qpos[2] + qpos[3]
    ctrl[10] = qpos[9]; ctrl[11] = qpos[10]; ctrl[12] = qpos[11] + qpos[12]
    ctrl[13] = qpos[13]; ctrl[14] = qpos[14]; ctrl[15] = qpos[15] + qpos[16]
    ctrl[16] = qpos[4]; ctrl[17] = qpos[5]; ctrl[18] = qpos[6]; ctrl[19] = qpos[7] + qpos[8]
    return ctrl


# ── 每种手的正面渲染相机参数 ─────────────────────────────────────────
FRONT_VIEW_CAMERAS = {
    "shadow_hand":      {"azimuth": 180, "elevation": -15, "distance": 0.8, "lookat": [0, 0, 0.15]},
    "wuji_hand":        {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "allegro_hand":     {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "inspire_hand":     {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "ability_hand":     {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "leap_hand":        {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "svh_hand":         {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "linkerhand_l21":   {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "rohand":           {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
    "unitree_dex5_hand":{"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]},
}


def render_mujoco_frame(model, data, cam_cfg, width=1280, height=720):
    """离屏渲染一帧，返回 BGR 图像。"""
    # 不超过模型的 offscreen framebuffer 尺寸
    max_w = int(model.vis.global_.offwidth)
    max_h = int(model.vis.global_.offheight)
    width = min(width, max_w)
    height = min(height, max_h)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = cam_cfg.get("azimuth", 180)
    cam.elevation = cam_cfg.get("elevation", -15)
    cam.distance = cam_cfg.get("distance", 0.5)
    cam.lookat[:] = cam_cfg.get("lookat", [0, 0, 0.05])
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.close()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def apply_qpos_to_mujoco(model, data, qpos, hand_cfg):
    """将 retarget 输出的 qpos 映射到 MuJoCo 并步进仿真。"""
    if hand_cfg.get("needs_menagerie_mapping"):
        ctrl = map_urdf_to_mujoco_menagerie(qpos)
    elif "qpos_mapping" in hand_cfg:
        ctrl = qpos[hand_cfg["qpos_mapping"]]
    else:
        ctrl = qpos
    ctrl = np.asarray(ctrl, dtype=np.float32)

    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    if qpos_servo_alpha is not None:
        # servo 模式：直接设置 qpos
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif model.nu > 0:
        # actuator 模式
        n = min(len(ctrl), model.nu)
        data.ctrl[:n] = ctrl[:n]
        for _ in range(200):
            mujoco.mj_step(model, data)
    else:
        # direct qpos 模式
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        mujoco.mj_forward(model, data)


# ── 从视频连续跑 retargeting，在目标帧保存结果 ─────────────────────

def run_benchmark(config_path, video_path, target_frames, hand_side, out_dir, depth_scale=1.25):
    """从头连续跑视频 retargeting，在目标帧保存仿真截图和 loss。

    连续跑保证优化器有 warm start，结果与实时管线一致。
    """
    config_file = EXAMPLE_DIR / config_path
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_type = config.get("robot", {}).get("type", "shadow_hand")
    print(f"\n{'='*60}")
    print(f"配置: {config_path}")
    print(f"机器人: {robot_type}")
    print(f"{'='*60}")

    # 加载 Retargeter
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # 加载 MuJoCo 模型
    hand_cfg = ROBOT_HAND_CONFIGS.get(robot_type)
    mj_model, mj_data = None, None
    if hand_cfg is None:
        print(f"  [!] 未知机器人类型: {robot_type}，跳过 MuJoCo 渲染")
    else:
        model_path = Path(hand_cfg["model_path"](hand_side))
        if not model_path.exists():
            print(f"  [!] MuJoCo 模型不存在: {model_path}，跳过渲染")
        else:
            mj_model = mujoco.MjModel.from_xml_path(str(model_path))
            mj_data = mujoco.MjData(mj_model)

    cam_cfg = FRONT_VIEW_CAMERAS.get(robot_type, {"azimuth": 180, "elevation": -15, "distance": 0.5, "lookat": [0, 0, 0.05]})

    cfg_dir = out_dir / robot_type
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频，从第 0 帧开始连续跑
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = max(target_frames)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    expected_label = "Left" if hand_side == "right" else "Right"

    target_set = set(target_frames)
    per_frame_results = []
    last_valid_kp = None

    print(f"  连续跑帧 0 ~ {max_frame}，在目标帧 {sorted(target_frames)} 保存结果...")

    for fi in range(max_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe 检测
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = mp_hands.process(rgb)

        chosen = None
        if det.multi_hand_landmarks and det.multi_handedness:
            for hlm, hcls in zip(det.multi_hand_landmarks, det.multi_handedness):
                if hcls.classification[0].label == expected_label:
                    chosen = hlm
                    break
            if chosen is None:
                chosen = det.multi_hand_landmarks[0]

        if chosen is not None:
            kp_raw = landmarks_to_array(chosen, w, h)
            kp_proc = process_landmarks(kp_raw, depth_scale)
            last_valid_kp = kp_proc
        else:
            kp_proc = last_valid_kp

        if kp_proc is None:
            continue

        # 连续 retarget（带 warm start + 低通滤波，与实时管线一致）
        t0 = time.perf_counter()
        qpos, verbose = retargeter.retarget_verbose(kp_proc, apply_filter=True)
        solve_ms = (time.perf_counter() - t0) * 1000

        # 在目标帧保存结果
        if fi in target_set:
            cost = verbose["cost"]
            pinch_alphas = verbose.get("pinch_alphas", None)
            timing = retargeter.optimizer.get_timing_stats()
            iter_losses = timing.get_last_iter_losses()
            num_iters = timing.iter_counts[-1] if timing.iter_counts else 0

            print(f"  帧 {fi:4d}: loss={cost:.4f}, iters={num_iters}, time={solve_ms:.1f}ms")

            # 渲染 MuJoCo 仿真
            if mj_model is not None and hand_cfg is not None:
                mujoco.mj_resetData(mj_model, mj_data)
                apply_qpos_to_mujoco(mj_model, mj_data, qpos, hand_cfg)
                sim_img = render_mujoco_frame(mj_model, mj_data, cam_cfg)
                cv2.imwrite(str(cfg_dir / f"frame_{fi:04d}_sim.png"), sim_img)

            per_frame_results.append({
                "frame_idx": fi,
                "qpos": qpos.copy(),
                "cost": cost,
                "solve_ms": solve_ms,
                "num_iters": num_iters,
                "iter_losses": iter_losses,
                "pinch_alphas": pinch_alphas,
                "landmarks": kp_proc.copy(),
                "mediapipe_kp": verbose["mediapipe_kp"].copy(),
            })

    cap.release()
    mp_hands.close()

    return {
        "config_path": config_path,
        "robot_type": robot_type,
        "frames": per_frame_results,
    }


def print_summary(all_results, out_dir):
    """打印并保存汇总表格。"""
    lines = []
    lines.append(f"{'='*80}")
    lines.append("Retargeting Benchmark 汇总")
    lines.append(f"{'='*80}")

    for res in all_results:
        robot = res["robot_type"]
        frames = res["frames"]
        lines.append(f"\n── {robot} ({res['config_path']}) ──")
        lines.append(f"{'帧号':>6s}  {'Loss':>10s}  {'迭代':>5s}  {'耗时(ms)':>10s}")
        lines.append("-" * 40)

        costs = []
        times = []
        iters = []
        for f in frames:
            lines.append(f"{f['frame_idx']:6d}  {f['cost']:10.4f}  {f['num_iters']:5d}  {f['solve_ms']:10.1f}")
            costs.append(f["cost"])
            times.append(f["solve_ms"])
            iters.append(f["num_iters"])

        lines.append("-" * 40)
        lines.append(f"{'平均':>6s}  {np.mean(costs):10.4f}  {np.mean(iters):5.1f}  {np.mean(times):10.1f}")
        lines.append(f"{'最大':>6s}  {np.max(costs):10.4f}  {np.max(iters):5d}  {np.max(times):10.1f}")
        lines.append(f"{'最小':>6s}  {np.min(costs):10.4f}  {np.min(iters):5d}  {np.min(times):10.1f}")

    text = "\n".join(lines)
    print(text)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n汇总已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Retargeting 帧级 Benchmark")
    parser.add_argument("--video", type=str,
                        default=str(EXAMPLE_DIR / "data" / "right.mp4"),
                        help="视频路径")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES,
                        help=f"测试帧号列表 (默认: {DEFAULT_FRAMES})")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["config/shadow_hand.yaml"],
                        help="配置文件列表 (相对于 example/)")
    parser.add_argument("--output", type=str,
                        default=str(EXAMPLE_DIR / "output" / "benchmark"),
                        help="输出目录")
    parser.add_argument("--depth-scale", type=float, default=1.25)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐配置运行 benchmark（每个配置从头连续跑视频）
    all_results = []
    for cfg in args.configs:
        result = run_benchmark(
            cfg, args.video, args.frames, args.hand, out_dir, args.depth_scale)
        all_results.append(result)

        # 每个手单独保存 results.pkl 和 summary.txt
        robot_type = result["robot_type"]
        robot_dir = out_dir / robot_type
        robot_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = robot_dir / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)
        print(f"\n结果已保存: {pkl_path}")

        print_summary([result], robot_dir)

    # 3. 如果跑了多个配置，额外保存一份汇总到顶层
    if len(all_results) > 1:
        all_pkl = out_dir / "all_results.pkl"
        with open(all_pkl, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\n全部结果已保存: {all_pkl}")
        print_summary(all_results, out_dir)


if __name__ == "__main__":
    main()
