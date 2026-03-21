"""手动控制视频播放，抽取关键帧用于对比优化器性能。

快捷键:
    空格        播放 / 暂停
    D / →       下一帧
    A / ←       上一帧
    W / ↑       快进 10 帧
    S / ↓       快退 10 帧
    E           提取当前帧（保存图片 + landmarks）
    Q / ESC     退出

提取结果保存在 example/output/extracted_frames/ 下:
    frame_0123.png          原始帧图片（带骨架叠加）
    frame_0123_raw.png      原始帧图片（无骨架）
    extracted_landmarks.pkl  所有已提取帧的 landmarks 字典列表
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MediaPipe 手部骨架连接
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 与 video.py 保持一致的参考值
_REFERENCE_WRIST_TO_MIDDLE_MCP = 0.092
_REFERENCE_SEGMENT_LENGTHS = {
    "thumb":  [0.0505, 0.0318, 0.0302],
    "index":  [0.0418, 0.0243, 0.0223],
    "middle": [0.0489, 0.0289, 0.0227],
    "ring":   [0.0422, 0.0274, 0.0227],
    "pinky":  [0.0343, 0.0195, 0.0201],
}
_FINGER_INDICES = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}


def _correct_segment_lengths(kp: np.ndarray) -> np.ndarray:
    kp_corrected = kp.copy()
    for finger_name, indices in _FINGER_INDICES.items():
        ref_lengths = _REFERENCE_SEGMENT_LENGTHS[finger_name]
        mcp_i, pip_i, dip_i, tip_i = indices
        base = kp_corrected[mcp_i].copy()

        seg1 = kp[pip_i] - kp[mcp_i]
        seg1_len = np.linalg.norm(seg1)
        if seg1_len > 1e-6:
            kp_corrected[pip_i] = base + (seg1 / seg1_len) * ref_lengths[0]

        seg2 = kp[dip_i] - kp[pip_i]
        seg2_len = np.linalg.norm(seg2)
        if seg2_len > 1e-6:
            kp_corrected[dip_i] = kp_corrected[pip_i] + (seg2 / seg2_len) * ref_lengths[1]

        seg3 = kp[tip_i] - kp[dip_i]
        seg3_len = np.linalg.norm(seg3)
        if seg3_len > 1e-6:
            kp_corrected[tip_i] = kp_corrected[dip_i] + (seg3 / seg3_len) * ref_lengths[2]
    return kp_corrected


def process_landmarks(kp: np.ndarray, depth_scale: float = 1.25) -> np.ndarray:
    """与 video.py 一致的 landmark 后处理：归一化 + 段长矫正 + 深度缩放。"""
    kp = kp - kp[0:1, :]
    dist = np.linalg.norm(kp[9])
    if dist < 1e-6:
        return kp
    scale = _REFERENCE_WRIST_TO_MIDDLE_MCP / dist
    kp = kp * scale
    kp = _correct_segment_lengths(kp)
    kp[:, 2] *= depth_scale
    return kp


def landmarks_to_array(hand_landmarks, w: int, h: int) -> np.ndarray:
    kp = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    kp[:, 0] *= w
    kp[:, 1] *= h
    kp[:, 2] *= w * 2.5
    return kp


def draw_skeleton(frame: np.ndarray, raw_lm, color=(0, 255, 0)):
    """在帧上绘制手部骨架。"""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in raw_lm]
    for s, e in _HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser(description="手动控制视频播放，提取关键帧")
    example_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", type=str,
                        default=str(example_dir / "data" / "right.mp4"),
                        help="视频文件路径")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="手的方向 (default: right)")
    parser.add_argument("--output", type=str,
                        default=str(example_dir / "output" / "extracted_frames"),
                        help="提取帧的输出目录")
    parser.add_argument("--depth-scale", type=float, default=1.25,
                        help="深度缩放系数 (default: 1.25)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {args.video}")
    print(f"  分辨率: {frame_w}x{frame_h} @ {fps:.1f}fps, 共 {total_frames} 帧")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "extracted_landmarks.pkl"

    # 加载已有的提取记录
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            extracted = pickle.load(f)
        print(f"  已加载 {len(extracted)} 条提取记录")
    else:
        extracted = []
    extracted_indices = {e["frame_idx"] for e in extracted}

    # MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,  # 逐帧独立检测，更稳定
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    expected_label = "Left" if args.hand == "right" else "Right"

    win_name = "Frame Extractor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cur_frame = 0
    playing = False
    need_refresh = True

    def on_trackbar(val):
        nonlocal cur_frame, need_refresh
        cur_frame = val
        need_refresh = True

    cv2.createTrackbar("Frame", win_name, 0, max(total_frames - 1, 0), on_trackbar)

    print("\n--- 快捷键 ---")
    print("  空格: 播放/暂停   D/→: 下一帧   A/←: 上一帧")
    print("  W/↑: +10帧       S/↓: -10帧    E: 提取当前帧")
    print("  Q/ESC: 退出")
    print("-" * 30)

    while True:
        if playing:
            cur_frame = min(cur_frame + 1, total_frames - 1)
            if cur_frame >= total_frames - 1:
                playing = False
            need_refresh = True

        if need_refresh:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"  读取第 {cur_frame} 帧失败")
                need_refresh = False
                continue

            # MediaPipe 检测
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            display = frame.copy()
            raw_lm = None
            kp_processed = None

            if results.multi_hand_landmarks and results.multi_handedness:
                # 优先匹配目标手
                chosen = None
                for hand_lm, hand_cls in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if hand_cls.classification[0].label == expected_label:
                        chosen = hand_lm
                        break
                if chosen is None:
                    chosen = results.multi_hand_landmarks[0]

                raw_lm = [(lm.x, lm.y) for lm in chosen.landmark]
                kp_raw = landmarks_to_array(chosen, frame_w, frame_h)
                kp_processed = process_landmarks(kp_raw, args.depth_scale)
                draw_skeleton(display, raw_lm)

            # 状态信息
            is_extracted = cur_frame in extracted_indices
            status = "PLAYING" if playing else "PAUSED"
            detected = "OK" if raw_lm else "NO HAND"
            mark = " [EXTRACTED]" if is_extracted else ""

            info_lines = [
                f"Frame {cur_frame}/{total_frames - 1} ({cur_frame / max(total_frames - 1, 1) * 100:.1f}%)  |  {status}  |  {detected}{mark}",
                f"Extracted: {len(extracted_indices)} frames  |  E=extract  SPACE=play/pause  Q=quit",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # 缩放显示
            scale = min(960 / display.shape[1], 720 / display.shape[0])
            if scale < 1.0:
                display = cv2.resize(display, None, fx=scale, fy=scale)

            cv2.imshow(win_name, display)
            cv2.setTrackbarPos("Frame", win_name, cur_frame)
            need_refresh = False

        wait_ms = int(1000 / fps) if playing else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == 27:  # Q / ESC
            break
        elif key == ord(" "):  # 空格
            playing = not playing
            need_refresh = True
        elif key == ord("d") or key == 83:  # D / →
            playing = False
            cur_frame = min(cur_frame + 1, total_frames - 1)
            need_refresh = True
        elif key == ord("a") or key == 81:  # A / ←
            playing = False
            cur_frame = max(cur_frame - 1, 0)
            need_refresh = True
        elif key == ord("w") or key == 82:  # W / ↑
            playing = False
            cur_frame = min(cur_frame + 10, total_frames - 1)
            need_refresh = True
        elif key == ord("s") or key == 84:  # S / ↓
            playing = False
            cur_frame = max(cur_frame - 10, 0)
            need_refresh = True
        elif key == ord("e"):  # E = 提取
            if raw_lm is None or kp_processed is None:
                print(f"  [!] 第 {cur_frame} 帧未检测到手部，跳过")
                continue

            # 保存图片
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, raw_frame = cap.read()
            if ret:
                raw_path = out_dir / f"frame_{cur_frame:04d}_raw.png"
                cv2.imwrite(str(raw_path), raw_frame)

                vis_frame = raw_frame.copy()
                draw_skeleton(vis_frame, raw_lm)
                vis_path = out_dir / f"frame_{cur_frame:04d}.png"
                cv2.imwrite(str(vis_path), vis_frame)

                # 记录 landmarks
                entry = {
                    "frame_idx": cur_frame,
                    "landmarks_processed": kp_processed.copy(),
                    "landmarks_raw_pixel": np.array(raw_lm, dtype=np.float32),
                    "hand_side": args.hand,
                }

                if cur_frame in extracted_indices:
                    # 更新已有记录
                    extracted = [e for e in extracted if e["frame_idx"] != cur_frame]
                extracted.append(entry)
                extracted_indices.add(cur_frame)

                # 按帧号排序后保存
                extracted.sort(key=lambda e: e["frame_idx"])
                with open(pkl_path, "wb") as f:
                    pickle.dump(extracted, f)

                print(f"  [v] 提取第 {cur_frame} 帧 -> {vis_path.name}  (共 {len(extracted)} 帧)")
                need_refresh = True

    cap.release()
    mp_hands.close()
    cv2.destroyAllWindows()

    print(f"\n完成! 共提取 {len(extracted)} 帧，保存在 {out_dir}")
    if extracted:
        print(f"  landmarks 文件: {pkl_path}")
        print(f"  帧号列表: {[e['frame_idx'] for e in extracted]}")


if __name__ == "__main__":
    main()
