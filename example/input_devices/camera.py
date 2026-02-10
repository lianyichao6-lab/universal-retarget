"""Camera input device using MediaPipe Hands.

Uses laptop/USB camera with MediaPipe for hand tracking.

Usage:
    python teleop_sim.py --input camera --hand right
"""

import cv2
import numpy as np
import mediapipe as mp


class Camera:
    """使用 OpenCV + MediaPipe 从摄像头获取手部关键点"""

    def __init__(self, camera_id: int = 0, show_preview: bool = True):
        """
        Args:
            camera_id: 摄像头ID (0=默认内置摄像头)
            show_preview: 是否显示预览窗口
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        self.show_preview = show_preview

        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        print(f"摄像头已打开: {camera_id}")
        print(f"预览窗口: {'开启' if show_preview else '关闭'}")

    def get_fingers_data(self) -> dict:
        """获取手部关键点数据，格式与 VisionPro 一致

        Returns:
            dict: {
                "left_fingers": np.ndarray (21, 3),
                "right_fingers": np.ndarray (21, 3),
            }
        """
        ret, frame = self.cap.read()
        if not ret:
            return {
                "left_fingers": np.zeros((21, 3), dtype=np.float32),
                "right_fingers": np.zeros((21, 3), dtype=np.float32),
            }

        # 镜像翻转 (让画面更直观，像照镜子)
        frame = cv2.flip(frame, 1)

        # BGR -> RGB (MediaPipe 需要 RGB 格式)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_fingers = np.zeros((21, 3), dtype=np.float32)
        right_fingers = np.zeros((21, 3), dtype=np.float32)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # 提取 21 个关键点
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                # 坐标转换:
                # MediaPipe 输出: x,y 归一化到 [0,1], z 是相对深度
                # 需要转换为: 以手腕为中心的米制坐标
                scale = 0.2  # 缩放因子，调整手的大小
                landmarks[:, 0] = (landmarks[:, 0] - landmarks[0, 0]) * scale
                landmarks[:, 1] = -(landmarks[:, 1] - landmarks[0, 1]) * scale  # Y轴翻转
                landmarks[:, 2] = -landmarks[:, 2] * scale  # Z轴翻转

                # 判断左右手
                # 注意: 因为镜像翻转，MediaPipe 检测到的 "Left" 实际是用户的右手
                label = handedness.classification[0].label
                if label == "Left":
                    right_fingers = landmarks
                else:
                    left_fingers = landmarks

                # 绘制预览
                if self.show_preview:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

        # 显示预览窗口
        if self.show_preview:
            # 添加提示文字
            cv2.putText(
                frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt("用户按下 q 键退出")

        return {
            "left_fingers": left_fingers,
            "right_fingers": right_fingers,
        }

    def release(self):
        """释放资源"""
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()
