"""RealSense camera input device for teleoperation.

Reads live RGB frames from an Intel RealSense camera, runs MediaPipe hand
detection in real time, and outputs hand landmarks in the standard (21, 3) format.

Usage:
    python teleop_sim.py --realsense --hand right
    python teleop_sim.py --realsense --hand right --show-video
"""

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# Reference finger segment lengths (meters) from AVP stereo tracking data.
_REFERENCE_SEGMENT_LENGTHS = {
    'thumb':  [0.0505, 0.0318, 0.0302],
    'index':  [0.0418, 0.0243, 0.0223],
    'middle': [0.0489, 0.0289, 0.0227],
    'ring':   [0.0422, 0.0274, 0.0227],
    'pinky':  [0.0343, 0.0195, 0.0201],
}

# MediaPipe landmark index groups per finger: [MCP, PIP, DIP, TIP]
_FINGER_INDICES = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}


class Realsense:
    """Read live RealSense RGB frames and extract hand landmarks via MediaPipe."""

    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(
        self,
        hand_side: str = "right",
        show_video: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        z_scale: float = 2.5,
        correct_segments: bool = True,
        reference_wrist_to_mid_mcp: float = 0.09,
    ):
        self.hand_side = hand_side.lower()
        self.show_video = show_video
        self.frame_width = width
        self.frame_height = height
        self.z_scale = z_scale
        self.correct_segments = correct_segments
        self._reference_wrist_to_mid_mcp = reference_wrist_to_mid_mcp

        # Initialize RealSense pipeline (RGB stream only)
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe reports handedness from camera view (mirrored)
        self._expected_mp_label = "Left" if self.hand_side == "right" else "Right"

        # Cache last valid landmarks for continuity
        self._last_valid_kp = None
        self._last_valid_raw = None

        print(f"RealSense camera initialized")
        print(f"  Resolution: {width}x{height} @ {fps}fps")
        print(f"  Hand side: {self.hand_side}, z_scale: {self.z_scale}, correct_segments: {self.correct_segments}")

    def get_fingers_data(self) -> dict:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        empty = np.zeros((21, 3), dtype=np.float32)
        if not color_frame:
            return {"left_fingers": empty, "right_fingers": empty}

        frame = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        kp = None
        raw_lm = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_cls in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = hand_cls.classification[0].label
                if label == self._expected_mp_label:
                    kp = self._landmarks_to_array(hand_lm)
                    raw_lm = [(lm.x, lm.y) for lm in hand_lm.landmark]
                    break

            if kp is None and results.multi_hand_landmarks:
                kp = self._landmarks_to_array(results.multi_hand_landmarks[0])
                raw_lm = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]

        if kp is not None:
            kp = self._process_landmarks(kp)
            self._last_valid_kp = kp
            self._last_valid_raw = raw_lm
        else:
            kp = self._last_valid_kp
            raw_lm = self._last_valid_raw

        if self.show_video:
            self._show_video_frame(frame, raw_lm)

        if kp is None:
            return {"left_fingers": empty, "right_fingers": empty}

        result = {"left_fingers": empty.copy(), "right_fingers": empty.copy()}
        result[f"{self.hand_side}_fingers"] = kp
        return result

    def _landmarks_to_array(self, hand_landmarks) -> np.ndarray:
        kp = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        kp[:, 0] *= self.frame_width
        kp[:, 1] *= self.frame_height
        kp[:, 2] *= self.frame_width * self.z_scale
        return kp

    def _process_landmarks(self, kp: np.ndarray) -> np.ndarray:
        kp = kp - kp[0:1, :]
        dist = np.linalg.norm(kp[9])
        if dist < 1e-6:
            return kp
        scale = self._reference_wrist_to_mid_mcp / dist
        kp = kp * scale
        if self.correct_segments:
            kp = self._correct_segment_lengths(kp)
        return kp

    def _correct_segment_lengths(self, kp: np.ndarray) -> np.ndarray:
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

    def _show_video_frame(self, frame: np.ndarray, raw_lm):
        display = frame.copy()
        if raw_lm is not None:
            h, w = display.shape[:2]
            pts = [(int(x * w), int(y * h)) for x, y in raw_lm]
            for s, e in self._HAND_CONNECTIONS:
                cv2.line(display, pts[s], pts[e], (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(display, pt, 4, (0, 0, 255), -1)

        scale = 480 / display.shape[0]
        display = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.putText(display, "RealSense Live",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("RealSense MediaPipe", display)
        cv2.waitKey(1)

    def __del__(self):
        if hasattr(self, "_pipeline"):
            self._pipeline.stop()
        if hasattr(self, "mp_hands"):
            self.mp_hands.close()
        if hasattr(self, "show_video") and self.show_video:
            cv2.destroyAllWindows()
