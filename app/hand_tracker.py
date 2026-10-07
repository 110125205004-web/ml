from typing import List

import cv2
import numpy as np
import mediapipe as mp

from .gestures import HandPose, build_pose


class HandTracker:
    """Thin wrapper around MediaPipe Hands -> list of HandPose (pixel space)."""

    def __init__(
        self,
        max_hands: int = 2,
        model_complexity: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        self.max_hands = max_hands
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.last_raw: List[np.ndarray] = []

    def process(self, frame_bgr: np.ndarray) -> List[HandPose]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        poses: List[HandPose] = []
        if not result.multi_hand_landmarks:
            self.last_raw = []
            return poses

        for i, hand_lms in enumerate(result.multi_hand_landmarks):
            pts = np.array([(lm.x * w, lm.y * h) for lm in hand_lms.landmark], np.float32)
            label, score = "Right", 0.0
            if result.multi_handedness and i < len(result.multi_handedness):
                cls = result.multi_handedness[i].classification[0]
                label, score = cls.label, float(cls.score)
            poses.append(build_pose(label, score, pts))

        self.last_raw = [p.landmarks for p in poses]
        poses.sort(key=lambda p: p.palm_center[0])  # stable left-to-right order
        return poses

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_landmarks(frame: np.ndarray, pose: HandPose, color=(80, 220, 80)) -> None:
    pts = pose.points.astype(int)
    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(pose.points[4].astype(int)), 6, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(pose.points[8].astype(int)), 6, (255, 120, 0), -1, cv2.LINE_AA)
    cx, cy = int(pose.palm_center[0]), int(pose.palm_center[1])
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
