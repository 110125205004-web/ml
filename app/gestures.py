import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Gesture(str, Enum):
    NONE = "none"
    PINCH = "pinch"      # thumb + index together  -> grab / drag
    PALM = "palm"        # open palm               -> move active object
    FIST = "fist"        # closed fist             -> rotate active object
    POINT = "point"      # index finger only       -> select object
    PEACE = "peace"      # index + middle finger   -> CREATE new object
    TWO_HANDS = "two_hands"  # handled separately


@dataclass
class HandPose:
    label: str
    score: float
    landmarks: np.ndarray            # (21, 3) normalised, pixel-space below uses shape
    points: np.ndarray               # (21, 2) pixel coords
    palm_center: Tuple[float, float]
    pinch_center: Tuple[float, float]
    size: float                      # wrist -> middle MCP distance in pixels
    gesture: Gesture = Gesture.NONE
    fingers: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    thumb_up: bool = False
    pinch_ratio: float = 1.0

    @property
    def index_tip(self) -> Tuple[float, float]:
        return float(self.points[INDEX_TIP][0]), float(self.points[INDEX_TIP][1])

    @property
    def wrist(self) -> Tuple[float, float]:
        return float(self.points[WRIST][0]), float(self.points[WRIST][1])


def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def build_pose(label: str, score: float, points: np.ndarray) -> HandPose:
    """points: (21, 2) pixel coordinates."""
    wrist = points[WRIST]
    middle_mcp = points[MIDDLE_MCP]
    size = max(_dist(wrist, middle_mcp), 1.0)

    palm_center = (
        float(np.mean([points[WRIST][0], points[INDEX_MCP][0], points[PINKY_MCP][0], points[MIDDLE_MCP][0]])),
        float(np.mean([points[WRIST][1], points[INDEX_MCP][1], points[PINKY_MCP][1], points[MIDDLE_MCP][1]])),
    )
    pinch_center = (
        float((points[THUMB_TIP][0] + points[INDEX_TIP][0]) / 2.0),
        float((points[THUMB_TIP][1] + points[INDEX_TIP][1]) / 2.0),
    )

    fingers = tuple(
        bool(_dist(points[tip], wrist) > _dist(points[pip], wrist) * 1.08)
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
    )
    thumb_up = _dist(points[THUMB_TIP], wrist) > _dist(points[INDEX_MCP], wrist) * 1.05
    pinch_ratio = _dist(points[THUMB_TIP], points[INDEX_TIP]) / size

    pose = HandPose(
        label=label,
        score=score,
        landmarks=np.zeros((21, 3), np.float32),
        points=np.asarray(points, np.float32),
        palm_center=palm_center,
        pinch_center=pinch_center,
        size=size,
        fingers=fingers,  # type: ignore[arg-type]
        thumb_up=thumb_up,
        pinch_ratio=pinch_ratio,
    )
    pose.gesture = classify(pose)
    return pose


def classify(p: HandPose) -> Gesture:
    index, middle, ring, pinky = p.fingers

    if p.pinch_ratio < 0.40:
        return Gesture.PINCH
    if index and middle and not ring and not pinky:
        return Gesture.PEACE
    if index and not middle and not ring and not pinky:
        return Gesture.POINT
    if index and middle and ring and pinky:
        return Gesture.PALM
    if not index and not middle and not ring and not pinky:
        return Gesture.FIST
    return Gesture.NONE


@dataclass
class TwoHandState:
    distance: float
    angle: float            # radians, in image space
    center: Tuple[float, float]


def two_hand_state(hands: Sequence[HandPose]) -> Optional[TwoHandState]:
    if len(hands) < 2:
        return None
    a, b = hands[0], hands[1]
    d = _dist(a.palm_center, b.palm_center)
    ang = math.atan2(b.palm_center[1] - a.palm_center[1], b.palm_center[0] - a.palm_center[0])
    center = (
        (a.palm_center[0] + b.palm_center[0]) / 2.0,
        (a.palm_center[1] + b.palm_center[1]) / 2.0,
    )
    return TwoHandState(distance=max(d, 1.0), angle=ang, center=center)


def gesture_label(g: Gesture) -> str:
    return {
        Gesture.PINCH: "PINCH - grab & drag",
        Gesture.PALM: "OPEN PALM - move object",
        Gesture.FIST: "FIST - rotate object",
        Gesture.POINT: "POINT - select object",
        Gesture.PEACE: "PEACE - create object",
        Gesture.TWO_HANDS: "TWO HANDS - scale / rotate",
        Gesture.NONE: "",
    }.get(g, "")
