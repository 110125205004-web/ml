import math
from typing import List, Tuple

import cv2
import numpy as np

from .gestures import Gesture, HandPose, gesture_label
from .objects import CUBE_FACES, Shape, project_box
from .scene import Scene

BGR_WHITE = (245, 245, 245)
BGR_DIM = (160, 160, 160)
BGR_ACTIVE = (0, 255, 255)


def _shade(color, intensity: float) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * intensity))) for c in color)


def _draw_circle(frame, s: Shape, active: bool) -> None:
    c = tuple(int(v) for v in s.color)
    thickness = -1 if s.filled else 3
    cv2.circle(frame, (int(s.x), int(s.y)), int(s.size), c, thickness, cv2.LINE_AA)
    cv2.circle(frame, (int(s.x), int(s.y)), int(s.size), (30, 30, 30), 1, cv2.LINE_AA)
    _draw_angle_marker(frame, s, active)


def _draw_square(frame, s: Shape, active: bool) -> None:
    a = math.radians(s.angle)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for lx, ly in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x = s.x + (lx * s.size) * ca - (ly * s.size) * sa
        y = s.y + (lx * s.size) * sa + (ly * s.size) * ca
        pts.append([int(round(x)), int(round(y))])
    pts = np.array(pts, np.int32)
    c = tuple(int(v) for v in s.color)
    cv2.fillPoly(frame, [pts], c, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (30, 30, 30), 2, cv2.LINE_AA)
    _draw_angle_marker(frame, s, active)


def _draw_angle_marker(frame, s: Shape, active: bool) -> None:
    a = math.radians(s.angle)
    r = s.size * 0.95 if s.kind != "box" else s.size * 1.2
    x2 = int(s.x + math.cos(a) * r)
    y2 = int(s.y + math.sin(a) * r)
    cv2.line(frame, (int(s.x), int(s.y)), (x2, y2), BGR_ACTIVE if active else BGR_DIM, 2, cv2.LINE_AA)
    cv2.circle(frame, (x2, y2), 5, BGR_ACTIVE if active else BGR_DIM, -1, cv2.LINE_AA)


def _draw_box(frame, s: Shape, active: bool) -> None:
    pts, depth = project_box(s, frame.shape)
    order = np.argsort(depth)  # far (small z) first
    light = np.array([0.35, -0.45, 0.82])
    light = light / np.linalg.norm(light)

    verts3d = pts
    for face in CUBE_FACES:
        idx = list(face)
        p0, p1, p2 = (verts3d[idx[0]], verts3d[idx[1]], verts3d[idx[2]])
        n = np.cross(
            np.array([p1[0] - p0[0], p1[1] - p0[1], depth[idx[1]] - depth[idx[0]]]),
            np.array([p2[0] - p0[0], p2[1] - p0[1], depth[idx[2]] - depth[idx[0]]]),
        )
        norm = np.linalg.norm(n)
        ndot = 0.0
        if norm > 1e-6:
            n = n / norm
            if n[2] < 0:
                n = -n
            ndot = float(np.dot(n, light))
        shade = 0.45 + 0.55 * max(0.0, ndot)
        poly = np.array([[int(verts3d[i][0]), int(verts3d[i][1])] for i in idx], np.int32)
        cv2.fillPoly(frame, [poly], _shade(s.color, shade), cv2.LINE_AA)
    for face in CUBE_FACES:
        poly = np.array([[int(verts3d[i][0]), int(verts3d[i][1])] for i in face], np.int32)
        cv2.polylines(frame, [poly], True, (25, 25, 25), 1, cv2.LINE_AA)
    _draw_angle_marker(frame, s, active)


def _draw_selection(frame, s: Shape) -> None:
    pad = int(s.radius()) + 12
    x1, y1 = int(s.x) - pad, int(s.y) - pad
    x2, y2 = int(s.x) + pad, int(s.y) + pad
    step = 10
    for x in range(x1, x2, step * 2):
        cv2.line(frame, (x, y1), (min(x + step, x2), y1), BGR_ACTIVE, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y2), (min(x + step, x2), y2), BGR_ACTIVE, 2, cv2.LINE_AA)
    for y in range(y1, y2, step * 2):
        cv2.line(frame, (x1, y), (x1, min(y + step, y2)), BGR_ACTIVE, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y), (x2, min(y + step, y2)), BGR_ACTIVE, 2, cv2.LINE_AA)
    label = f"#{s.id} {s.kind}"
    if s.kind == "box":
        label += f"  yaw{int(s.ry)%360:3d} pit{int(s.rx)%360:3d}"
    else:
        label += f"  rot{int(s.angle)%360:3d}"
    _text(frame, label, (x1, max(18, y1 - 8)), BGR_ACTIVE, 0.55, 2)


def _text(frame, txt, org, color, scale=0.55, thickness=1):
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_background(frame) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (24, 24, 28), -1)
    for x in range(0, w, 60):
        cv2.line(frame, (x, 0), (x, h), (34, 34, 40), 1, cv2.LINE_AA)
    for y in range(0, h, 60):
        cv2.line(frame, (0, y), (w, y), (34, 34, 40), 1, cv2.LINE_AA)


def draw_scene(frame, scene: Scene) -> None:
    for s in scene.shapes:
        active = s.id == scene.active_id
        if s.kind == "circle":
            _draw_circle(frame, s, active)
        elif s.kind == "square":
            _draw_square(frame, s, active)
        else:
            _draw_box(frame, s, active)
        _text(frame, str(s.id), (int(s.x) - 8, int(s.y) + 5), (255, 255, 255), 0.5, 2)
    act = scene.active
    if act is not None:
        _draw_selection(frame, act)


def draw_hands(frame, hands: List[HandPose]) -> None:
    from .hand_tracker import draw_landmarks

    for h in hands:
        color = {
            Gesture.PINCH: (0, 160, 255),
            Gesture.PALM: (80, 220, 80),
            Gesture.FIST: (60, 60, 255),
            Gesture.POINT: (255, 200, 80),
            Gesture.PEACE: (255, 120, 255),
        }.get(h.gesture, (180, 180, 180))
        draw_landmarks(frame, h, color)
        cx, cy = int(h.pinch_center[0]), int(h.pinch_center[1])
        if h.gesture == Gesture.PINCH:
            cv2.circle(frame, (cx, cy), 16, (0, 160, 255), 2, cv2.LINE_AA)


def draw_hud(
    frame,
    scene: Scene,
    hands: List[HandPose],
    palette_kind: str,
    typing: bool,
    command_buffer: str,
    message: str,
    fps: float,
    help_visible: bool,
) -> None:
    h, w = frame.shape[:2]
    panel = np.zeros((150, 470, 3), np.uint8)
    panel[:] = (18, 18, 22)
    frame[8:158, 8:478] = cv2.addWeighted(frame[8:158, 8:478], 0.25, panel, 0.75, 0)

    lines = [
        f"objects: {len(scene.shapes)}   active: "
        + (f"#{scene.active.id} {scene.active.kind}" if scene.active else "-"),
        "create (hand): PEACE sign   create (key): 1=circle 2=square 3=box",
        "move: OPEN PALM / PINCH drag    rotate: FIST (move hand)",
        "select: POINT at object   scale+rotate: TWO HANDS pull/push & tilt",
    ]
    g = hands[0].gesture if hands else Gesture.NONE
    if len(hands) >= 2:
        g = Gesture.TWO_HANDS
    gl = gesture_label(g)
    lines.append(f"gesture: {gl if gl else 'no hand'}")
    lines.append(f"palette: {palette_kind}    fps: {fps:4.1f}")

    y = 30
    for i, ln in enumerate(lines):
        col = BGR_ACTIVE if i == 4 else BGR_WHITE
        _text(frame, ln, (18, y), col, 0.52, 1 if i != 4 else 2)
        y += 22

    if message:
        _text(frame, message, (18, h - 60), (80, 220, 255), 0.6, 2)

    if typing:
        cv2.rectangle(frame, (8, h - 46), (w - 8, h - 8), (40, 40, 50), -1)
        _text(frame, "> " + command_buffer + "_", (18, h - 18), (0, 255, 180), 0.7, 2)

    if help_visible:
        _draw_help(frame)


HELP_LINES = [
    "KEYBOARD",
    "  t         type a command (Enter=run, Esc=cancel)",
    "  1/2/3     create circle / square / box",
    "  [ ]       rotate active object   - / +  scale",
    "  arrows    nudge active object    Tab  change palette",
    "  h         toggle help     s=save scene.json   q/Esc quit",
    "",
    "TYPING COMMANDS",
    "  add circle|square|box [size] [color]",
    "  select <id> | move <dx> <dy> | goto <x> <y>",
    "  rotate <deg> | rotate x|y <deg> | scale <factor>",
    "  color <name> | delete | clear | list | save [file] | help",
]


def _draw_help(frame) -> None:
    h, w = frame.shape[:2]
    box_w, box_h = 640, len(HELP_LINES) * 24 + 30
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), BGR_ACTIVE, 1, cv2.LINE_AA)
    y = y1 + 26
    for ln in HELP_LINES:
        col = BGR_ACTIVE if ln.isupper() and ln.strip() else BGR_WHITE
        _text(frame, ln, (x1 + 14, y), col, 0.5, 1)
        y += 24
