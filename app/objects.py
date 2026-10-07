import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

KINDS = ("circle", "square", "box")

PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red": (60, 60, 235),
    "green": (70, 200, 70),
    "blue": (230, 140, 60),
    "yellow": (60, 210, 240),
    "cyan": (230, 220, 60),
    "magenta": (220, 70, 200),
    "orange": (40, 140, 255),
    "white": (245, 245, 245),
    "purple": (170, 80, 160),
    "pink": (180, 130, 250),
}
COLOR_NAMES = list(PALETTE.keys())

CUBE_FACES = [
    (0, 1, 3, 2),   # front
    (4, 6, 7, 5),   # back
    (0, 4, 5, 1),   # top
    (2, 3, 7, 6),   # bottom
    (0, 2, 6, 4),   # left
    (1, 5, 7, 3),   # right
]


def parse_color(token: str) -> Optional[Tuple[int, int, int]]:
    t = token.strip().lower()
    if t in PALETTE:
        return PALETTE[t]
    if t.startswith("#") and len(t) == 7:
        try:
            r, g, b = int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16)
            return (b, g, r)
        except ValueError:
            return None
    if t.startswith("0x") and len(t) == 8:
        try:
            v = int(t, 16)
            return ((v >> 16) & 255, (v >> 8) & 255, v & 255)  # rrggbb -> approx
        except ValueError:
            return None
    return None


@dataclass
class Shape:
    id: int
    kind: str
    x: float
    y: float
    size: float
    color: Tuple[int, int, int]
    angle: float = 0.0       # in-plane rotation, degrees
    rx: float = 0.0          # box pitch  (degrees)
    ry: float = 0.0          # box yaw    (degrees)
    filled: bool = True

    def radius(self) -> float:
        return self.size * 1.05

    def hit(self, px: float, py: float, slack: float = 0.0) -> bool:
        dx, dy = px - self.x, py - self.y
        if self.kind == "circle":
            return math.hypot(dx, dy) <= self.size + slack
        if self.kind == "square":
            a = math.radians(self.angle)
            ca, sa = math.cos(a), math.sin(a)
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            return abs(lx) <= self.size + slack and abs(ly) <= self.size + slack
        # box: 2D projected extent is a bit larger than the cube half edge
        return math.hypot(dx, dy) <= self.size * 1.55 + slack

    def to_dict(self) -> dict:
        d = asdict(self)
        d["color"] = list(self.color)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Shape":
        d = dict(d)
        d["color"] = tuple(d["color"])
        return Shape(**d)


def cube_vertices(shape: Shape) -> np.ndarray:
    """8 vertices of a unit cube, rotated, returned as (8, 3) float array."""
    s = shape.size
    base = np.array(
        [[-s, -s, -s], [s, -s, -s], [-s, s, -s], [s, s, -s],
         [-s, -s, s], [s, -s, s], [-s, s, s], [s, s, s]],
        np.float64,
    )
    a, b, c = map(math.radians, (shape.angle, shape.ry, shape.rx))
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rx = np.array([[1, 0, 0], [0, math.cos(c), -math.sin(c)], [0, math.sin(c), math.cos(c)]])
    return base @ rz.T @ ry.T @ rx.T


def project_box(shape: Shape, frame_shape) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (screen_points (8,2), depth (8,)) for the cube."""
    h, w = frame_shape[:2]
    verts = cube_vertices(shape)
    depth = verts[:, 2]
    f = max(w, h) * 1.2
    scale = f / np.maximum(f - depth, 1e-3)
    sx = shape.x + verts[:, 0] * scale
    sy = shape.y + verts[:, 1] * scale
    pts = np.stack([sx, sy], axis=1)
    return pts, depth
