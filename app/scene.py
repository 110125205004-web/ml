import json
import math
import os
from typing import List, Optional, Tuple

from .objects import KINDS, PALETTE, Shape, parse_color


class Scene:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shapes: List[Shape] = []
        self._next_id = 1
        self.active_id: Optional[int] = None

    # ------------------------------------------------------------------ create
    def add(
        self,
        kind: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        size: Optional[float] = None,
        color=None,
    ) -> Shape:
        kind = kind.lower()
        if kind in ("cube", "rectangle", "sq"):
            kind = "square" if kind in ("rectangle", "sq") else "box"
        if kind not in KINDS:
            raise ValueError(f"unknown kind '{kind}' (choose from {', '.join(KINDS)})")

        if x is None or y is None:
            x, y = self._free_spot()
        base = min(self.width, self.height)
        if size is None:
            size = base * (0.10 if kind != "box" else 0.08)
        if color is None:
            color = self._cycle_color()
        elif isinstance(color, str):
            parsed = parse_color(color)
            if parsed is None:
                raise ValueError(f"unknown colour '{color}'")
            color = parsed

        shape = Shape(
            id=self._next_id,
            kind=kind,
            x=float(x),
            y=float(y),
            size=float(size),
            color=tuple(int(c) for c in color),
        )
        self._next_id += 1
        self.shapes.append(shape)
        self.active_id = shape.id
        return shape

    def _cycle_color(self) -> Tuple[int, int, int]:
        names = list(PALETTE.keys())
        return PALETTE[names[(len(self.shapes)) % len(names)]]

    def _free_spot(self) -> Tuple[float, float]:
        cols = max(1, int(math.sqrt(len(self.shapes) + 1)) + 1)
        cell_w = self.width / (cols + 1)
        row = (len(self.shapes)) // cols
        col = (len(self.shapes)) % cols
        x = cell_w * (col + 1)
        y = self.height * 0.5 + (row % 3 - 1) * self.height * 0.22
        return min(max(x, 60), self.width - 60), min(max(y, 60), self.height - 60)

    # ------------------------------------------------------------------ query
    def get(self, shape_id: Optional[int]) -> Optional[Shape]:
        if shape_id is None:
            return None
        for s in self.shapes:
            if s.id == shape_id:
                return s
        return None

    @property
    def active(self) -> Optional[Shape]:
        return self.get(self.active_id)

    def ensure_active(self, near: Optional[Tuple[float, float]] = None) -> Optional[Shape]:
        if self.active is not None:
            return self.active
        if not self.shapes:
            return None
        if near is not None:
            self.select_nearest(*near)
        else:
            self.active_id = self.shapes[0].id
        return self.active

    def select(self, shape_id: int) -> Optional[Shape]:
        s = self.get(shape_id)
        if s:
            self.active_id = s.id
        return s

    def select_nearest(self, x: float, y: float, max_dist: Optional[float] = None) -> Optional[Shape]:
        best, best_d = None, None
        for s in self.shapes:
            d = math.hypot(s.x - x, s.y - y)
            if best_d is None or d < best_d:
                best, best_d = s, d
        if best is None:
            return None
        if max_dist is not None and best_d > max_dist:
            return None
        self.active_id = best.id
        return best

    def pick(self, x: float, y: float, slack: float = 25.0) -> Optional[Shape]:
        for s in reversed(self.shapes):
            if s.hit(x, y, slack):
                return s
        return None

    # ------------------------------------------------------------------ mutate
    def move(self, dx: float, dy: float, shape_id: Optional[int] = None) -> None:
        s = self.active if shape_id is None else self.get(shape_id)
        if not s:
            return
        s.x = min(max(s.x + dx, 0), self.width)
        s.y = min(max(s.y + dy, 0), self.height)

    def move_to(self, x: float, y: float, shape_id: Optional[int] = None) -> None:
        s = self.active if shape_id is None else self.get(shape_id)
        if not s:
            return
        s.x = min(max(x, 0), self.width)
        s.y = min(max(y, 0), self.height)

    def rotate(self, deg: float, axis: str = "z", shape_id: Optional[int] = None) -> None:
        s = self.active if shape_id is None else self.get(shape_id)
        if not s:
            return
        if axis == "x":
            s.rx = (s.rx + deg) % 360
        elif axis == "y":
            s.ry = (s.ry + deg) % 360
        else:
            s.angle = (s.angle + deg) % 360

    def scale(self, factor: float, shape_id: Optional[int] = None) -> None:
        s = self.active if shape_id is None else self.get(shape_id)
        if not s:
            return
        s.size = min(max(s.size * factor, 8.0), min(self.width, self.height) * 0.45)

    def set_color(self, color, shape_id: Optional[int] = None) -> bool:
        s = self.active if shape_id is None else self.get(shape_id)
        if not s:
            return False
        if isinstance(color, str):
            parsed = parse_color(color)
            if parsed is None:
                return False
            color = parsed
        s.color = tuple(int(c) for c in color)
        return True

    def delete(self, shape_id: Optional[int] = None) -> Optional[int]:
        sid = shape_id if shape_id is not None else self.active_id
        if sid is None:
            return None
        before = len(self.shapes)
        self.shapes = [s for s in self.shapes if s.id != sid]
        if len(self.shapes) == before:
            return None
        if self.active_id == sid:
            self.active_id = self.shapes[-1].id if self.shapes else None
        return sid

    def clear(self) -> int:
        n = len(self.shapes)
        self.shapes.clear()
        self.active_id = None
        return n

    # ------------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "next_id": self._next_id,
            "active_id": self.active_id,
            "shapes": [s.to_dict() for s in self.shapes],
        }

    def save(self, path: str) -> str:
        path = os.path.expanduser(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "Scene":
        path = os.path.expanduser(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scene = cls(int(data.get("width", 1280)), int(data.get("height", 720)))
        scene._next_id = int(data.get("next_id", 1))
        scene.shapes = [Shape.from_dict(d) for d in data.get("shapes", [])]
        active = data.get("active_id")
        scene.active_id = active if scene.get(active) else (scene.shapes[0].id if scene.shapes else None)
        return scene
