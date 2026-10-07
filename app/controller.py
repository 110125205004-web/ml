import math
from typing import List, Optional

from .gestures import Gesture, HandPose, two_hand_state
from .objects import KINDS, Shape
from .scene import Scene


class InteractionController:
    """Turns hand gestures into edits on the Scene."""

    MAX_MSG_TIME = 3.0

    def __init__(self, scene: Scene, palette: str = "box"):
        self.scene = scene
        self.palette = palette if palette in KINDS else "box"
        self.message = ""
        self.msg_time = 0.0

        self.last_gesture: Gesture = Gesture.NONE
        self.prev_palm: Optional[tuple] = None
        self.grab_id: Optional[int] = None
        self.palm_drag = False

        self.point_candidate: Optional[int] = None
        self.point_dwell = 0.0
        self.point_locked: Optional[int] = None

        self.two_prev: Optional[tuple] = None
        self.create_lock: Optional[int] = None

    # --------------------------------------------------------------- helpers
    def say(self, text: str) -> None:
        self.message = text
        self.msg_time = self.MAX_MSG_TIME

    def tick(self, dt: float) -> None:
        if self.msg_time > 0:
            self.msg_time -= dt
            if self.msg_time <= 0:
                self.message = ""

    def cycle_palette(self) -> str:
        self.palette = KINDS[(KINDS.index(self.palette) + 1) % len(KINDS)]
        return self.palette

    def _set_active(self, shape: Optional[Shape]) -> None:
        if shape is not None:
            self.scene.active_id = shape.id

    # ---------------------------------------------------------------- update
    def update(self, hands: List[HandPose], dt: float) -> None:
        self.tick(dt)
        if len(hands) >= 2:
            self._two_hands(hands)
            self.last_gesture = Gesture.TWO_HANDS
            return

        if not hands:
            self._reset_drag()
            self.prev_palm = None
            self.two_prev = None
            self.last_gesture = Gesture.NONE
            return

        hand = hands[0]
        g = hand.gesture
        if g != Gesture.PEACE:
            self.create_lock = None

        if g == Gesture.PEACE:
            self._create(hand)
        elif g == Gesture.PINCH:
            self._pinch(hand)
        elif g == Gesture.PALM:
            self._palm(hand, dt)
        elif g == Gesture.FIST:
            self._fist(hand)
        elif g == Gesture.POINT:
            self._point(hand, dt)
        else:
            self._reset_drag()
            self.prev_palm = None

        if g != Gesture.PINCH:
            self.grab_id = None
        if g != Gesture.PALM:
            self.palm_drag = False
        if g != Gesture.POINT:
            self.point_candidate = None
            self.point_dwell = 0.0
            self.point_locked = None

        self.last_gesture = g

    # ------------------------------------------------------------- gestures
    def _create(self, hand: HandPose) -> None:
        """Peace sign -> spawn one object (debounced until the hand changes pose)."""
        if self.create_lock:
            return
        try:
            s = self.scene.add(self.palette)
        except ValueError as e:
            self.say(str(e))
            return
        self.scene.move_to(hand.palm_center[0], hand.palm_center[1] + 40, shape_id=s.id)
        self.say(f"created #{s.id} {s.kind}  (peace sign)")
        self.create_lock = s.id

    def _pinch(self, hand: HandPose) -> None:
        self.prev_palm = None
        px, py = hand.pinch_center
        if self.grab_id is None:
            target = self.scene.pick(px, py, slack=35)
            if target is None:
                target = self.scene.ensure_active((px, py))
            if target is None:
                return
            self.grab_id = target.id
            self.say(f"grabbed #{target.id} {target.kind}")
        s = self.scene.get(self.grab_id)
        if s is not None:
            self.scene.move_to(px, py, shape_id=s.id)

    def _palm(self, hand: HandPose, dt: float) -> None:
        cx, cy = hand.palm_center
        act = self.scene.ensure_active((cx, cy))
        if act is None:
            self.prev_palm = (cx, cy)
            return
        near = math.hypot(cx - act.x, cy - act.y) <= act.radius() + 70
        if near or self.palm_drag:
            if self.prev_palm is not None:
                dx, dy = cx - self.prev_palm[0], cy - self.prev_palm[1]
                self.scene.move(dx, dy, shape_id=act.id)
                self.palm_drag = True
        self.prev_palm = (cx, cy)

    def _fist(self, hand: HandPose) -> None:
        cx, cy = hand.palm_center
        act = self.scene.ensure_active((cx, cy))
        if act is None:
            self.prev_palm = None
            return
        if self.prev_palm is not None:
            dx = cx - self.prev_palm[0]
            dy = cy - self.prev_palm[1]
            if abs(dx) + abs(dy) > 0.5:
                if act.kind == "box":
                    act.ry = (act.ry + dx * 0.55) % 360
                    act.rx = (act.rx + dy * 0.55) % 360
                else:
                    act.angle = (act.angle + dx * 0.55) % 360
        self.prev_palm = (cx, cy)

    def _point(self, hand: HandPose, dt: float) -> None:
        tx, ty = hand.index_tip
        self.prev_palm = None
        target = self.scene.pick(tx, ty, slack=25)
        if target is None:
            self.point_candidate = None
            self.point_dwell = 0.0
            self.point_locked = None
            return
        if self.point_locked == target.id:
            return
        if self.point_candidate == target.id:
            self.point_dwell += dt
        else:
            self.point_candidate = target.id
            self.point_dwell = 0.0
        if self.point_dwell >= 0.35:
            self.scene.active_id = target.id
            self.point_locked = target.id
            self.say(f"selected #{target.id} {target.kind}")

    def _two_hands(self, hands: List[HandPose]) -> None:
        self.grab_id = None
        self.palm_drag = False
        self.prev_palm = None
        st = two_hand_state(hands)
        act = self.scene.ensure_active(st.center)
        if st is None or act is None:
            return
        if self.two_prev is not None:
            prev_dist, prev_ang = self.two_prev
            ratio = st.distance / prev_dist
            ratio = min(max(ratio, 0.92), 1.08)
            if abs(ratio - 1.0) > 0.002:
                self.scene.scale(ratio, shape_id=act.id)

            d_ang = math.degrees(st.angle - prev_ang)
            while d_ang > 180:
                d_ang -= 360
            while d_ang < -180:
                d_ang += 360
            if abs(d_ang) > 0.3:
                if act.kind == "box":
                    act.ry = (act.ry + d_ang) % 360
                else:
                    act.angle = (act.angle + d_ang) % 360
                self.say(f"#{act.id} scaled/rotated with two hands")
        self.two_prev = (st.distance, st.angle)
        if self.create_lock and self.create_lock not in [s.id for s in self.scene.shapes]:
            self.create_lock = None

    def _reset_drag(self) -> None:
        self.grab_id = None
        self.palm_drag = False
        self.point_candidate = None
        self.point_dwell = 0.0
        self.point_locked = None
        self.create_lock = None
