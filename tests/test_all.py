"""Self-contained tests (no pytest needed):  python tests/test_all.py
Also works with pytest:  pytest tests/"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from app import commands as cmd
from app.controller import InteractionController
from app.gestures import Gesture, build_pose, classify, two_hand_state
from app.hand_tracker import HandTracker
from app.objects import KINDS, Shape, parse_color
from app.renderer import draw_background, draw_scene, draw_hud
from app.scene import Scene

W, H = 1280, 720


def make_hand(cx=640, cy=360, fingers=(True, True, True, True), pinch=False, scale=1.0):
    """Synthetic 21x2 landmark set. Fingers point 'up' (smaller y)."""
    pts = np.zeros((21, 2), np.float32)
    wrist = np.array([cx, cy], np.float32)
    pts[0] = wrist
    mcp_y, pip_y, tip_y = cy - 45 * scale, cy - 85 * scale, cy - 125 * scale
    xs = [cx - 35 * scale, cx, cx + 30 * scale, cx + 60 * scale]
    for i, (mcp_i, pip_i, tip_i) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        x = xs[i]
        pts[mcp_i] = (x, mcp_y)
        pts[pip_i] = (x, pip_y)
        if fingers[i]:
            pts[tip_i] = (x, tip_y)
        else:                                     # folded: tip closer to the wrist than the pip
            pts[tip_i] = (x, cy - 40 * scale)
    # thumb: side-ways
    pts[4] = (cx - 90 * scale, cy) if not pinch else (pts[8] + np.array([8 * scale, 4 * scale]))
    return build_pose("Right", 0.99, pts)


def test_gestures():
    assert make_hand(fingers=(True, True, True, True)).gesture == Gesture.PALM
    assert make_hand(fingers=(False, False, False, False)).gesture == Gesture.FIST
    assert make_hand(fingers=(True, False, False, False)).gesture == Gesture.POINT
    assert make_hand(fingers=(True, True, False, False)).gesture == Gesture.PEACE
    assert make_hand(pinch=True).gesture == Gesture.PINCH
    a, b = make_hand(cx=400), make_hand(cx=900)
    st = two_hand_state([a, b])
    assert st is not None and abs(st.distance - 500) < 5
    print("ok  gestures")


def test_scene():
    s = Scene(W, H)
    c = s.add("circle", 200, 200, 80, "red")
    q = s.add("square", 400, 200, 60, "green")
    b = s.add("box", 700, 300, 50, "#3366ff")
    assert [x.kind for x in s.shapes] == ["circle", "square", "box"]
    assert b.color == (0xFF, 0x66, 0x33)  # "#3366ff" stored as BGR
    assert s.active_id == b.id
    assert s.pick(200, 200).id == c.id
    assert s.select_nearest(410, 190).id == q.id
    s.move(10, -10)
    assert (q.x, q.y) == (410, 190)
    s.rotate(90)
    assert abs(q.angle - 90) < 1e-6
    s.scale(2.0)
    assert abs(q.size - 120) < 1e-6
    assert c.hit(200, 200) and not c.hit(200 + 90, 200)
    assert q.hit(200, 200 + 80, slack=0) is False  # rotated square: corner region check
    path = os.path.join(tempfile.mkdtemp(), "scene.json")
    s.save(path)
    loaded = Scene.load(path)
    assert len(loaded.shapes) == 3 and loaded.get(b.id).kind == "box"
    assert s.delete(q.id) == q.id and len(s.shapes) == 2
    assert s.clear() == 2 and not s.shapes
    print("ok  scene")


def test_commands():
    s = Scene(W, H)
    ok, msg = cmd.run_command("add circle 120 red", s)
    assert ok and s.shapes[0].kind == "circle" and s.shapes[0].size == 120
    assert cmd.run_command("square 90 cyan", s)[0] and len(s.shapes) == 2
    assert cmd.run_command("rotate 45", s)[0] and s.active.angle == 45
    assert cmd.run_command("rotate y 30", s)[0] and s.active.ry == 30
    assert cmd.run_command("scale 200%", s)[0] and s.active.size == 180
    assert cmd.run_command("color purple", s)[0]
    assert cmd.run_command("move 10 20", s)[0]
    assert cmd.run_command("goto 100 100", s)[0] and (s.active.x, s.active.y) == (100, 100)
    assert cmd.run_command("select 1", s)[0] and s.active_id == 1
    assert cmd.run_command("list", s)[0]
    assert cmd.run_command("save " + os.path.join(tempfile.mkdtemp(), "s.json"), s)[0]
    assert cmd.run_command("delete 1", s)[0] and s.get(1) is None
    assert cmd.run_command("clear", s)[0] and not s.shapes
    assert cmd.run_command("help", s)[0]
    assert not cmd.run_command("add banana", s)[0]
    assert not cmd.run_command("fly 3", s)[0]
    assert not cmd.run_command("scale abc", s)[0]
    assert parse_color("#ff8000") == (0x00, 0x80, 0xFF)
    print("ok  commands")


def test_renderer():
    s = Scene(W, H)
    s.add("circle", 300, 300, 90, "red")
    s.add("square", 640, 300, 90, "green")
    s.add("box", 950, 300, 70, "blue")
    frame = np.zeros((H, W, 3), np.uint8)
    draw_background(frame)
    draw_scene(frame, s)
    draw_hud(frame, s, [], "box", True, "add circle", "message", 30.0, True)
    assert frame.max() > 60, "nothing was drawn"
    assert frame.std() > 5
    print("ok  renderer")


def test_controller():
    s = Scene(W, H)
    ctl = InteractionController(s, palette="box")

    ctl.update([make_hand(cx=640, cy=360, fingers=(True, True, False, False))], 0.033)  # peace
    assert len(s.shapes) == 1, "peace should create"
    ctl.update([make_hand(cx=640, cy=360, fingers=(True, True, False, False))], 0.033)  # still peace
    assert len(s.shapes) == 1, "must debounce repeats"
    ctl.update([make_hand(fingers=(False, False, False, False))], 0.033)                 # fist
    ctl.update([make_hand(cx=700, cy=360, fingers=(False, False, False, False))], 0.033)
    assert abs(s.active.ry) > 0, "fist should rotate the box (yaw)"
    angle_before = s.active.angle
    ry_before = s.active.ry

    ctl.update([make_hand(cx=640, cy=360, pinch=True)], 0.033)
    drag_hand = make_hand(cx=900, cy=500, pinch=True)
    ctl.update([drag_hand], 0.033)
    exp = (drag_hand.pinch_center[0], drag_hand.pinch_center[1])
    assert (s.active.x, s.active.y) == exp, "pinch should drag the object to the pinch point"
    assert abs(s.active.angle - angle_before) < 1e-6 and s.active.ry == ry_before

    ctl.update([], 0.033)
    s.add("circle", 300, 300, 60, "red")
    s.active_id = None
    for _ in range(30):                                            # dwell-select with a pointing finger
        ctl.update([make_hand(cx=300, cy=380, fingers=(True, False, False, False))], 0.033)
    assert s.active is not None and s.active.x == 300, "point should select the object"

    size_before = s.active.size
    ctl.update([make_hand(cx=400, fingers=(True, True, True, True)), make_hand(cx=880, fingers=(True, True, True, True))], 0.033)
    ctl.update([make_hand(cx=300, fingers=(True, True, True, True)), make_hand(cx=980, fingers=(True, True, True, True))], 0.033)
    assert s.active.size != size_before, "two hands should scale"
    print("ok  controller")


def test_hand_tracker_blank():
    with HandTracker(model_complexity=0) as t:
        hands = t.process(np.zeros((480, 640, 3), np.uint8))
    assert hands == []
    print("ok  hand_tracker (blank frame, model loads)")


def test_keys_and_typing():
    from main import handle_keys

    s = Scene(W, H)
    ctl = InteractionController(s)
    state = {"typing": False, "buffer": "", "help": False, "cmd_msg": "", "cmd_ok": True,
             "palette": "box", "controller": ctl}

    assert handle_keys(ord("3"), state, s) is False and s.shapes[-1].kind == "box"
    handle_keys(ord("t"), state, s)
    assert state["typing"] is True
    for ch in "add circle 120 red":
        handle_keys(ord(ch), state, s)
    assert state["buffer"] == "add circle 120 red"
    handle_keys(8, state, s)
    assert state["buffer"] == "add circle 120 re"
    handle_keys(ord("d"), state, s)
    handle_keys(13, state, s)                                   # Enter runs it
    assert not state["typing"] and state["cmd_ok"]
    assert any(sh.kind == "circle" and sh.size == 120 for sh in s.shapes)
    handle_keys(ord("t"), state, s)
    for ch in "add banana":
        handle_keys(ord(ch), state, s)
    handle_keys(13, state, s)
    assert not state["cmd_ok"] and "unknown" in state["cmd_msg"]
    handle_keys(ord("t"), state, s)
    handle_keys(27, state, s)                                   # Esc cancels typing
    assert not state["typing"] and state["buffer"] == ""
    handle_keys(ord("["), state, s)                             # rotate
    handle_keys(ord("]"), state, s)
    handle_keys(ord("-"), state, s)
    handle_keys(ord("+"), state, s)
    handle_keys(ord("\t"), state, s)                            # cycle palette
    assert state["palette"] in KINDS
    assert handle_keys(ord("q"), state, s) is True
    assert handle_keys(27, state, s) is True
    print("ok  keys + typing mode")


def test_main_headless():
    from main import main

    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "canvas.png")
    snap = os.path.join(tmp, "out.png")
    cv2.imwrite(src, np.full((720, 1280, 3), 20, np.uint8))
    main(["--source", src, "--no-display", "--max-frames", "5", "--snapshot", snap,
          "--command", "add circle 150 red", "--command", "add box 100 blue", "--command", "rotate 30"])
    assert os.path.exists(snap) and os.path.getsize(snap) > 1000
    out = cv2.imread(snap)
    assert out.shape == (720, 1280, 3)
    print("ok  main (headless run + snapshot)")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    failed = 0
    for fn in ALL:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
