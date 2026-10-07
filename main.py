#!/usr/bin/env python3
"""Hand Object Studio - create and manipulate shapes with hand gestures + typed commands.

Run:  python main.py                                  (webcam)
      python main.py --camera 1                       (another webcam)
      python main.py --command "add circle 120 red"   (spawn from the start)
      python main.py --source clip.mp4 --no-display --max-frames 100 --snapshot out.png
"""
import argparse
import os
import sys
import time

import cv2

from app import commands as cmd
from app.controller import InteractionController
from app.hand_tracker import HandTracker
from app.objects import KINDS
from app.renderer import draw_background, draw_hands, draw_hud, draw_scene
from app.scene import Scene

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
ARROW_KEYS = {65361: (-1, 0), 81: (-1, 0), 65362: (0, -1), 82: (0, -1),
              65363: (1, 0), 83: (1, 0), 65364: (0, 1), 84: (0, 1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Create box / square / circle objects with hand gestures.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--source", type=str, default=None, help="video/image file instead of the webcam")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--palette", type=str, default="box", choices=list(KINDS),
                   help="shape created by the PEACE gesture and by key 3 / Tab")
    p.add_argument("--command", action="append", default=[], metavar="CMD",
                   help="run a typed command at start (repeatable)")
    p.add_argument("--load", type=str, default=None, help="load a scene.json")
    p.add_argument("--no-display", action="store_true", help="headless mode (no GUI window)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = forever)")
    p.add_argument("--snapshot", type=str, default=None, help="save the last frame to this image")
    p.add_argument("--record", type=str, default=None, help="record the session to this .mp4")
    p.add_argument("--hide-landmarks", action="store_true", help="do not draw the hand skeleton")
    return p.parse_args(argv)


def open_source(args):
    if args.source:
        is_image = os.path.splitext(args.source)[1].lower() in IMAGE_EXTS
        if is_image:
            img = cv2.imread(args.source)
            if img is None:
                sys.exit(f"could not read image: {args.source}")
            return None, img, True
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            sys.exit(f"could not open source: {args.source}")
        return cap, None, False

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"could not open camera {args.camera}. Use --source <file> to run without a camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap, None, False


def handle_keys(key, state, scene):
    """Return True when the app should quit."""
    if key is None or key < 0:
        return False

    if state["typing"]:
        if key in (13, 10):                                   # Enter
            ok, msg = cmd.run_command(state["buffer"], scene)
            state["cmd_ok"], state["cmd_msg"] = ok, msg
            if ok and msg.startswith("palette "):
                state["palette"] = msg.split()[1]
            state["buffer"] = ""
            state["typing"] = False
        elif key == 27:                                       # Esc cancels typing
            state["buffer"] = ""
            state["typing"] = False
        elif key in (8, 127):                                 # Backspace
            state["buffer"] = state["buffer"][:-1]
        elif 32 <= key < 127:
            state["buffer"] += chr(key)
        return False

    if key in ARROW_KEYS:
        dx, dy = ARROW_KEYS[key]
        scene.move(dx * 15, dy * 15)
        return False

    k = key & 0xFF
    if key in (27,) or k in (ord("q"), ord("Q")):
        return True
    if k == ord("t"):
        state["typing"] = True
    elif k == ord("h"):
        state["help"] = not state["help"]
    elif k == ord("s"):
        state["cmd_ok"], state["cmd_msg"] = True, "saved " + scene.save("scene.json")
    elif k in (ord("1"), ord("2"), ord("3")):
        kind = {"1": "circle", "2": "square", "3": "box"}[chr(k)]
        scene.add(kind)
        state["palette"] = kind
        state["cmd_ok"], state["cmd_msg"] = True, f"created {kind}"
    elif k == ord("\t") or k == ord("p"):
        state["palette"] = state["controller"].cycle_palette()
        state["cmd_ok"], state["cmd_msg"] = True, "palette -> " + state["palette"]
    elif k == ord("["):
        scene.rotate(-15)
    elif k == ord("]"):
        scene.rotate(15)
    elif k in (ord("-"), ord("_")):
        scene.scale(0.9)
    elif k in (ord("+"), ord("=")):
        scene.scale(1.1)
    elif k in (8, 127, ord("x"), ord("X")):
        removed = scene.delete()
        state["cmd_ok"] = bool(removed)
        state["cmd_msg"] = f"deleted #{removed}" if removed else "nothing to delete"
    return False


def main(argv=None):
    args = parse_args(argv)
    display = not args.no_display
    mirror = not args.no_mirror

    cap, image, is_image = open_source(args)
    frame = image if image is not None else None
    if frame is None:
        ok, frame = cap.read()
        if not ok or frame is None:
            sys.exit("could not read a frame from the source")

    h, w = frame.shape[:2]
    scene = Scene.load(args.load) if args.load else Scene(w, h)
    scene.width, scene.height = w, h
    controller = InteractionController(scene, palette=args.palette)

    state = {"typing": False, "buffer": "", "help": False, "cmd_msg": "", "cmd_ok": True,
             "palette": args.palette, "controller": controller}

    for c in args.command:
        ok, msg = cmd.run_command(c, scene)
        state["cmd_ok"], state["cmd_msg"] = ok, msg
        print(("[ok]  " if ok else "[err] ") + msg)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        if not writer.isOpened():
            sys.exit(f"could not open video writer {args.record}")

    window = "Hand Object Studio"
    if display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    tracker = HandTracker(max_hands=args.max_hands, model_complexity=1)
    fps, t_prev, frame_i, last_frame, quit_requested = 0.0, time.time(), 0, frame, False
    recent_dts = []
    print("keys: t=type  1/2/3=create  Tab=palette  arrows=nudge  h=help  s=save  q=quit")

    try:
        while not quit_requested:
            if not is_image:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if args.source:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # loop videos
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                    else:
                        print("camera frame grab failed, stopping.")
                        break
            if mirror:
                frame = cv2.flip(frame, 1)
            last_frame = frame
            frame_i += 1

            now = time.time()
            dt = max(1e-3, now - t_prev)
            t_prev = now
            recent_dts.append(dt)
            del recent_dts[:-30]
            fps = len(recent_dts) / max(sum(recent_dts), 1e-6)

            hands = tracker.process(frame)
            controller.update(hands, dt)

            draw_background(frame)
            draw_scene(frame, scene)
            if not args.hide_landmarks:
                draw_hands(frame, hands)

            if display:
                quit_requested = handle_keys(cv2.waitKeyEx(1), state, scene)

            draw_hud(
                frame, scene, hands, state["palette"],
                typing=state["typing"],
                command_buffer=state["buffer"],
                message=controller.message or state["cmd_msg"],
                fps=fps,
                help_visible=state["help"],
            )
            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow(window, frame)

            if args.max_frames and frame_i >= args.max_frames:
                break
            if is_image and frame_i >= (args.max_frames or 60):
                break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if args.snapshot:
            cv2.imwrite(args.snapshot, last_frame)
            print("snapshot ->", os.path.abspath(args.snapshot))
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
