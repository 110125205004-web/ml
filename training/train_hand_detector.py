#!/usr/bin/env python3
"""OPTIONAL: train your own hand detector (the default app uses MediaPipe's
pre-trained model, so you only need this if you want a custom/robust detector).

Two supported routes, both YOLOv8-first because it is the simplest:

  1) Fine-tune YOLOv8 on a hand dataset (recommended)
       python training/train_hand_detector.py --data data/hand.yaml --epochs 50
  2) Train from a public dataset, e.g. HaGRID (https://github.com/hukenovs/hagrid)
       - export HaGRID images + labels to YOLO format (see make_hagrid_yolo() below)
       - then run the same command

Requires:  pip install ultralytics
"""
import argparse
import os
import sys


def make_hagrid_yolo(src: str, dst: str, classes=("hand",), limit=0):
    """Convert a HaGRID-style folder (images/ + labels/ with keypoint json) to YOLO.

    If your data is already YOLO format (images/train, labels/train) skip this.
    """
    import shutil
    import cv2
    import json
    from pathlib import Path

    src_p, dst_p = Path(src), Path(dst)
    for split in ("train", "val"):
        (dst_p / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_p / "labels" / split).mkdir(parents=True, exist_ok=True)

    ann = list(src_p.rglob("*.json"))
    if not ann:
        sys.exit(f"no annotation .json files under {src}")
    n = 0
    for jf in ann[: limit or None]:
        data = json.loads(jf.read_text())
        img_rel = data.get("image", {}).get("path") or data.get("filename")
        img_path = src_p / img_rel if img_rel else jf.with_suffix(".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for hand in data.get("hands", []):
            bb = hand.get("bbox")
            if not bb:
                continue
            x1, y1, x2, y2 = bb if isinstance(bb, (list, tuple)) else (bb["x1"], bb["y1"], bb["x2"], bb["y2"])
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            boxes.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not boxes:
            continue
        split = "val" if "val" in str(jf) or "test" in str(jf) else "train"
        name = f"{n:06d}"
        shutil.copy(img_path, dst_p / "images" / split / (name + img_path.suffix))
        (dst_p / "labels" / split / (name + ".txt")).write_text("\n".join(boxes) + "\n")
        n += 1
    yaml = dst_p / "hand.yaml"
    yaml.write_text(
        f"path: {dst_p.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: hand\n"
    )
    print(f"wrote {n} samples -> {dst_p} (config {yaml})")
    return str(yaml)


def train(data_yaml: str, epochs: int, imgsz: int, model: str, batch: int, out: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed:  pip install ultralytics")

    yolo = YOLO(model)  # yolov8n.pt is downloaded automatically
    yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=out, name="hand")
    print("best weights:", os.path.join(out, "hand", "weights", "best.pt"))


def export_for_mediapipe(weights: str):
    """Optional: export the trained model and evaluate it like MediaPipe does."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed:  pip install ultralytics")
    model = YOLO(weights)
    metrics = model.val()
    print(metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="YOLO dataset yaml (images/labels layout)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--out", default="runs")
    p.add_argument("--hagrid", default=None, help="convert a HaGRID folder to YOLO first, then train")
    p.add_argument("--hagrid-limit", type=int, default=0)
    p.add_argument("--eval-only", default=None, help="just evaluate these weights")
    args = p.parse_args()

    if args.eval_only:
        export_for_mediapipe(args.eval_only)
        return
    data = args.data
    if args.hagrid:
        data = make_hagrid_yolo(args.hagrid, args.data, limit=args.hagrid_limit)
    train(data, args.epochs, args.imgsz, args.model, args.batch, args.out)


if __name__ == "__main__":
    main()
