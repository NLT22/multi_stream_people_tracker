"""
Convert MTA_ext_short dataset to Ultralytics YOLO format for fine-tuning.

Extracts frames from cam_*.mp4 videos and converts CSV ground-truth annotations
to YOLO .txt label files (one label file per image frame).

Output structure:
    <output-dir>/
        images/train/   cam{C}_frame{F:07d}.jpg
        images/val/     cam{C}_frame{F:07d}.jpg
        labels/train/   cam{C}_frame{F:07d}.txt
        labels/val/     cam{C}_frame{F:07d}.txt
        dataset.yaml

YOLO label format (one line per person):
    0 cx cy w h          (class_id=0, normalized 0-1 relative to 1920×1080)

Run:
    python scripts/mta_to_yolo.py \\
        --mta-root dataset/mta/MTA_ext_short \\
        --output-dir dataset/mta_yolo \\
        --sample-rate 5          # keep every 5th frame (~8 fps)
        [--cameras 0 1 2 3 4 5]  # default: all
        [--split both]           # train | val | both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml


IMG_W, IMG_H = 1920, 1080
CLASS_ID = 0   # person only

# Difficulty filter defaults — same thresholds as src/eval/metrics.py.
# YOLOv11n input = 640px, source = 1920px → scale ≈ 1/3.
# 60px source height ≈ 20px at model input — minimum reliably detectable.
DEFAULT_MIN_HEIGHT     = 60    # px in 1920×1080 source
DEFAULT_MIN_WIDTH      = 20    # px in 1920×1080 source
DEFAULT_MIN_VISIBILITY = 0.30  # fraction of box area inside frame


def _convert_csv_to_yolo(
    csv_path: Path,
    min_height: float = DEFAULT_MIN_HEIGHT,
    min_width: float  = DEFAULT_MIN_WIDTH,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> dict[int, list[str]]:
    """Read one CSV and return {frame_no: [yolo_line, ...]}.

    Applies the same difficulty filter used during evaluation so that the
    training distribution matches the eval distribution:
      - min_width / min_height: skip boxes too small to detect at 640px input
      - min_visibility: skip boxes mostly outside the frame (simulation artifact)
    """
    df = pd.read_csv(csv_path)
    frame_labels: dict[int, list[str]] = {}
    skipped = 0
    for _, row in df.iterrows():
        x1 = float(row["x_top_left_BB"])
        y1 = float(row["y_top_left_BB"])
        x2 = float(row["x_bottom_right_BB"])
        y2 = float(row["y_bottom_right_BB"])

        raw_w = x2 - x1
        raw_h = y2 - y1

        # --- Difficulty filter 1: minimum size ---
        if raw_w < min_width or raw_h < min_height:
            skipped += 1
            continue

        # --- Difficulty filter 2: visibility (fraction inside frame) ---
        vis_x1 = max(0.0, x1); vis_y1 = max(0.0, y1)
        vis_x2 = min(IMG_W, x2); vis_y2 = min(IMG_H, y2)
        vis_area   = max(0.0, vis_x2 - vis_x1) * max(0.0, vis_y2 - vis_y1)
        total_area = max(1.0, raw_w * raw_h)
        if vis_area / total_area < min_visibility:
            skipped += 1
            continue

        # Clip coords to frame before converting to YOLO format
        x1 = max(0.0, min(IMG_W, x1))
        y1 = max(0.0, min(IMG_H, y1))
        x2 = max(0.0, min(IMG_W, x2))
        y2 = max(0.0, min(IMG_H, y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 1.0 or bh < 1.0:
            skipped += 1
            continue

        cx = (x1 + x2) / 2.0 / IMG_W
        cy = (y1 + y2) / 2.0 / IMG_H
        nw = bw / IMG_W
        nh = bh / IMG_H

        fn = int(row["frame_no_cam"])
        frame_labels.setdefault(fn, []).append(
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    return frame_labels


def _process_split(
    mta_root: Path,
    split: str,
    out_img: Path,
    out_lbl: Path,
    cam_ids: list[int],
    sample_rate: int,
    min_height: float,
    min_width: float,
    min_visibility: float,
) -> tuple[int, int]:
    """Extract frames and write labels for one split. Returns (n_frames, n_boxes)."""
    split_dir = mta_root / split
    n_frames = 0
    n_boxes  = 0

    for cam_id in cam_ids:
        cam_dir = split_dir / f"cam_{cam_id}"
        video   = cam_dir / f"cam_{cam_id}.mp4"
        csv     = cam_dir / f"coords_fib_cam_{cam_id}.csv"

        if not video.exists():
            print(f"  [skip] {video} not found")
            continue
        if not csv.exists():
            print(f"  [skip] {csv} not found")
            continue

        frame_labels = _convert_csv_to_yolo(
            csv, min_height=min_height,
            min_width=min_width, min_visibility=min_visibility)

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  cam_{cam_id}: {total} frames, {len(frame_labels)} annotated frames")

        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % sample_rate == 0 and frame_no in frame_labels:
                lines = frame_labels[frame_no]
                stem  = f"cam{cam_id}_frame{frame_no:07d}"
                img_path = out_img / f"{stem}.jpg"
                lbl_path = out_lbl / f"{stem}.txt"

                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                lbl_path.write_text("\n".join(lines) + "\n")

                n_frames += 1
                n_boxes  += len(lines)

            frame_no += 1

        cap.release()

    return n_frames, n_boxes


def _write_dataset_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["person"],
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[mta_to_yolo] dataset.yaml written → {yaml_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert MTA_ext_short to Ultralytics YOLO format")
    p.add_argument("--mta-root", required=True, metavar="PATH",
                   help="Path to MTA_ext_short (contains train/ and test/)")
    p.add_argument("--output-dir", default="dataset/mta_yolo", metavar="PATH")
    p.add_argument("--sample-rate", type=int, default=5,
                   help="Keep every N-th frame (default: 5 → ~8fps from 41fps)")
    p.add_argument("--cameras", nargs="+", type=int, default=list(range(6)),
                   help="Camera IDs to include (default: 0-5)")
    p.add_argument("--split", choices=["train", "val", "both"], default="both",
                   help="Which split(s) to process (train→images/train, test→images/val)")
    p.add_argument("--min-height", type=float, default=DEFAULT_MIN_HEIGHT,
                   help=f"Skip boxes shorter than this many pixels (default: {DEFAULT_MIN_HEIGHT}). "
                        f"Set 0 to disable.")
    p.add_argument("--min-width", type=float, default=DEFAULT_MIN_WIDTH,
                   help=f"Skip boxes narrower than this many pixels (default: {DEFAULT_MIN_WIDTH}). "
                        f"Set 0 to disable.")
    p.add_argument("--min-visibility", type=float, default=DEFAULT_MIN_VISIBILITY,
                   help=f"Skip boxes with less than this fraction inside the frame "
                        f"(default: {DEFAULT_MIN_VISIBILITY}). Set 0 to disable.")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable all difficulty filters (include all GT boxes).")
    args = p.parse_args()

    mta_root   = Path(args.mta_root)
    output_dir = Path(args.output_dir)

    if args.no_filter:
        min_h = min_w = min_vis = 0.0
    else:
        min_h   = args.min_height
        min_w   = args.min_width
        min_vis = args.min_visibility

    print(f"[mta_to_yolo] MTA root : {mta_root}")
    print(f"[mta_to_yolo] Output   : {output_dir}")
    if min_h > 0 or min_w > 0 or min_vis > 0:
        print(f"[mta_to_yolo] Filter   : min_height={min_h}px  "
              f"min_width={min_w}px  min_visibility={min_vis:.0%}")
    else:
        print("[mta_to_yolo] Filter   : disabled (all GT boxes included)")

    splits_to_process: list[tuple[str, str]] = []
    if args.split in ("train", "both"):
        splits_to_process.append(("train", "train"))
    if args.split in ("val", "both"):
        splits_to_process.append(("test", "val"))

    for mta_split, yolo_split in splits_to_process:
        out_img = output_dir / "images" / yolo_split
        out_lbl = output_dir / "labels" / yolo_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        print(f"\n[mta_to_yolo] Processing split '{mta_split}' → {yolo_split}/")
        n_frames, n_boxes = _process_split(
            mta_root, mta_split, out_img, out_lbl,
            args.cameras, args.sample_rate,
            min_height=min_h, min_width=min_w, min_visibility=min_vis)
        print(f"  → {n_frames} images, {n_boxes} boxes")

    _write_dataset_yaml(output_dir)
    print("\n[mta_to_yolo] Done.")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Next:   python scripts/train_yolo_mta.py --data {output_dir}/dataset.yaml")


if __name__ == "__main__":
    main()
