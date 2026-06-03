"""
Convert MMPTracking_short to Ultralytics YOLO format for detection fine-tuning.

Source: dataset/MMPTracking_short/<scene>/gt_cam<N>.csv + cam<N>.mp4
Output:
    dataset/mmp_yolo/
        images/train/   <scene>_cam<N>_f<FFFFF>.jpg
        images/val/     ...
        labels/train/   <scene>_cam<N>_f<FFFFF>.txt    (YOLO normalized xywh)
        labels/val/     ...
        dataset.yaml

Frame resolution: 640×360
Val split: last 2 scenes per environment (scene index >= n-2)

Run:
    python scripts/mmp_to_yolo.py [--short-root dataset/MMPTracking_short]
                                   [--output-dir dataset/mmp_yolo]
                                   [--sample-rate 5]
                                   [--min-height 20] [--min-width 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

IMG_W, IMG_H = 640, 360
CLASS_ID = 0

# MMPTracking is 640×360 real-world indoor — persons are larger relative to frame.
# Min 20px height ≈ 5.6% of frame height, keeps small but visible persons.
DEFAULT_MIN_H   = 20
DEFAULT_MIN_W   = 8
DEFAULT_MIN_VIS = 0.30

# Environments and their scenes (sorted = alphabetical within each env)
ENVS = {
    "cafe_shop":       ["cafe_shop_0","cafe_shop_1","cafe_shop_2","cafe_shop_3"],
    "industry_safety": ["industry_safety_0","industry_safety_1","industry_safety_2",
                        "industry_safety_3","industry_safety_4"],
    "lobby":           ["lobby_0","lobby_1","lobby_2","lobby_3"],
    "office":          ["office_0","office_1","office_2"],
    "retail":          ["retail_0","retail_1","retail_2","retail_3",
                        "retail_4","retail_5","retail_6","retail_7"],
}


def _split_scenes() -> tuple[list[str], list[str]]:
    """Last 1 scene per env → val; rest → train."""
    train, val = [], []
    for scenes in ENVS.values():
        val.append(scenes[-1])
        train.extend(scenes[:-1])
    return train, val


def _filter_box(row, min_h, min_w, min_vis) -> tuple[float,float,float,float] | None:
    x1, y1 = float(row["left"]), float(row["top"])
    w,  h  = float(row["width"]), float(row["height"])
    if w < min_w or h < min_h:
        return None
    # Clamp box to frame boundary first
    cx1 = max(0.0, x1); cy1 = max(0.0, y1)
    cx2 = min(float(IMG_W), x1 + w); cy2 = min(float(IMG_H), y1 + h)
    vis = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1) / max(1.0, w * h)
    if vis < min_vis:
        return None
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw < min_w or ch < min_h:
        return None
    # YOLO normalized cx, cy, w, h — all guaranteed [0,1]
    ncx = max(0.0, min(1.0, (cx1 + cw / 2) / IMG_W))
    ncy = max(0.0, min(1.0, (cy1 + ch / 2) / IMG_H))
    nw  = max(0.0, min(1.0, cw / IMG_W))
    nh  = max(0.0, min(1.0, ch / IMG_H))
    return (ncx, ncy, nw, nh)


def process_scene(scene: str, split: str, short_root: Path, out_root: Path,
                  sample_rate: int, min_h: int, min_w: int, min_vis: float) -> dict:
    scene_dir = short_root / scene
    imgs_out  = out_root / "images" / split
    lbls_out  = out_root / "labels" / split
    imgs_out.mkdir(parents=True, exist_ok=True)
    lbls_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "skipped_frames": 0}

    for csv_path in sorted(scene_dir.glob("gt_cam*.csv")):
        cam_id  = int(csv_path.stem.replace("gt_cam", ""))
        vid_path = scene_dir / f"cam{cam_id}.mp4"
        if not vid_path.exists():
            print(f"  [WARN] video not found: {vid_path}")
            continue

        df = pd.read_csv(csv_path)
        # Build {frame: [yolo_line, ...]}
        frame_labels: dict[int, list[str]] = {}
        for _, row in df.iterrows():
            frame = int(row["frame"])
            box = _filter_box(row, min_h, min_w, min_vis)
            if box is None:
                continue
            cx, cy, nw, nh = box
            frame_labels.setdefault(frame, []).append(
                f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )

        # Extract frames via OpenCV
        cap = cv2.VideoCapture(str(vid_path))
        frame_no = 0
        while True:
            ret, frame_img = cap.read()
            if not ret:
                break
            if frame_no % sample_rate == 0 and frame_no in frame_labels:
                stem = f"{scene}_cam{cam_id}_f{frame_no:05d}"
                img_path = imgs_out / f"{stem}.jpg"
                lbl_path = lbls_out / f"{stem}.txt"
                cv2.imwrite(str(img_path), frame_img)
                lbl_path.write_text("\n".join(frame_labels[frame_no]))
                stats["images"] += 1
                stats["labels"] += len(frame_labels[frame_no])
            elif frame_no % sample_rate == 0:
                stats["skipped_frames"] += 1
            frame_no += 1
        cap.release()

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--short-root", default="dataset/MMPTracking_short")
    p.add_argument("--output-dir", default="dataset/mmp_yolo")
    p.add_argument("--sample-rate", type=int, default=5,
                   help="Keep every Nth frame (default 5 → 5fps at 25fps source)")
    p.add_argument("--min-height", type=int, default=DEFAULT_MIN_H)
    p.add_argument("--min-width",  type=int, default=DEFAULT_MIN_W)
    p.add_argument("--min-vis",    type=float, default=DEFAULT_MIN_VIS)
    args = p.parse_args()

    short_root = Path(args.short_root)
    out_root   = Path(args.output_dir)

    if not short_root.exists():
        print(f"[ERROR] Short root not found: {short_root}")
        sys.exit(1)

    train_scenes, val_scenes = _split_scenes()
    known_scenes = train_scenes + val_scenes
    if not any((short_root / scene).exists() for scene in known_scenes):
        nested_root = Path("dataset/MMPTracking/MMPTracking_short")
        hint = ""
        if nested_root.exists():
            hint = f" Did you mean --short-root {nested_root}?"
        print(f"[ERROR] No known MMPTracking_short scenes found under: {short_root}.{hint}")
        sys.exit(1)

    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val   scenes ({len(val_scenes)}):   {val_scenes}")

    total = {"images": 0, "labels": 0}
    for split, scenes in [("train", train_scenes), ("val", val_scenes)]:
        split_imgs = split_lbls = 0
        for scene in scenes:
            if not (short_root / scene).exists():
                print(f"  [SKIP] {scene} not found")
                continue
            print(f"  [{split}] {scene} ...")
            st = process_scene(
                scene, split, short_root, out_root,
                args.sample_rate, args.min_height, args.min_width, args.min_vis,
            )
            split_imgs += st["images"]
            split_lbls += st["labels"]
            print(f"    → {st['images']} images, {st['labels']} boxes, "
                  f"{st['skipped_frames']} frames skipped (no GT)")
        print(f"  [{split}] total: {split_imgs} images, {split_lbls} boxes\n")
        total["images"] += split_imgs
        total["labels"] += split_lbls

    # dataset.yaml
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["person"],
    }))
    print(f"[done] {total['images']} images, {total['labels']} boxes total")
    print(f"       dataset.yaml → {yaml_path}")
    if total["images"] == 0:
        print("[ERROR] Conversion produced 0 images; check --short-root and GT/video files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
