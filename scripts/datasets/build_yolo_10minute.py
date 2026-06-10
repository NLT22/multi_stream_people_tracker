"""Build a YOLO detection dataset from MMPTracking_10minute.

Reuses mmp_to_yolo.process_scene but honors the session-based split that already
exists on disk (train/ vs val/ subdirs) instead of the per-environment split.

    dataset/MMPTracking_10minute/train/<scene>/  -> images/train, labels/train
    dataset/MMPTracking_10minute/val/<scene>/    -> images/val,   labels/val

Run:
    python scripts/datasets/build_yolo_10minute.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.datasets.mmp_to_yolo import (
    DEFAULT_MIN_H, DEFAULT_MIN_W, DEFAULT_MIN_VIS, process_scene,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="dataset/MMPTracking_10minute", type=Path)
    ap.add_argument("--output-dir", default="dataset/mmp_yolo_10minute", type=Path)
    ap.add_argument("--sample-rate", type=int, default=5,
                    help="Keep every Nth frame (15 fps source -> 3 fps at 5).")
    ap.add_argument("--min-height", type=int, default=DEFAULT_MIN_H)
    ap.add_argument("--min-width", type=int, default=DEFAULT_MIN_W)
    ap.add_argument("--min-vis", type=float, default=DEFAULT_MIN_VIS)
    args = ap.parse_args()

    out_root = args.output_dir
    grand = {"images": 0, "labels": 0}
    for split in ("train", "val"):
        split_dir = args.src_root / split
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} missing")
            continue
        scenes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        print(f"\n=== {split}: {len(scenes)} scenes ===", flush=True)
        s_img = s_lbl = 0
        for scene in scenes:
            st = process_scene(scene, split, split_dir, out_root,
                               args.sample_rate, args.min_height,
                               args.min_width, args.min_vis)
            s_img += st["images"]
            s_lbl += st["labels"]
            print(f"  {scene:>26}  imgs={st['images']} boxes={st['labels']}",
                  flush=True)
        print(f"  -> {split}: {s_img} images, {s_lbl} boxes", flush=True)
        grand["images"] += s_img
        grand["labels"] += s_lbl

    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }, sort_keys=False))
    print(f"\nTotal: {grand['images']} images, {grand['labels']} boxes")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
