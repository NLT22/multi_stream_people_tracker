"""Build a JPEG crop cache for MMPTracking ReID training.

The ReID trainer can train directly from videos, but that forces random MP4
seeks during every batch. This script pays that cost once and writes:

    dataset/MMPTracking_10minute_reid_cache/
        train/manifest.csv
        train/<pid>/<scene>_cam<id>_f<frame>_<n>.jpg
        val/manifest.csv
        val/<pid>/<scene>_cam<id>_f<frame>_<n>.jpg

Run:
    python -m scripts.datasets.build_reid_crop_cache \
        --src-root dataset/MMPTracking_10minute \
        --output-dir dataset/MMPTracking_10minute_reid_cache \
        --exclude-retail
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.train.finetune_reid_mmp import MMPReidDataset


def _scan_scenes(src_root: Path, split: str, exclude_retail: bool) -> list[str]:
    split_dir = src_root / split
    if not split_dir.exists():
        return []
    names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    if exclude_retail:
        names = [name for name in names if "retail" not in name]
    return [f"{split}/{name}" for name in names]


def _safe_scene_name(video_path: str) -> str:
    return Path(video_path).parent.name


def build_split(args, split: str) -> dict[str, int]:
    scenes = _scan_scenes(args.src_root, split, args.exclude_retail)
    if args.max_scenes_per_split > 0:
        scenes = scenes[:args.max_scenes_per_split]
    if not scenes:
        print(f"[SKIP] no scenes for split={split}")
        return {"crops": 0, "persons": 0}

    split_out = args.output_dir / split
    manifest_path = split_out / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{manifest_path} already exists. Pass --overwrite to rebuild."
        )
    if split_out.exists() and args.overwrite:
        shutil.rmtree(split_out)
    split_out.mkdir(parents=True, exist_ok=True)

    ds = MMPReidDataset(
        args.src_root,
        scenes,
        transform=None,
        sample_rate=args.sample_rate,
        min_w=args.min_w,
        min_h=args.min_h,
        min_imgs_per_pid=args.min_imgs_pid,
        split_name=f"{split}_cache",
        prefer_clean_gt=args.prefer_clean_gt,
    )

    rows: list[dict[str, str | int]] = []
    order = sorted(
        range(len(ds.samples)),
        key=lambda i: (
            ds.samples[i].video_path,
            ds.samples[i].frame_no,
            ds.samples[i].gid,
            ds.samples[i].cam_id,
            i,
        ),
    )

    for ordinal, idx in enumerate(tqdm(order, desc=f"cache {split}", unit="crop")):
        sample = ds.samples[idx]
        pid_cls = ds.pid_to_cls[sample.gid]
        scene = _safe_scene_name(sample.video_path)
        pid_dir = split_out / f"{pid_cls:06d}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{scene}_cam{sample.cam_id}_f{sample.frame_no:06d}_"
            f"{ordinal:08d}.jpg"
        )
        image_path = pid_dir / filename

        crop_rgb = ds.load_crop_rgb(idx)
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(
            str(image_path),
            crop_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"Failed to write crop: {image_path}")

        rows.append({
            "rel_path": str(image_path.relative_to(args.output_dir)),
            "pid": pid_cls,
            "cam_id": sample.cam_id,
            "scene": scene,
            "frame": sample.frame_no,
        })

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rel_path", "pid", "cam_id", "scene", "frame"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {split}: {len(rows)} crops, {ds.num_classes} persons")
    print(f"       manifest: {manifest_path}")
    return {"crops": len(rows), "persons": ds.num_classes}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path, default=Path("dataset/MMPTracking_10minute"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("dataset/MMPTracking_10minute_reid_cache"))
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=["train", "val"])
    p.add_argument("--sample-rate", type=int, default=5)
    p.add_argument("--min-w", type=int, default=20)
    p.add_argument("--min-h", type=int, default=40)
    p.add_argument("--min-imgs-pid", type=int, default=4)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--exclude-retail", action="store_true")
    p.add_argument("--prefer-clean-gt", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-scenes-per-split", type=int, default=0,
                   help="Debug only: build at most N scenes per split.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_crops = 0
    for split in args.splits:
        stats = build_split(args, split)
        total_crops += stats["crops"]
    print(f"\nTotal crops: {total_crops}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
