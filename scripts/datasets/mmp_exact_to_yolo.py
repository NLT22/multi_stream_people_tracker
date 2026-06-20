"""Build a YOLO detector dataset from the official MMPTracking zip files.

This script intentionally reads from:

  dataset/MMPTracking/MMPTracking_training/train/{images,labels}/...
  dataset/MMPTracking/MMPTracking_validation/validation/{images,labels}/...

It does not use the older extracted MP4/CSV caches. The output is a standard
Ultralytics dataset under dataset/mmp_exact_yolo by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image


CLASS_ID = 0
DEFAULT_MIN_H = 20
DEFAULT_MIN_W = 8
DEFAULT_MIN_VIS = 0.30
RGB_RE = re.compile(r"rgb_(\d+)_(\d+)\.(json|jpg)$")


@dataclass(frozen=True)
class SceneZip:
    split: str
    time_name: str
    scene: str
    images_zip: Path
    labels_zip: Path


def _scene_zips(mmp_root: Path, split: str) -> list[SceneZip]:
    if split == "train":
        base = mmp_root / "MMPTracking_training" / "train"
        out_split = "train"
    elif split == "val":
        base = mmp_root / "MMPTracking_validation" / "validation"
        out_split = "val"
    else:
        raise ValueError(f"Unsupported split: {split}")

    image_root = base / "images"
    label_root = base / "labels"
    items: list[SceneZip] = []
    for label_zip in sorted(label_root.glob("*/*.zip")):
        time_name = label_zip.parent.name
        scene = label_zip.stem
        images_zip = image_root / time_name / label_zip.name
        if images_zip.exists():
            items.append(SceneZip(out_split, time_name, scene, images_zip, label_zip))
        else:
            print(f"[WARN] Missing matching image zip for {label_zip}")
    return items


def _parse_rgb_name(path_in_zip: str) -> tuple[int, int]:
    match = RGB_RE.search(Path(path_in_zip).name)
    if not match:
        raise ValueError(f"Unexpected rgb filename: {path_in_zip}")
    return int(match.group(1)), int(match.group(2))


def _read_scene_size(images_zip: zipfile.ZipFile) -> tuple[int, int]:
    first = next((n for n in sorted(images_zip.namelist()) if n.endswith(".jpg")), None)
    if first is None:
        raise ValueError("image zip contains no jpg files")
    with Image.open(BytesIO(images_zip.read(first))) as image:
        return image.size


def _box_to_yolo(
    bbox: list[float],
    image_w: int,
    image_h: int,
    min_w: int,
    min_h: int,
    min_vis: float,
) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    raw_w = x2 - x1
    raw_h = y2 - y1
    if raw_w < min_w or raw_h < min_h:
        return None

    cx1 = max(0.0, min(float(image_w), x1))
    cy1 = max(0.0, min(float(image_h), y1))
    cx2 = max(0.0, min(float(image_w), x2))
    cy2 = max(0.0, min(float(image_h), y2))
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w < min_w or crop_h < min_h:
        return None

    visible_ratio = (crop_w * crop_h) / max(1.0, raw_w * raw_h)
    if visible_ratio < min_vis:
        return None

    return (
        (cx1 + crop_w / 2.0) / image_w,
        (cy1 + crop_h / 2.0) / image_h,
        crop_w / image_w,
        crop_h / image_h,
    )


def _keep_frame(label_name: str, sample_rate: int) -> bool:
    frame, _cam = _parse_rgb_name(label_name)
    return frame % sample_rate == 0


def _process_scene(
    item: SceneZip,
    out_root: Path,
    sample_rate: int,
    min_w: int,
    min_h: int,
    min_vis: float,
    max_labels_per_scene: int | None,
    manifest: csv.writer,
) -> dict[str, int]:
    image_out = out_root / "images" / item.split
    label_out = out_root / "labels" / item.split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "labels_seen": 0,
        "labels_sampled": 0,
        "images": 0,
        "boxes": 0,
        "missing_images": 0,
        "empty_after_filter": 0,
    }

    with zipfile.ZipFile(item.images_zip) as image_zip, zipfile.ZipFile(item.labels_zip) as label_zip:
        image_names = set(image_zip.namelist())
        image_w, image_h = _read_scene_size(image_zip)
        label_names = sorted(n for n in label_zip.namelist() if n.endswith(".json"))

        for label_name in label_names:
            if max_labels_per_scene is not None and stats["labels_sampled"] >= max_labels_per_scene:
                break
            stats["labels_seen"] += 1
            if not _keep_frame(label_name, sample_rate):
                continue

            image_name = str(Path(label_name).with_suffix(".jpg"))
            if image_name not in image_names:
                stats["missing_images"] += 1
                continue

            labels = json.loads(label_zip.read(label_name))
            yolo_lines: list[str] = []
            seen_lines: set[str] = set()
            for bbox in labels.values():
                box = _box_to_yolo(bbox, image_w, image_h, min_w, min_h, min_vis)
                if box is None:
                    continue
                cx, cy, bw, bh = box
                line = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                yolo_lines.append(line)

            stats["labels_sampled"] += 1
            if not yolo_lines:
                stats["empty_after_filter"] += 1
                continue

            frame, cam = _parse_rgb_name(label_name)
            stem = f"{item.time_name}_{item.scene}_cam{cam}_f{frame:05d}"
            image_path = image_out / f"{stem}.jpg"
            label_path = label_out / f"{stem}.txt"
            image_path.write_bytes(image_zip.read(image_name))
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            stats["images"] += 1
            stats["boxes"] += len(yolo_lines)
            manifest.writerow(
                [
                    item.split,
                    item.time_name,
                    item.scene,
                    cam,
                    frame,
                    len(yolo_lines),
                    image_name,
                    str(image_path),
                    str(label_path),
                ]
            )

    return stats


def _write_dataset_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }
    (out_root / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmp-root", default="dataset/MMPTracking")
    parser.add_argument("--output-dir", default="dataset/mmp_exact_yolo")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_H)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_W)
    parser.add_argument("--min-vis", type=float, default=DEFAULT_MIN_VIS)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-labels-per-scene", type=int, default=None)
    parser.add_argument("--clean", action="store_true", help="Delete output dir before conversion.")
    args = parser.parse_args()

    if args.sample_rate < 1:
        raise SystemExit("--sample-rate must be >= 1")

    mmp_root = Path(args.mmp_root)
    out_root = Path(args.output_dir)
    if not mmp_root.exists():
        raise SystemExit(f"MMPTracking root not found: {mmp_root}")
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes: list[SceneZip] = []
    for split in args.splits:
        scenes.extend(_scene_zips(mmp_root, split))
    scenes = sorted(scenes, key=lambda s: (s.split, s.time_name, s.scene))
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise SystemExit("No MMPTracking scene zips found.")

    print(f"[convert] source={mmp_root}")
    print(f"[convert] output={out_root}")
    print(f"[convert] scenes={len(scenes)} sample_rate={args.sample_rate}")

    totals = {
        "images": 0,
        "boxes": 0,
        "missing_images": 0,
        "empty_after_filter": 0,
    }
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "time", "scene", "cam", "frame", "boxes", "zip_image", "image", "label"])
        for idx, item in enumerate(scenes, start=1):
            print(f"[{idx:03d}/{len(scenes):03d}] {item.split} {item.time_name}/{item.scene}")
            stats = _process_scene(
                item,
                out_root,
                args.sample_rate,
                args.min_width,
                args.min_height,
                args.min_vis,
                args.max_labels_per_scene,
                writer,
            )
            for key in totals:
                totals[key] += stats[key]
            print(
                "  -> "
                f"{stats['images']} images, {stats['boxes']} boxes, "
                f"{stats['empty_after_filter']} empty, {stats['missing_images']} missing"
            )

    _write_dataset_yaml(out_root)
    print("[done]")
    print(f"  images: {totals['images']}")
    print(f"  boxes:  {totals['boxes']}")
    print(f"  yaml:   {out_root / 'dataset.yaml'}")
    print(f"  manifest: {manifest_path}")
    if totals["images"] == 0:
        raise SystemExit("Conversion produced 0 images.")


if __name__ == "__main__":
    main()
