"""Build a ReID crop dataset from the official MMPTracking zip files.

This reads the exact MMPTracking image/label zips, not the older extracted
MP4/CSV cache. Output:

  dataset/mmp_exact_reid/
    train/manifest.csv
    train/<pid_index>/*.jpg
    val/manifest.csv
    val/<pid_index>/*.jpg
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

from PIL import Image


RGB_RE = re.compile(r"rgb_(\d+)_(\d+)\.(json|jpg)$")
DEFAULT_MIN_H = 32
DEFAULT_MIN_W = 12
DEFAULT_MIN_VIS = 0.30


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
    elif split == "val":
        base = mmp_root / "MMPTracking_validation" / "validation"
    else:
        raise ValueError(f"unsupported split: {split}")

    items: list[SceneZip] = []
    image_root = base / "images"
    label_root = base / "labels"
    for label_zip in sorted(label_root.glob("*/*.zip")):
        time_name = label_zip.parent.name
        scene = label_zip.stem
        images_zip = image_root / time_name / label_zip.name
        if images_zip.exists():
            items.append(SceneZip(split, time_name, scene, images_zip, label_zip))
        else:
            print(f"[WARN] missing matching image zip: {images_zip}")
    return items


def _parse_rgb_name(path_in_zip: str) -> tuple[int, int]:
    match = RGB_RE.search(Path(path_in_zip).name)
    if not match:
        raise ValueError(f"unexpected rgb filename: {path_in_zip}")
    return int(match.group(1)), int(match.group(2))


def _keep_frame(label_name: str, sample_rate: int) -> bool:
    frame, _cam = _parse_rgb_name(label_name)
    return frame % sample_rate == 0


def _clamp_box(
    bbox: list[float],
    image_w: int,
    image_h: int,
    min_w: int,
    min_h: int,
    min_vis: float,
) -> tuple[int, int, int, int] | None:
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

    return round(cx1), round(cy1), round(cx2), round(cy2)


def _pid_key(item: SceneZip, raw_pid: str) -> str:
    # MMP labels are scene-local for MTMC. Keep scenes separate so unrelated
    # people with the same JSON id in different scenes do not collide.
    return f"{item.time_name}/{item.scene}/{raw_pid}"


def _process_scene(
    item: SceneZip,
    out_root: Path,
    sample_rate: int,
    min_w: int,
    min_h: int,
    min_vis: float,
    max_labels_per_scene: int | None,
    max_crops_per_scene: int | None,
    pid_to_index: dict[str, int],
    manifest: csv.DictWriter,
) -> dict[str, int]:
    split_out = out_root / item.split
    split_out.mkdir(parents=True, exist_ok=True)
    stats = {
        "labels_seen": 0,
        "labels_sampled": 0,
        "crops": 0,
        "missing_images": 0,
        "empty_after_filter": 0,
    }

    with zipfile.ZipFile(item.images_zip) as image_zip, zipfile.ZipFile(item.labels_zip) as label_zip:
        image_names = set(image_zip.namelist())
        label_names = sorted(n for n in label_zip.namelist() if n.endswith(".json"))

        for label_name in label_names:
            if max_labels_per_scene is not None and stats["labels_sampled"] >= max_labels_per_scene:
                break
            if max_crops_per_scene is not None and stats["crops"] >= max_crops_per_scene:
                break

            stats["labels_seen"] += 1
            if not _keep_frame(label_name, sample_rate):
                continue

            image_name = str(Path(label_name).with_suffix(".jpg"))
            if image_name not in image_names:
                stats["missing_images"] += 1
                continue

            frame, cam = _parse_rgb_name(label_name)
            labels = json.loads(label_zip.read(label_name))
            image = Image.open(BytesIO(image_zip.read(image_name))).convert("RGB")
            image_w, image_h = image.size
            crops_this_frame = 0

            for raw_pid, bbox in sorted(labels.items(), key=lambda kv: str(kv[0])):
                if max_crops_per_scene is not None and stats["crops"] >= max_crops_per_scene:
                    break

                box = _clamp_box(bbox, image_w, image_h, min_w, min_h, min_vis)
                if box is None:
                    continue

                key = _pid_key(item, str(raw_pid))
                if key not in pid_to_index:
                    pid_to_index[key] = len(pid_to_index)
                pid_index = pid_to_index[key]

                pid_dir = split_out / f"{pid_index:06d}"
                pid_dir.mkdir(parents=True, exist_ok=True)
                x1, y1, x2, y2 = box
                crop = image.crop((x1, y1, x2, y2))
                crop_name = (
                    f"{item.time_name}_{item.scene}_cam{cam}_"
                    f"f{frame:05d}_pid{raw_pid}_{stats['crops']:08d}.jpg"
                )
                crop_path = pid_dir / crop_name
                crop.save(crop_path, quality=92)

                manifest.writerow(
                    {
                        "rel_path": str(crop_path.relative_to(out_root)),
                        "pid": pid_index,
                        "pid_key": key,
                        "raw_pid": raw_pid,
                        "split": item.split,
                        "time": item.time_name,
                        "scene": item.scene,
                        "cam": cam,
                        "frame": frame,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
                crops_this_frame += 1
                stats["crops"] += 1

            stats["labels_sampled"] += 1
            if crops_this_frame == 0:
                stats["empty_after_filter"] += 1

    return stats


def _write_split(
    scenes: list[SceneZip],
    out_root: Path,
    split: str,
    sample_rate: int,
    min_w: int,
    min_h: int,
    min_vis: float,
    max_labels_per_scene: int | None,
    max_crops_per_scene: int | None,
) -> dict[str, int]:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.csv"
    pid_to_index: dict[str, int] = {}
    totals = {"crops": 0, "missing_images": 0, "empty_after_filter": 0}

    fields = [
        "rel_path",
        "pid",
        "pid_key",
        "raw_pid",
        "split",
        "time",
        "scene",
        "cam",
        "frame",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, item in enumerate(scenes, start=1):
            print(f"[{idx:03d}/{len(scenes):03d}] {split} {item.time_name}/{item.scene}")
            stats = _process_scene(
                item,
                out_root,
                sample_rate,
                min_w,
                min_h,
                min_vis,
                max_labels_per_scene,
                max_crops_per_scene,
                pid_to_index,
                writer,
            )
            for key in totals:
                totals[key] += stats[key]
            print(
                "  -> "
                f"{stats['crops']} crops, "
                f"{stats['empty_after_filter']} empty, {stats['missing_images']} missing"
            )

    totals["pids"] = len(pid_to_index)
    print(f"[done] {split}: {totals['crops']} crops, {totals['pids']} pids")
    print(f"       manifest: {manifest_path}")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmp-root", default="dataset/MMPTracking")
    parser.add_argument("--output-dir", default="dataset/mmp_exact_reid")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_H)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_W)
    parser.add_argument("--min-vis", type=float, default=DEFAULT_MIN_VIS)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-labels-per-scene", type=int, default=None)
    parser.add_argument("--max-crops-per-scene", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
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

    print(f"[reid-convert] source={mmp_root}")
    print(f"[reid-convert] output={out_root}")
    print(f"[reid-convert] sample_rate={args.sample_rate}")

    grand = {"crops": 0, "pids": 0}
    for split in args.splits:
        scenes = sorted(_scene_zips(mmp_root, split), key=lambda s: (s.time_name, s.scene))
        if args.max_scenes is not None:
            scenes = scenes[: args.max_scenes]
        if not scenes:
            print(f"[SKIP] no scenes for split={split}")
            continue
        totals = _write_split(
            scenes,
            out_root,
            split,
            args.sample_rate,
            args.min_width,
            args.min_height,
            args.min_vis,
            args.max_labels_per_scene,
            args.max_crops_per_scene,
        )
        grand["crops"] += totals["crops"]
        grand["pids"] += totals["pids"]

    print("[done]")
    print(f"  crops: {grand['crops']}")
    print(f"  pids:  {grand['pids']}")


if __name__ == "__main__":
    main()
