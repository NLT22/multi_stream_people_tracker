"""Filter ReID crop manifests with a stronger YOLO person detector.

This is a crop-quality gate for training data, not the production detector.
It reads an existing ReID crop root with manifest.csv files, runs a YOLO model
on the crop images, and writes a new manifest that keeps only crops where YOLO
finds a person.

Example:

  ./venv/bin/python scripts/datasets/filter_reid_crops_yolo.py \
    --input-root dataset/mmp_exact_reid_labeled_clean_full_envmerge \
    --output-root dataset/mmp_exact_reid_labeled_clean_full_envmerge_yolo11x \
    --weights yolo11x.pt \
    --splits train \
    --batch 64 --imgsz 320 --conf 0.15
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


def env_of(scene: str) -> str:
    parts = scene.split("_")
    return "_".join(parts[:-1]) if parts and parts[-1].isdigit() else scene


def chunks(items: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def best_person(result) -> tuple[float, float]:
    """Return best person confidence and crop area ratio from one YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, 0.0
    best_conf = 0.0
    best_ratio = 0.0
    image_h, image_w = result.orig_shape[:2]
    image_area = max(1.0, float(image_w * image_h))
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # COCO person
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
        if conf > best_conf:
            best_conf = conf
            best_ratio = ratio
    return best_conf, best_ratio


def read_rows(manifest: Path, limit: int | None) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not fieldnames:
        raise SystemExit(f"empty or invalid manifest: {manifest}")
    return fieldnames, rows


def filter_split(
    model: YOLO,
    input_root: Path,
    output_root: Path,
    split: str,
    batch: int,
    imgsz: int,
    conf: float,
    min_person_area_ratio: float,
    device: str | int | None,
    limit: int | None,
) -> None:
    src_manifest = input_root / split / "manifest.csv"
    if not src_manifest.exists():
        print(f"[yolo-filter] skip missing split: {src_manifest}")
        return

    fieldnames, rows = read_rows(src_manifest, limit=limit)
    out_split = output_root / split
    out_split.mkdir(parents=True, exist_ok=True)
    kept_manifest = out_split / "manifest.csv"
    rejected_manifest = out_split / "rejected_yolo.csv"

    kept = rejected = missing = 0
    kept_env: Counter[str] = Counter()
    rejected_env: Counter[str] = Counter()

    with kept_manifest.open("w", encoding="utf-8", newline="") as kept_fh, rejected_manifest.open(
        "w", encoding="utf-8", newline=""
    ) as reject_fh:
        kept_writer = csv.DictWriter(kept_fh, fieldnames=fieldnames)
        kept_writer.writeheader()
        reject_writer = csv.DictWriter(
            reject_fh,
            fieldnames=fieldnames + ["yolo_person_conf", "yolo_person_area_ratio", "reject_reason"],
        )
        reject_writer.writeheader()

        for group in chunks(rows, batch):
            existing: list[dict[str, str]] = []
            image_paths: list[str] = []
            for row in group:
                image_path = (input_root / row["rel_path"]).resolve()
                if not image_path.exists():
                    miss = dict(row)
                    miss["yolo_person_conf"] = "0.0000"
                    miss["yolo_person_area_ratio"] = "0.0000"
                    miss["reject_reason"] = "missing_file"
                    reject_writer.writerow(miss)
                    missing += 1
                    rejected += 1
                    rejected_env[env_of(row.get("scene", ""))] += 1
                    continue
                existing.append(row)
                image_paths.append(str(image_path))
            if not existing:
                continue

            results = model.predict(
                source=image_paths,
                imgsz=imgsz,
                conf=conf,
                classes=[0],
                device=device,
                verbose=False,
            )

            for row, result in zip(existing, results):
                best_conf, best_ratio = best_person(result)
                env = env_of(row.get("scene", ""))
                if best_conf >= conf and best_ratio >= min_person_area_ratio:
                    out_row = dict(row)
                    out_row["rel_path"] = os.path.relpath((input_root / row["rel_path"]).resolve(), output_root)
                    kept_writer.writerow(out_row)
                    kept += 1
                    kept_env[env] += 1
                else:
                    reject_row = dict(row)
                    reject_row["yolo_person_conf"] = f"{best_conf:.4f}"
                    reject_row["yolo_person_area_ratio"] = f"{best_ratio:.4f}"
                    reject_row["reject_reason"] = "no_person"
                    reject_writer.writerow(reject_row)
                    rejected += 1
                    rejected_env[env] += 1

            done = kept + rejected
            if done % max(batch * 20, 1) == 0:
                print(f"[yolo-filter] {split}: processed={done} kept={kept} rejected={rejected}")

    print(
        f"[yolo-filter] {split}: kept={kept} rejected={rejected} missing={missing} "
        f"-> {kept_manifest}"
    )
    print(f"[yolo-filter] {split}: kept per env={dict(kept_env)}")
    print(f"[yolo-filter] {split}: rejected per env={dict(rejected_env)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11x.pt")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--min-person-area-ratio", type=float, default=0.0)
    parser.add_argument("--device", default=0)
    parser.add_argument("--limit", type=int, default=None, help="debug/audit only; process first N rows")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    for split in args.splits:
        filter_split(
            model=model,
            input_root=input_root,
            output_root=output_root,
            split=split,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=args.conf,
            min_person_area_ratio=args.min_person_area_ratio,
            device=args.device,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
