"""Apply manual exact-source ReID labels to an MMPTracking ReID crop manifest.

Input labels are written by scripts/datasets/reid_label_app_exact.py:
  reid_labels_exact/labels_<env>.json

Each label maps `pid_key` (time/scene/raw_pid) to a manual group such as P12.
The output manifest reuses the original crop files through relative paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def env_of(scene: str) -> str:
    parts = scene.split("_")
    return "_".join(parts[:-1]) if parts and parts[-1].isdigit() else scene


def load_labels(labels_dir: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in sorted(labels_dir.glob("labels_*.json")):
        labels.update(json.load(path.open()))
    if not labels:
        raise SystemExit(f"no labels_*.json found in {labels_dir}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path("reid_labels_exact"))
    parser.add_argument("--crop-root", type=Path, default=Path("dataset/mmp_exact_reid_original"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/mmp_exact_reid_original_labeled"))
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    args = parser.parse_args()

    crop_root = args.crop_root.resolve()
    out_root = args.out_dir.resolve()
    labels = load_labels(args.labels_dir)

    identities = sorted(
        {
            f"{key.split('/')[0]}::{env_of(key.split('/')[1])}::{group}"
            for key, group in labels.items()
            if group != "JUNK"
        }
    )
    ident_to_pid = {ident: i for i, ident in enumerate(identities)}
    print(f"[apply-exact] {len(labels)} labeled tracks -> {len(identities)} identities")
    print(f"[apply-exact] per env: {dict(Counter(i.split('::')[1] for i in identities))}")

    for split in args.splits:
        src = crop_root / split / "manifest.csv"
        if not src.exists():
            print(f"[apply-exact] skip missing split: {src}")
            continue
        dst_dir = out_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "manifest.csv"
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
        kept = dropped = 0
        with src.open(encoding="utf-8", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                key = row.get("pid_key") or f"{row.get('time', split)}/{row['scene']}/{row.get('raw_pid', row['pid'])}"
                group = labels.get(key)
                if group is None or group == "JUNK":
                    dropped += 1
                    continue
                ident = f"{key.split('/')[0]}::{env_of(key.split('/')[1])}::{group}"
                row = {k: row.get(k, "") for k in fields}
                row["pid"] = ident_to_pid[ident]
                row["pid_key"] = key
                row["rel_path"] = os.path.relpath(crop_root / row["rel_path"], out_root)
                writer.writerow(row)
                kept += 1
        print(f"[apply-exact] {split}: kept={kept} dropped={dropped} -> {dst}")


if __name__ == "__main__":
    main()
