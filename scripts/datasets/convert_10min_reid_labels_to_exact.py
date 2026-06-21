"""Convert old 10-minute ReID manual labels to exact MMPTracking pid_key labels.

Old labels are keyed like:
  63am_cafe_shop_0|0

Exact-source labels are keyed like:
  63am/cafe_shop_0/1

The safest bridge is scene-local rank:
  sorted old pids in a scene -> sorted exact manifest pids in the same scene

This handles cases where the old 10-minute cache and exact-source converter used
different global pid offsets, while preserving the per-scene raw-id order.

Run after building exact-source crops with scripts/datasets/mmp_exact_to_reid.py:

    ./venv/bin/python scripts/datasets/convert_10min_reid_labels_to_exact.py \
      --labels-dir reid_labels \
      --exact-crop-root dataset/mmp_exact_reid_original \
      --out-dir reid_labels_exact
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def env_of_scene(scene_with_time: str) -> str:
    parts = scene_with_time.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[1:-1])
    return scene_with_time


def load_exact_scene_tracks(exact_crop_root: Path, split: str) -> dict[str, list[tuple[int, str]]]:
    manifest = exact_crop_root / split / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"exact manifest not found: {manifest}")

    by_scene: dict[str, dict[int, str]] = defaultdict(dict)
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            old_scene = f"{row['time']}_{row['scene']}"
            exact_pid = int(row["pid"])
            exact_key = row["pid_key"]
            existing = by_scene[old_scene].get(exact_pid)
            if existing is not None and existing != exact_key:
                raise SystemExit(f"ambiguous mapping for {old_scene}|{exact_pid}: {existing} vs {exact_key}")
            by_scene[old_scene][exact_pid] = exact_key
    return {scene: sorted(items.items()) for scene, items in by_scene.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path("reid_labels"))
    parser.add_argument("--exact-crop-root", type=Path, default=Path("dataset/mmp_exact_reid_original"))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--out-dir", type=Path, default=Path("reid_labels_exact"))
    parser.add_argument(
        "--mapping",
        choices=["scene-rank", "global-pid"],
        default="scene-rank",
        help="scene-rank is recommended; global-pid is only for debugging old cache offsets.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any old label has no exact manifest match.")
    args = parser.parse_args()

    exact_by_scene = load_exact_scene_tracks(args.exact_crop_root, args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_by_scene: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for label_path in sorted(args.labels_dir.glob("labels_*.json")):
        labels = json.load(label_path.open())
        for old_key, group in labels.items():
            old_scene, old_pid = old_key.split("|", 1)
            old_by_scene[old_scene].append((int(old_pid), old_key, group))

    old_to_exact: dict[str, str] = {}
    if args.mapping == "global-pid":
        for scene, exact_tracks in exact_by_scene.items():
            for exact_pid, exact_key in exact_tracks:
                old_to_exact[f"{scene}|{exact_pid}"] = exact_key
    else:
        for scene, old_tracks in old_by_scene.items():
            exact_tracks = exact_by_scene.get(scene, [])
            old_tracks = sorted(old_tracks)
            exact_tracks = sorted(exact_tracks)
            if len(old_tracks) != len(exact_tracks):
                print(
                    f"[convert] warning: {scene} old_tracks={len(old_tracks)} "
                    f"exact_tracks={len(exact_tracks)}; mapping common prefix only"
                )
            for (_old_pid, old_key, _group), (_exact_pid, exact_key) in zip(old_tracks, exact_tracks):
                old_to_exact[old_key] = exact_key

    total = matched = missing = 0
    per_env_counts: Counter[str] = Counter()
    per_env_missing: Counter[str] = Counter()
    grouped: dict[str, dict[str, str]] = {}

    for old_tracks in old_by_scene.values():
        for _old_pid, old_key, group in old_tracks:
            total += 1
            env = env_of_scene(old_key.split("|", 1)[0])
            exact_key = old_to_exact.get(old_key)
            if exact_key is None:
                missing += 1
                per_env_missing[env] += 1
                if args.strict:
                    raise SystemExit(f"missing exact mapping for {old_key}")
                continue
            grouped.setdefault(env, {})[exact_key] = group
            per_env_counts[env] += 1
            matched += 1

    for env, labels in sorted(grouped.items()):
        out = args.out_dir / f"labels_{env}.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(labels, fh, indent=2, sort_keys=True)
        print(f"[convert] {env}: {len(labels)} labels -> {out}")

    print(
        f"[convert] total={total} matched={matched} missing={missing} "
        f"exact_manifest_keys={sum(len(v) for v in exact_by_scene.values())} "
        f"mapping={args.mapping}"
    )
    if per_env_missing:
        print(f"[convert] missing per env: {dict(per_env_missing)}")
    print(f"[convert] matched per env: {dict(per_env_counts)}")


if __name__ == "__main__":
    main()
