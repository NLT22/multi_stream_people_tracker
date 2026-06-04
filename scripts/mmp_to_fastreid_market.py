"""Convert MMPTracking_short crops to FastReID/Market1501-style folders.

FastReID already ships a Market1501 dataset loader.  This script writes MMP
person crops with Market-like filenames so we can use official FastReID
training without patching the third-party repo.

Output layout:

    dataset/fastreid_mmp/
      Market-1501-v15.09.15/
        bounding_box_train/
        query/
        bounding_box_test/

Filenames follow Market1501's regex: ``0001_c1s1_000001_00.jpg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


ENVS = {
    "cafe_shop":       ["cafe_shop_0", "cafe_shop_1", "cafe_shop_2", "cafe_shop_3"],
    "industry_safety": ["industry_safety_0", "industry_safety_1", "industry_safety_2",
                        "industry_safety_3", "industry_safety_4"],
    "lobby":           ["lobby_0", "lobby_1", "lobby_2", "lobby_3"],
    "office":          ["office_0", "office_1", "office_2"],
    "retail":          ["retail_0", "retail_1", "retail_2", "retail_3",
                        "retail_4", "retail_5", "retail_6", "retail_7"],
}


def _train_val_scenes() -> tuple[list[str], list[str]]:
    train, val = [], []
    for scenes in ENVS.values():
        val.append(scenes[-1])
        train.extend(scenes[:-1])
    return train, val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create FastReID Market1501-style crops from MMPTracking_short")
    p.add_argument("--short-root", default="dataset/MMPTracking_short")
    p.add_argument("--output", default="dataset/fastreid_mmp")
    p.add_argument("--sample-rate", type=int, default=5)
    p.add_argument("--min-w", type=int, default=20)
    p.add_argument("--min-h", type=int, default=40)
    p.add_argument("--min-visible-ratio", type=float, default=0.30,
                   help="Visible fraction after clipping to frame. "
                        "Does not catch shelf occlusion; only frame-boundary visibility.")
    p.add_argument("--query-per-pid-cam", type=int, default=1,
                   help="For validation scenes, put first N crops per pid/cam into query.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _clip_box(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int, float]:
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    total = max(1.0, w * h)
    visible = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return int(x1), int(y1), int(x2), int(y2), visible / total


def _scene_gid(scene_index: int, local_pid: int) -> tuple[int, int]:
    """Return stable global pid key for compact remapping."""
    return scene_index, local_pid


def _build_pid_map(short_root: Path, scenes: list[str]) -> dict[tuple[int, int], int]:
    keys = []
    for scene_index, scene in enumerate(scenes):
        scene_dir = short_root / scene
        for csv_path in sorted(scene_dir.glob("gt_cam*.csv")):
            df = pd.read_csv(csv_path, usecols=["person_id"])
            keys.extend(_scene_gid(scene_index, int(pid)) for pid in df["person_id"].unique())
    unique = sorted(set(keys))
    return {key: idx + 1 for idx, key in enumerate(unique)}


def _write_crop(path: Path, crop, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def _convert_split(
    short_root: Path,
    scenes: list[str],
    out_train: Path,
    out_query: Path,
    out_gallery: Path,
    pid_map: dict[tuple[int, int], int],
    sample_rate: int,
    min_w: int,
    min_h: int,
    min_visible_ratio: float,
    query_per_pid_cam: int,
    jpeg_quality: int,
    split: str,
) -> dict[str, int]:
    stats = {
        "raw": 0,
        "kept": 0,
        "small": 0,
        "low_visible": 0,
        "empty": 0,
        "query": 0,
        "gallery": 0,
        "train": 0,
    }
    query_counts: dict[tuple[int, int], int] = {}

    for scene_index, scene in enumerate(tqdm(scenes, desc=f"convert:{split}", unit="scene")):
        scene_dir = short_root / scene
        for csv_path in sorted(scene_dir.glob("gt_cam*.csv")):
            cam_id = int(csv_path.stem.replace("gt_cam", ""))
            vid_path = scene_dir / f"cam{cam_id}.mp4"
            if not vid_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df = df[df["frame"] % max(1, sample_rate) == 0]
            by_frame: dict[int, list[dict]] = {}
            for row in df.to_dict("records"):
                by_frame.setdefault(int(row["frame"]), []).append(row)

            cap = cv2.VideoCapture(str(vid_path))
            frame_no = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rows = by_frame.get(frame_no)
                if rows:
                    img_h, img_w = frame.shape[:2]
                    for row in rows:
                        stats["raw"] += 1
                        w = float(row["width"])
                        h = float(row["height"])
                        if w < min_w or h < min_h:
                            stats["small"] += 1
                            continue
                        x1, y1, x2, y2, visible_ratio = _clip_box(
                            float(row["left"]), float(row["top"]), w, h, img_w, img_h)
                        if visible_ratio < min_visible_ratio:
                            stats["low_visible"] += 1
                            continue
                        if x2 <= x1 or y2 <= y1:
                            stats["empty"] += 1
                            continue

                        local_pid = int(row["person_id"])
                        pid = pid_map[_scene_gid(scene_index, local_pid)]
                        crop = frame[y1:y2, x1:x2]
                        name = (
                            f"{pid:04d}_c{cam_id}s1_"
                            f"{scene_index:02d}{frame_no:05d}_{stats['kept'] % 100:02d}.jpg"
                        )

                        if split == "train":
                            out_path = out_train / name
                            stats["train"] += 1
                        else:
                            key = (pid, cam_id)
                            count = query_counts.get(key, 0)
                            if count < query_per_pid_cam:
                                out_path = out_query / name
                                query_counts[key] = count + 1
                                stats["query"] += 1
                            else:
                                out_path = out_gallery / name
                                stats["gallery"] += 1
                        _write_crop(out_path, crop, jpeg_quality)
                        stats["kept"] += 1
                frame_no += 1
            cap.release()

    return stats


def main() -> None:
    args = _parse_args()
    short_root = Path(args.short_root)
    out_root = Path(args.output)
    market_root = out_root / "Market-1501-v15.09.15"
    train_dir = market_root / "bounding_box_train"
    query_dir = market_root / "query"
    gallery_dir = market_root / "bounding_box_test"

    if market_root.exists() and not args.overwrite:
        raise SystemExit(
            f"{market_root} already exists. Use --overwrite to rebuild crops.")

    if args.overwrite and market_root.exists():
        import shutil
        shutil.rmtree(market_root)

    train_scenes, val_scenes = _train_val_scenes()
    train_pid_map = _build_pid_map(short_root, train_scenes)
    val_pid_map = _build_pid_map(short_root, val_scenes)

    print(f"[fastreid-data] train scenes={len(train_scenes)} pids={len(train_pid_map)}")
    print(f"[fastreid-data] val scenes={len(val_scenes)} pids={len(val_pid_map)}")
    print(f"[fastreid-data] output={market_root}")

    train_stats = _convert_split(
        short_root, train_scenes, train_dir, query_dir, gallery_dir, train_pid_map,
        args.sample_rate, args.min_w, args.min_h, args.min_visible_ratio,
        args.query_per_pid_cam, args.jpeg_quality, split="train")
    val_stats = _convert_split(
        short_root, val_scenes, train_dir, query_dir, gallery_dir, val_pid_map,
        args.sample_rate, args.min_w, args.min_h, args.min_visible_ratio,
        args.query_per_pid_cam, args.jpeg_quality, split="val")

    print(f"[fastreid-data] train stats={train_stats}")
    print(f"[fastreid-data] val stats={val_stats}")
    print("[fastreid-data] done")


if __name__ == "__main__":
    main()
