"""Build MMPTracking_10minute (short-format) from the full MMPTracking dataset.

The full set ships as per-scene zips of per-frame, per-camera JPGs + JSON labels:
    <split>/images/<session>/<scene>.zip   ->  <scene>/rgb_<frame5>_<cam>.jpg
    <split>/labels/<session>/<scene>.zip   ->  <scene>/rgb_<frame5>_<cam>.json
JSON label = {"<person_id>": [x1, y1, x2, y2], ...}  (absolute px, 640x360).

Each (scene, camera) becomes one video of up to --minutes minutes (15 fps source),
re-emitted in the established MMPTracking_short layout so every downstream tool
works unchanged:
    dataset/MMPTracking_10minute/<split>/<session>_<scene>/
        cam<N>.mp4
        gt_cam<N>.csv          (frame,person_id,left,top,width,height)

  train  <- MMPTracking_training/train       (sessions 63am, 64am -> kept separate)
  val    <- MMPTracking_validation/validation (session 64pm)

Frames are streamed straight out of the zip in memory (no extractall) to avoid
millions of small-file writes on the external disk.

Run:
    python scripts/build_mmp_10minute.py --minutes 10
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

FPS = 15                      # MMPTracking native frame rate
W, H = 640, 360
FNAME_RE = re.compile(r"rgb_(\d+)_(\d+)\.(jpg|json)$")

SPLITS = {
    "train": Path("dataset/MMPTracking/MMPTracking_training/train"),
    "val":   Path("dataset/MMPTracking/MMPTracking_validation/validation"),
}


def _scene_zips(split_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Yield (session, scene, images_zip, labels_zip) for a split."""
    out = []
    img_root = split_root / "images"
    lbl_root = split_root / "labels"
    for session_dir in sorted(img_root.iterdir()):
        if not session_dir.is_dir():
            continue
        session = session_dir.name
        for img_zip in sorted(session_dir.glob("*.zip")):
            scene = img_zip.stem
            lbl_zip = lbl_root / session / f"{scene}.zip"
            if lbl_zip.exists():
                out.append((session, scene, img_zip, lbl_zip))
    return out


def _index_members(zf: zipfile.ZipFile, ext: str, max_frame: int) -> dict:
    """cam -> {frame: member_name} for members of the given extension <= max_frame."""
    out: dict[int, dict[int, str]] = defaultdict(dict)
    for name in zf.namelist():
        m = FNAME_RE.search(name)
        if not m or m.group(3) != ext:
            continue
        frame, cam = int(m.group(1)), int(m.group(2))
        if frame <= max_frame:
            out[cam][frame] = name
    return out


def _load_boxes(zf: zipfile.ZipFile, name: str) -> dict[int, list]:
    """{pid: [x1,y1,x2,y2]} from a single JSON member."""
    try:
        data = json.loads(zf.read(name))
    except Exception:
        return {}
    per = {}
    for pid, box in data.items():
        if isinstance(box, list) and len(box) == 4:
            per[int(pid)] = [float(v) for v in box]
    return per


def process_scene(session: str, scene: str, img_zip: Path, lbl_zip: Path,
                  split: str, out_root: Path, max_frame: int) -> dict:
    out_dir = out_root / split / f"{session}_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"scene": f"{session}_{scene}", "cams": 0, "frames": 0, "boxes": 0}

    with zipfile.ZipFile(img_zip) as izf, zipfile.ZipFile(lbl_zip) as lzf:
        img_idx = _index_members(izf, "jpg", max_frame)
        lbl_idx = _index_members(lzf, "json", max_frame)

        for cam in sorted(img_idx):
            frames = sorted(img_idx[cam])
            if not frames:
                continue
            vw = cv2.VideoWriter(str(out_dir / f"cam{cam}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            gt_lines = ["frame,person_id,left,top,width,height"]
            out_idx = 0
            for fid in frames:
                buf = np.frombuffer(izf.read(img_idx[cam][fid]), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H))
                vw.write(img)
                lbl_name = lbl_idx.get(cam, {}).get(fid)
                if lbl_name:
                    for pid, (x1, y1, x2, y2) in _load_boxes(lzf, lbl_name).items():
                        if x2 <= x1 or y2 <= y1:
                            continue
                        gt_lines.append(
                            f"{out_idx},{pid},{x1:.1f},{y1:.1f},"
                            f"{x2 - x1:.1f},{y2 - y1:.1f}")
                        stats["boxes"] += 1
                out_idx += 1
            vw.release()
            (out_dir / f"gt_cam{cam}.csv").write_text("\n".join(gt_lines) + "\n")
            stats["cams"] += 1
            stats["frames"] += out_idx
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="Max minutes of footage per (scene, camera).")
    ap.add_argument("--out-root", default="dataset/MMPTracking_10minute", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N scenes per split (smoke test).")
    args = ap.parse_args()

    max_frame = int(args.minutes * 60 * FPS) - 1
    for split in args.splits:
        split_root = SPLITS[split]
        if not split_root.exists():
            print(f"[SKIP] split root missing: {split_root}")
            continue
        scenes = _scene_zips(split_root)
        if args.limit:
            scenes = scenes[:args.limit]
        print(f"\n=== {split}: {len(scenes)} scenes (cap {args.minutes} min) ===",
              flush=True)
        tot_frames = tot_boxes = 0
        for i, (session, scene, iz, lz) in enumerate(scenes, 1):
            st = process_scene(session, scene, iz, lz, split, args.out_root, max_frame)
            tot_frames += st["frames"]
            tot_boxes += st["boxes"]
            print(f"  [{i}/{len(scenes)}] {st['scene']:>26}  "
                  f"cams={st['cams']} frames={st['frames']} boxes={st['boxes']}",
                  flush=True)
        print(f"  -> {split}: {tot_frames} frame-images, {tot_boxes} boxes", flush=True)


if __name__ == "__main__":
    main()
