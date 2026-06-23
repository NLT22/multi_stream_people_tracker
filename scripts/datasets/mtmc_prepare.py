#!/usr/bin/env python3
"""Prepare MTMC_Tracking_2026 (AI-City warehouse MTMC) for YOLO + ReID retraining.

ONE decode pass emits both datasets (decoding 1080p video is the bottleneck, so we
read each camera video once and write the YOLO image+label AND the ReID person crops
for every kept frame).

Source layout (per warehouse):
    dataset/MTMC_Tracking_2026/<split>/Warehouse_XXX/
        ground_truth.json   {frame: [ {object type, object id, 2d bounding box visible:{Cam:[x1,y1,x2,y2]}, ...} ]}
        calibration.json
        videos/Camera_XXXX.mp4   (1920x1080, 30fps, 9000 frames)

Outputs:
    dataset/mtmc_yolo/
        images/{train,val}/<wh>_<cam>_f<FFFFFF>.jpg
        labels/{train,val}/<wh>_<cam>_f<FFFFFF>.txt    (YOLO: class 0, normalized xywh)
        dataset.yaml
    dataset/mtmc_reid_cache/
        {train,val}/<pid>/<pid>_<cam>_f<FFFFFF>.jpg    (pid = "<wh>_<objid>", cross-camera)
        {train,val}/manifest.csv

Memory-safe: ground_truth.json is streamed with ijson; only boxes on kept (subsampled)
frames are held. No GPU used.

Smoke test (1 warehouse, 1 camera, a few frames):
    python scripts/datasets/mtmc_prepare.py --split val --warehouses Warehouse_020 \
        --max-cams 1 --max-frames 5 --stride 30 --out-suffix _smoke

Full train prep (1 fps, all train warehouses):
    python scripts/datasets/mtmc_prepare.py --split train --stride 30
    python scripts/datasets/mtmc_prepare.py --split val   --stride 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import ijson

PERSON = "Person"
FRAME_W, FRAME_H = 1920, 1080   # constant across the dataset (verified)
CLASS_ID = 0


def _kept_boxes(gt_path: Path, stride: int, max_frames: int | None,
                cams_filter: set[str] | None):
    """Stream ground_truth.json -> {camera: {frame_int: [(pid, x1,y1,x2,y2), ...]}}.

    Only Person objects on frames where frame % stride == 0 are retained, so memory
    stays ~ (#person boxes / stride)."""
    cam2frames: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    kept = 0
    with open(gt_path, "rb") as f:
        for fk, objs in ijson.kvitems(f, ""):
            fi = int(fk)
            if fi % stride != 0:
                continue
            for o in objs:
                if o.get("object type") != PERSON:
                    continue
                pid = o.get("object id")
                for cam, b in (o.get("2d bounding box visible") or {}).items():
                    if cams_filter and cam not in cams_filter:
                        continue
                    cam2frames[cam][fi].append((pid, float(b[0]), float(b[1]),
                                                float(b[2]), float(b[3])))
            kept += 1
            if max_frames and kept >= max_frames:
                break
    return cam2frames


def _clamp_box(x1, y1, x2, y2):
    x1 = max(0.0, min(x1, FRAME_W)); x2 = max(0.0, min(x2, FRAME_W))
    y1 = max(0.0, min(y1, FRAME_H)); y2 = max(0.0, min(y2, FRAME_H))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _process_camera(video: Path, wh: str, cam: str, frames: dict[int, list],
                    split: str, args, reid_writer) -> tuple[int, int]:
    """Decode one camera video sequentially; emit YOLO image+label and ReID crops."""
    yolo_img_dir = args.yolo_root / "images" / split
    yolo_lbl_dir = args.yolo_root / "labels" / split
    n_imgs = n_crops = 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [warn] cannot open {video}", file=sys.stderr)
        return 0, 0
    want = set(frames)
    idx = 0
    last = max(want) if want else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            stem = f"{wh}_{cam}_f{idx:06d}"
            boxes = frames[idx]
            # ---- YOLO ----
            if args.emit_yolo:
                lines = []
                for pid, x1, y1, x2, y2 in boxes:
                    cb = _clamp_box(x1, y1, x2, y2)
                    if cb is None:
                        continue
                    bx1, by1, bx2, by2 = cb
                    if (by2 - by1) < args.min_h or (bx2 - bx1) < args.min_w:
                        continue
                    xc = (bx1 + bx2) / 2 / FRAME_W; yc = (by1 + by2) / 2 / FRAME_H
                    w = (bx2 - bx1) / FRAME_W; h = (by2 - by1) / FRAME_H
                    lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                if lines or args.keep_empty:
                    cv2.imwrite(str(yolo_img_dir / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    (yolo_lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
                    n_imgs += 1
            # ---- ReID crops ----
            if args.emit_reid:
                for pid, x1, y1, x2, y2 in boxes:
                    cb = _clamp_box(x1, y1, x2, y2)
                    if cb is None:
                        continue
                    bx1, by1, bx2, by2 = (int(round(v)) for v in cb)
                    if (by2 - by1) < args.reid_min_h or (bx2 - bx1) < args.reid_min_w:
                        continue
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size == 0:
                        continue
                    pid_s = f"{wh}_{pid}"
                    pid_int = args.pid_map.setdefault(pid_s, len(args.pid_map))
                    cam_id = int(cam.split("_")[-1])
                    d = args.reid_root / split / pid_s
                    d.mkdir(parents=True, exist_ok=True)
                    fn = f"{pid_s}_{cam}_f{idx:06d}.jpg"
                    cv2.imwrite(str(d / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    # columns match CachedReidDataset (finetune_reid): scene,pid,cam_id,rel_path
                    reid_writer.writerow([wh, pid_int, cam_id, idx, f"{split}/{pid_s}/{fn}"])
                    n_crops += 1
        idx += 1
    cap.release()
    return n_imgs, n_crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("dataset/MTMC_Tracking_2026"))
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--warehouses", nargs="*", default=None,
                    help="subset e.g. Warehouse_000 (default: all in split)")
    ap.add_argument("--stride", type=int, default=30,
                    help="keep every Nth frame (30 -> 1fps from 30fps source)")
    ap.add_argument("--emit", default="both", choices=["yolo", "reid", "both"])
    ap.add_argument("--out-suffix", default="", help="suffix for output dirs (e.g. _smoke)")
    # YOLO box filters (px on 1920x1080)
    ap.add_argument("--min-h", type=float, default=12)
    ap.add_argument("--min-w", type=float, default=6)
    ap.add_argument("--keep-empty", action="store_true",
                    help="also write frames with no person (negatives)")
    # ReID crop filters (px) — ReID needs more pixels than detection
    ap.add_argument("--reid-min-h", type=float, default=64)
    ap.add_argument("--reid-min-w", type=float, default=24)
    # smoke limits
    ap.add_argument("--max-cams", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="kept frames per warehouse (GT scan cap)")
    args = ap.parse_args()

    args.emit_yolo = args.emit in ("yolo", "both")
    args.emit_reid = args.emit in ("reid", "both")
    args.pid_map = {}   # "<wh>_<objid>" -> contiguous int (for the manifest pid column)
    args.yolo_root = Path(f"dataset/mtmc_yolo{args.out_suffix}")
    args.reid_root = Path(f"dataset/mtmc_reid_cache{args.out_suffix}")
    # YOLO val dir uses the val split name; reid likewise. (MTMC has explicit splits.)
    split = args.split

    whs = args.warehouses or sorted(
        d.name for d in (args.root / split).iterdir() if d.is_dir())
    if args.emit_yolo:
        (args.yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    reid_writer = reid_fp = None
    if args.emit_reid:
        (args.reid_root / split).mkdir(parents=True, exist_ok=True)
        reid_fp = open(args.reid_root / split / "manifest.csv", "a", newline="")
        reid_writer = csv.writer(reid_fp)
        if reid_fp.tell() == 0:
            reid_writer.writerow(["scene", "pid", "cam_id", "frame", "rel_path"])

    tot_imgs = tot_crops = 0
    pid_set = set()
    for wh in whs:
        wdir = args.root / split / wh
        gt = wdir / "ground_truth.json"
        if not gt.exists():
            print(f"[skip] {wh}: no ground_truth.json (test split has GT withheld)")
            continue
        cams = sorted(p.stem for p in (wdir / "videos").glob("*.mp4"))
        if args.max_cams:
            cams = cams[: args.max_cams]
        print(f"[{wh}] streaming GT (stride={args.stride}) for {len(cams)} cams ...")
        cam2frames = _kept_boxes(gt, args.stride, args.max_frames, set(cams))
        for cam in cams:
            frames = cam2frames.get(cam, {})
            if not frames:
                continue
            ni, nc = _process_camera(wdir / "videos" / f"{cam}.mp4", wh, cam,
                                     frames, split, args, reid_writer)
            tot_imgs += ni; tot_crops += nc
            for fr in frames.values():
                for b in fr:
                    pid_set.add(f"{wh}_{b[0]}")
            print(f"  {cam}: {ni} imgs, {nc} crops")
    if reid_fp:
        reid_fp.close()
        with open(args.reid_root / f"pid_map_{split}.csv", "w", newline="") as pf:
            w = csv.writer(pf); w.writerow(["pid_str", "pid_int"])
            for k, v in args.pid_map.items():
                w.writerow([k, v])

    if args.emit_yolo:
        import yaml
        yml = args.yolo_root / "dataset.yaml"
        cfg = {}
        if yml.exists():
            cfg = yaml.safe_load(yml.read_text()) or {}
        cfg.update({"path": str(args.yolo_root.resolve()),
                    "train": "images/train", "val": "images/val",
                    "names": {0: "person"}})
        yml.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"[yolo] dataset.yaml -> {yml}")

    print(f"\nDONE split={split}: {tot_imgs} images, {tot_crops} reid crops, "
          f"{len(pid_set)} distinct pids")


if __name__ == "__main__":
    main()
