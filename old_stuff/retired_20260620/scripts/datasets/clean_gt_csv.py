#!/usr/bin/env python3
"""Model-assisted cleaning of MMPTracking GT csv files (retail-focused).

Diagnosis (same root cause as clean_yolo_labels.py): MMP GT annotates every
person's *projected* box in every camera, including people fully occluded by
shelves/counters/walls. In retail those phantom boxes sit over background, so
ReID crops built from them are pure shelf/floor → they poison identity training.

This is the GT-csv analogue of clean_yolo_labels.py. While clean_yolo_labels.py
cleans the extracted YOLO `.txt` labels (detector training), this cleans the
`gt_cam<N>.csv` files that feed the ReID crop cache (build_reid_crop_cache.py
with --prefer-clean-gt reads `gt_cam<N>_clean.csv` when present).

Fix: an INDEPENDENT COCO-pretrained verifier (YOLO11x) judges whether a person
is actually visible in each GT box. Keep a GT row iff a verifier person
detection (conf >= conf) overlaps it (IoU >= iou). Optionally drop a box that is
>occ contained by a *closer* (lower-bottom) GT box (person-on-person occlusion;
off by default — the phantom/behind-shelf removal is the verifier IoU step).

Operates per scene directory, reading `cam<N>.mp4` + `gt_cam<N>.csv`, writing
`gt_cam<N>_clean.csv` (identical schema; phantom rows dropped, person_id kept).

Usage:
  # all retail train+val scenes in MMPTracking_10minute
  python scripts/datasets/clean_gt_csv.py \
      --root dataset/MMPTracking_10minute --scene-glob '*retail*' \
      --verifier yolo11x.pt --iou 0.3 --conf 0.25 --stride 5 \
      --qa-dir output/gt_clean_qa

  # one scene, dry-run stats only (no write)
  python scripts/datasets/clean_gt_csv.py \
      --root dataset/MMPTracking_10minute --scene 63am_retail_0 --dry-run
"""
from __future__ import annotations
import argparse
import fnmatch
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

GT_COLS = ["frame", "person_id", "left", "top", "width", "height"]


def iou_matrix(gt_xyxy: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """IoU of each GT box (rows) against each verifier box (cols)."""
    if len(gt_xyxy) == 0 or len(vb) == 0:
        return np.zeros((len(gt_xyxy), len(vb)), np.float32)
    out = np.zeros((len(gt_xyxy), len(vb)), np.float32)
    va = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
    for i, g in enumerate(gt_xyxy):
        ix1 = np.maximum(g[0], vb[:, 0]); iy1 = np.maximum(g[1], vb[:, 1])
        ix2 = np.minimum(g[2], vb[:, 2]); iy2 = np.minimum(g[3], vb[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        ga = (g[2] - g[0]) * (g[3] - g[1])
        denom = ga + va - inter
        out[i] = np.where(denom > 0, inter / denom, 0.0)
    return out


def contain_frac(a, b):  # fraction of box a covered by box b (xyxy)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    return inter / aa if aa > 0 else 0.0


def find_scenes(root: Path, scene: str | None, scene_glob: str | None,
                splits: list[str]) -> list[Path]:
    if scene:
        # scene may be a bare name (search splits) or split/name
        for sp in splits:
            cand = root / sp / scene
            if cand.is_dir():
                return [cand]
        cand = root / scene
        return [cand] if cand.is_dir() else []
    out: list[Path] = []
    for sp in splits:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for d in sorted(sp_dir.iterdir()):
            if d.is_dir() and (scene_glob is None or fnmatch.fnmatch(d.name, scene_glob)):
                out.append(d)
    return out


def clean_cam(scene_dir: Path, cam_id: int, model: YOLO, args,
              qa_drop: list, qa_keep: list) -> dict:
    gt_path = scene_dir / f"gt_cam{cam_id}.csv"
    vid_path = scene_dir / f"cam{cam_id}.mp4"
    df = pd.read_csv(gt_path)
    cap = cv2.VideoCapture(str(vid_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError(f"cannot read video size: {vid_path}")

    # frames we must verify (have GT and pass the stride filter)
    gt_frames = sorted(int(f) for f in df["frame"].unique() if int(f) % args.stride == 0)
    gt_frame_set = set(gt_frames)
    by_frame = {f: g for f, g in df.groupby("frame")}

    keep_mask = pd.Series(False, index=df.index)
    st = {"boxes": 0, "kept": 0, "drop_occ": 0, "drop_unverified": 0}

    # sequential decode; buffer frames into batches for the verifier
    batch_imgs: list[np.ndarray] = []
    batch_fno: list[int] = []

    def flush():
        if not batch_imgs:
            return
        res = model.predict(batch_imgs, conf=args.conf, classes=[0],
                            verbose=False, imgsz=args.imgsz)
        for fno, r in zip(batch_fno, res):
            vb = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                  else np.zeros((0, 4), np.float32))
            g = by_frame[fno]
            # clip GT boxes to frame for matching
            x1 = g["left"].to_numpy(float).clip(0, W)
            y1 = g["top"].to_numpy(float).clip(0, H)
            x2 = (g["left"] + g["width"]).to_numpy(float).clip(0, W)
            y2 = (g["top"] + g["height"]).to_numpy(float).clip(0, H)
            gt_xyxy = np.stack([x1, y1, x2, y2], axis=1)
            # person-on-person occlusion (optional; off when occ>=1.0)
            keep_occ = np.ones(len(g), bool)
            if args.occ < 1.0:
                for a in range(len(g)):
                    for b in range(len(g)):
                        if a == b:
                            continue
                        if gt_xyxy[b][3] > gt_xyxy[a][3] and \
                           contain_frac(gt_xyxy[a], gt_xyxy[b]) > args.occ:
                            keep_occ[a] = False
                            break
            ious = iou_matrix(gt_xyxy, vb)
            best = ious.max(axis=1) if vb.size else np.zeros(len(g))
            for k, (idx, row) in enumerate(g.iterrows()):
                st["boxes"] += 1
                deg = (x2[k] <= x1[k]) or (y2[k] <= y1[k])  # box fully off-frame
                if not keep_occ[k]:
                    st["drop_occ"] += 1
                elif (not deg) and best[k] >= args.iou:
                    st["kept"] += 1
                    keep_mask.loc[idx] = True
                    if args.qa_dir and len(qa_keep) < 80 and np.random.rand() < 0.02:
                        qa_keep.append((str(vid_path), fno, gt_xyxy[k]))
                else:
                    st["drop_unverified"] += 1
                    if args.qa_dir and len(qa_drop) < 80 and np.random.rand() < 0.1:
                        qa_drop.append((str(vid_path), fno, gt_xyxy[k]))
        batch_imgs.clear()
        batch_fno.clear()

    pos = 0
    ok, frame = cap.read()
    while ok:
        if pos in gt_frame_set:
            batch_imgs.append(frame)
            batch_fno.append(pos)
            if len(batch_imgs) >= args.batch:
                flush()
        pos += 1
        ok, frame = cap.read()
    flush()
    cap.release()

    # rows on frames we skipped (stride) are kept as-is so the clean csv still
    # covers every frame the crop cache might sample at a finer rate
    skipped = ~df["frame"].isin(gt_frame_set)
    keep_mask = keep_mask | skipped
    clean_df = df[keep_mask].copy()

    if not args.dry_run:
        out = scene_dir / f"gt_cam{cam_id}_clean.csv"
        clean_df.to_csv(out, index=False)
    return st


def render_qa(qa_dir: Path, name: str, items: list):
    qd = Path(qa_dir); qd.mkdir(parents=True, exist_ok=True)
    caps: dict[str, cv2.VideoCapture] = {}
    crops = []
    for vid, fno, box in items[:48]:
        cap = caps.get(vid) or cv2.VideoCapture(vid)
        caps[vid] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, im = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(cv2.resize(im[y1:y2, x1:x2], (64, 128)))
    for c in caps.values():
        c.release()
    if not crops:
        return
    rows = []
    for r in range(0, len(crops), 8):
        row = crops[r:r + 8]
        while len(row) < 8:
            row.append(np.zeros((128, 64, 3), np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(qd / f"montage_{name}.jpg"), np.vstack(rows))
    print(f"QA: {qd}/montage_{name}.jpg ({len(crops)} crops)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/MMPTracking_10minute")
    ap.add_argument("--scene", default=None, help="single scene (bare name or split/name)")
    ap.add_argument("--scene-glob", default="*retail*", help="glob over scenes when --scene not given")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--verifier", default="yolo11x.pt")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--occ", type=float, default=1.0,
                    help="drop box if >occ contained by a closer box (>=1.0 disables)")
    ap.add_argument("--stride", type=int, default=5,
                    help="verify every Nth GT frame (5 matches crop-cache sample_rate)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--qa-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="compute stats, do not write _clean.csv")
    args = ap.parse_args()

    root = Path(args.root)
    scenes = find_scenes(root, args.scene, args.scene_glob, args.splits)
    if not scenes:
        raise SystemExit(f"No scenes matched (root={root}, scene={args.scene}, glob={args.scene_glob})")
    print(f"[clean-gt] {len(scenes)} scene(s); verifier={args.verifier} "
          f"iou={args.iou} conf={args.conf} stride={args.stride} occ={args.occ} "
          f"{'(DRY-RUN)' if args.dry_run else ''}")
    model = YOLO(args.verifier)
    qa_drop, qa_keep = [], []
    grand = {"boxes": 0, "kept": 0, "drop_occ": 0, "drop_unverified": 0}

    for scene_dir in scenes:
        cams = sorted(int(p.stem.replace("gt_cam", ""))
                      for p in scene_dir.glob("gt_cam*.csv")
                      if p.stem.replace("gt_cam", "").isdigit())
        sc = {"boxes": 0, "kept": 0, "drop_occ": 0, "drop_unverified": 0}
        for cam_id in cams:
            st = clean_cam(scene_dir, cam_id, model, args, qa_drop, qa_keep)
            for k in sc:
                sc[k] += st[k]
        dropped = sc["boxes"] - sc["kept"]
        print(f"[{scene_dir.name}] boxes={sc['boxes']} kept={sc['kept']} "
              f"drop_unverified={sc['drop_unverified']} drop_occ={sc['drop_occ']} "
              f"({100 * dropped / max(1, sc['boxes']):.1f}% dropped)")
        for k in grand:
            grand[k] += sc[k]

    if args.qa_dir:
        render_qa(Path(args.qa_dir), "dropped", qa_drop)
        render_qa(Path(args.qa_dir), "kept", qa_keep)

    tot = grand["boxes"]
    print(f"\n[TOTAL] {tot} verified boxes -> kept {grand['kept']} "
          f"({100 * grand['kept'] / max(1, tot):.1f}%), dropped "
          f"unverified={grand['drop_unverified']} occ={grand['drop_occ']} "
          f"({100 * (tot - grand['kept']) / max(1, tot):.1f}%)")


if __name__ == "__main__":
    main()
