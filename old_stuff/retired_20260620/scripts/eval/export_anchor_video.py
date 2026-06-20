#!/usr/bin/env python3
"""Render an offline anchor-guided result to a tiled multi-camera MP4.

Reads the stage-2/stage-3 output (cam_*_predictions.csv with a global_id column)
and overlays each detection box on its source video, colour-coded by global_id so
the SAME person carries the SAME colour + GID label across every camera. Cameras
are tiled into a grid to make the cross-camera identity consistency visible.

Usage:
  python scripts/eval/export_anchor_video.py \
      --pred-dir output/eval/theirft_63am_lobby_3_anchor \
      --scene 63am_lobby_3 \
      --out output/videos/lobby_3_anchor.mp4 \
      --fps 15 --max-frames 1500
"""
from __future__ import annotations
import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def gid_color(gid: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR colour per global id."""
    h = (gid * 0.61803398875) % 1.0          # golden-ratio hue hashing
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="anchor/stcra out-dir with cam_*_predictions.csv")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--tile-w", type=int, default=640)
    ap.add_argument("--tile-h", type=int, default=360)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    scene_dir = Path(args.short_root) / args.scene
    cam_csvs = sorted(pred_dir.glob("cam_*_predictions.csv"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if not cam_csvs:
        raise SystemExit(f"no cam_*_predictions.csv in {pred_dir}")

    # load per-cam predictions grouped by frame; open each source video
    cams = []
    for csv in cam_csvs:
        src = int(csv.stem.split("_")[1])           # cam_<src>_predictions.csv
        vid = scene_dir / f"cam{src + 1}.mp4"        # src 0 -> cam1.mp4
        if not vid.exists():
            raise SystemExit(f"video not found: {vid}")
        df = pd.read_csv(csv)
        df = df[df["global_id"] >= 0]
        by_frame = {int(f): g for f, g in df.groupby("frame_no_cam")}
        cap = cv2.VideoCapture(str(vid))
        cams.append({"src": src, "cap": cap, "by_frame": by_frame,
                     "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})

    n_cam = len(cams)
    cols = 2 if n_cam <= 4 else 3
    rows = int(np.ceil(n_cam / cols))
    grid_w, grid_h = cols * args.tile_w, rows * args.tile_h
    total = min(c["n"] for c in cams)
    if args.max_frames:
        total = min(total, args.max_frames)

    args_out = Path(args.out); args_out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args_out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (grid_w, grid_h))
    print(f"[export] {args.scene}: {n_cam} cams, {total} frames -> {args_out} "
          f"({grid_w}x{grid_h})")

    for fno in range(total):
        canvas = np.zeros((grid_h, grid_w, 3), np.uint8)
        for i, c in enumerate(cams):
            ok, frame = c["cap"].read()
            if not ok:
                frame = np.zeros((args.tile_h, args.tile_w, 3), np.uint8)
            frame = cv2.resize(frame, (args.tile_w, args.tile_h))
            sx, sy = args.tile_w / 640.0, args.tile_h / 360.0  # pred-space is 640x360
            for r in c["by_frame"].get(fno, pd.DataFrame()).itertuples():
                gid = int(r.global_id)
                x1, y1 = int(r.left * sx), int(r.top * sy)
                x2, y2 = int((r.left + r.width) * sx), int((r.top + r.height) * sy)
                col = gid_color(gid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"ID{gid}", (x1, max(11, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(frame, f"cam{c['src'] + 1}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            gy, gx = divmod(i, cols)
            canvas[gy * args.tile_h:(gy + 1) * args.tile_h,
                   gx * args.tile_w:(gx + 1) * args.tile_w] = frame
        vw.write(canvas)
        if fno % 300 == 0:
            print(f"  frame {fno}/{total}", flush=True)

    vw.release()
    for c in cams:
        c["cap"].release()
    print(f"[export] done -> {args_out}")


if __name__ == "__main__":
    main()
