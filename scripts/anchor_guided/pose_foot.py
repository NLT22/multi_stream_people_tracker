#!/usr/bin/env python3
"""Pose-based foot localization (paper's STCRA F_g, eq.2) using YOLO11n-pose.

For each detection crop, run pose; foot = midpoint of the two ankle keypoints when
their confidence >= tau, else bbox bottom-center (their fallback). Project the foot
to world via the MMP calibration -> pose_bev.csv with the SAME columns as
tracklet_bev.csv, so run_stcra_literal.py can use it as the world source.

This replaces the bbox-foot world coords (noisy) with pose-ankle world coords, to run
STCRA exactly like their pipeline (which uses HigherHRNet ankle keypoints).
"""
from __future__ import annotations
import argparse, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reid.geometry import GroundPlaneGeometry
from src.dataset.mmp_tracking import MMPTrackingShortDataset
import json

L_ANK, R_ANK = 15, 16   # COCO keypoint indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mmp-root", default="dataset/MMPTracking")
    ap.add_argument("--tau", type=float, default=0.5, help="ankle keypoint conf threshold")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from ultralytics import YOLO
    pose = YOLO("models/pose/yolo11n-pose.pt")

    env = args.scene.split("_", 1)[1].rsplit("_", 1)[0]
    calib = json.load(open(Path(args.mmp_root) / "MMPTracking_training/train"
                           / "calibrations" / env / "calibrations.json"))
    geo = GroundPlaneGeometry(calib)
    ds = MMPTrackingShortDataset(str(args.short_root), args.scene)
    cam_ids = ds.get_cam_ids()
    pred_dir = Path(args.pred_dir)
    scene_dir = Path(args.short_root) / args.scene

    rows = []  # tracklet_id, frame_no_cam, cam_id(src), local_track_id, global_id, world_x, world_y
    for src, cam in enumerate(cam_ids):
        csvf = pred_dir / f"cam_{src}_predictions.csv"
        if not csvf.exists():
            continue
        df = pd.read_csv(csvf)
        by_frame = {f: g for f, g in df.groupby("frame_no_cam")}
        cap = cv2.VideoCapture(str(scene_dir / f"cam{cam}.mp4"))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        crops, meta = [], []   # meta: (frame, ltid, x1, y1, x2, y2)

        def flush():
            if not crops:
                return
            res = pose.predict(crops, verbose=False, imgsz=128)
            for (fr, lt, x1, y1, x2, y2), r in zip(meta, res):
                u = (x1 + x2) / 2.0; v = float(y2)          # bbox-foot fallback
                if r.keypoints is not None and len(r.keypoints) > 0:
                    # pick the largest detected person in the crop
                    if r.boxes is not None and len(r.boxes) > 1:
                        bi = int(np.argmax((r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()))
                    else:
                        bi = 0
                    kxy = r.keypoints.xy[bi].cpu().numpy()       # (17,2) crop coords
                    kcf = r.keypoints.conf[bi].cpu().numpy() if r.keypoints.conf is not None else np.zeros(17)
                    if kcf[L_ANK] >= args.tau and kcf[R_ANK] >= args.tau:
                        au = (kxy[L_ANK, 0] + kxy[R_ANK, 0]) / 2.0 + x1
                        av = (kxy[L_ANK, 1] + kxy[R_ANK, 1]) / 2.0 + y1
                        u, v = au, av
                w = geo.foot_to_world(cam, u, v)
                if w is not None:
                    rows.append((0, fr, src, lt, 0, round(float(w[0]), 3), round(float(w[1]), 3)))
            crops.clear(); meta.clear()

        fidx = -1
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fidx += 1
            g = by_frame.get(fidx)
            if g is None:
                continue
            for r in g.itertuples():
                x1 = max(0, int(r.left)); y1 = max(0, int(r.top))
                x2 = min(W, int(r.left + r.width)); y2 = min(H, int(r.top + r.height))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crops.append(im[y1:y2, x1:x2]); meta.append((fidx, int(r.local_track_id), x1, y1, x2, y2))
                if len(crops) >= args.batch:
                    flush()
        flush()
        cap.release()
        print(f"  cam{cam} (src{src}): {sum(1 for r in rows if r[2]==src)} foot points")

    out = pred_dir / "pose_bev.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tracklet_id", "frame_no_cam", "cam_id", "local_track_id",
                     "global_id", "world_x", "world_y"])
        wr.writerows(rows)
    print(f"[pose-foot] wrote {len(rows)} pose-based world points -> {out}")


if __name__ == "__main__":
    main()
