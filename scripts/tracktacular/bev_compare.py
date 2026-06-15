#!/usr/bin/env python3
"""Fair 3-way world-space comparison: current pipeline vs anchor-guided vs
TrackTacular, all scored against the SAME topdown GT with the SAME metric.

TrackTacular `test` writes (in its log dir):
    mota_gt.txt    seq,frame,pid,-1,-1,-1,-1,1,gx,gy,-1     (topdown GT, step-frames)
    mota_pred.txt  seq,frame,trkid,-1,-1,-1,-1,score,gx,gy  (its BEV tracks)
We reuse mota_gt.txt as the canonical GT for ALL methods, and produce aligned
predictions for the DeepStream exports (current / anchor) by joining their
per-camera predictions (frame,cam,ltid,global_id) with tracklet_bev.csv
(frame,cam,ltid,world_x_mm,world_y_mm), converting mm->grid via the per-env
affine, and remapping original frame -> step-frame (orig // frame_step).

Metric: motmetrics IDF1/MOTA/MODA at a 1 m gate (TrackTacular's mot_bev protocol).
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
import motmetrics as mm

# grid->world(mm) affine per env (must match mmptracking_dataset.py)
AFFINE = {
    "industry_safety": np.array([[1.706, 33.103, -6301.5],
                                 [36.942, 3.970, -10626.8],
                                 [0.0, 0.0, 1.0]]),
}


def _idf1(gt_fid, gt_id, gt_xy, pr_fid, pr_id, pr_xy, gate_m=1.0, scale=0.001):
    """motmetrics over frames; positions in mm, gate in metres."""
    acc = mm.MOTAccumulator()
    frames = np.union1d(np.unique(gt_fid), np.unique(pr_fid)).astype(int)
    for f in frames:
        g = gt_fid == f
        p = pr_fid == f
        C = mm.distances.norm2squared_matrix(gt_xy[g] * scale, pr_xy[p] * scale,
                                             max_d2=gate_m ** 2)
        acc.update(gt_id[g].astype(int).tolist(), pr_id[p].astype(int).tolist(),
                   np.sqrt(C), frameid=int(f))
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=['idf1', 'mota', 'idp', 'idr', 'num_switches'],
                   name='x')
    return s.iloc[0]


def load_gt(gt_path):
    g = np.loadtxt(gt_path, delimiter=',') if Path(gt_path).read_text().count(',') \
        else np.loadtxt(gt_path)
    # cols: seq,frame,pid,...,gx(8),gy(9)
    fid, pid, gx, gy = g[:, 1], g[:, 2], g[:, 8], g[:, 9]
    xy = (AFFINE_ENV @ np.stack([gx, gy, np.ones_like(gx)]))[:2].T  # grid->mm
    return fid.astype(int), pid.astype(int), xy


def load_tt(pred_path):
    d = np.loadtxt(pred_path, delimiter=',')
    fid, tid, gx, gy = d[:, 1], d[:, 2], d[:, 8], d[:, 9]
    xy = (AFFINE_ENV @ np.stack([gx, gy, np.ones_like(gx)]))[:2].T
    return fid.astype(int), tid.astype(int), xy


def load_export(pred_dir, gt_frames, frame_step):
    """DeepStream export -> (step_frame, gid, xy_mm) aligned to GT step-frames."""
    pred_dir = Path(pred_dir)
    bev = pd.read_csv(pred_dir / "tracklet_bev.csv")  # frame_no_cam,cam_id,local_track_id,global_id,world_x,world_y
    # use the (possibly remapped) global_id from the cam predictions
    key2gid = {}
    for fp in pred_dir.glob("cam_*_predictions.csv"):
        for r in csv.DictReader(open(fp)):
            key2gid[(int(r["cam_id"]), int(r["local_track_id"]),
                     int(r["frame_no_cam"]))] = int(float(r["global_id"]))
    want = set(int(f) * frame_step for f in gt_frames)   # GT step-frame -> orig
    fid, gid, xs, ys = [], [], [], []
    for r in bev.itertuples():
        of = int(r.frame_no_cam)
        if of not in want:
            continue
        g = key2gid.get((int(r.cam_id), int(r.local_track_id), of), int(r.global_id))
        if g < 0:
            continue
        fid.append(of // frame_step); gid.append(g)
        xs.append(r.world_x); ys.append(r.world_y)
    return (np.array(fid), np.array(gid), np.stack([xs, ys], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="mota_gt.txt from TrackTacular test")
    ap.add_argument("--tt-pred", help="mota_pred.txt from TrackTacular test")
    ap.add_argument("--current-dir", help="DeepStream export dir (online baseline)")
    ap.add_argument("--anchor-dir", help="anchor-guided export dir")
    ap.add_argument("--env", default="industry_safety")
    ap.add_argument("--frame-step", type=int, default=2)
    args = ap.parse_args()

    global AFFINE_ENV
    AFFINE_ENV = AFFINE[args.env]

    gt_fid, gt_id, gt_xy = load_gt(args.gt)
    gt_frames = np.unique(gt_fid)
    print(f"shared GT: {len(gt_frames)} test frames, {len(gt_id)} dets, "
          f"{len(np.unique(gt_id))} identities\n")
    rows = []
    if args.tt_pred:
        f, i, xy = load_tt(args.tt_pred)
        rows.append(("TrackTacular (BEV)", _idf1(gt_fid, gt_id, gt_xy, f, i, xy)))
    if args.current_dir:
        f, i, xy = load_export(args.current_dir, gt_frames, args.frame_step)
        rows.append(("Current (online)", _idf1(gt_fid, gt_id, gt_xy, f, i, xy)))
    if args.anchor_dir:
        f, i, xy = load_export(args.anchor_dir, gt_frames, args.frame_step)
        rows.append(("Anchor-guided", _idf1(gt_fid, gt_id, gt_xy, f, i, xy)))

    print(f"{'method':22s} {'IDF1':>7} {'MOTA':>7} {'IDP':>7} {'IDR':>7} {'IDsw':>6}")
    for name, s in rows:
        print(f"{name:22s} {s['idf1']:7.3f} {s['mota']:7.3f} {s['idp']:7.3f} "
              f"{s['idr']:7.3f} {int(s['num_switches']):6d}")


if __name__ == "__main__":
    main()
