#!/usr/bin/env python3
"""MTMC tracklet-level world-coordinate linker (geometry-first, fragmentation-robust).

The per-frame position linker (mtmc_position_linker.py) re-derives identity every frame
and spawns a fresh global id whenever a person's detection drops out for a moment — so a
17-person scene fragments into ~26 ids. This linker instead works on whole TRACKLETS:

  * a tracklet = one (cam, local_track_id) — the NvDCF local id is stable within a camera,
    so a tracklet already spans the person's continuous stay in that camera (no per-frame
    fragmentation to undo),
  * each detection's foot point is back-projected to world (x, y) via WarehouseCalibration,
  * tracklets are linked into global identities by two edge types, then union-find:
      - SPATIAL (cross-camera only): two tracklets from DIFFERENT cameras that co-occur in
        time and whose world positions coincide (median distance over the temporal overlap
        < --spatial-thr) are the same person seen from two views,
      - TEMPORAL (same or cross camera): a tracklet that ENDS where another BEGINS within
        --temporal-gap frames and --endpoint-thr world units — a camera hand-off or an
        NvDCF id-switch on one person. Two tracklets from the SAME camera that *overlap*
        in time are always different people and are never linked.

Outputs an assign-csv (group,cam_id,frame_no,local_track_id,global_id) — drop-in for
score_mtmc_idf1.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--group", default="w")
    ap.add_argument("--spatial-thr", type=float, default=1.5,
                    help="world units; median distance over temporal overlap to call two "
                         "cross-camera tracklets the same person")
    ap.add_argument("--min-overlap", type=int, default=5,
                    help="min co-present frames before a spatial link is trusted")
    ap.add_argument("--temporal-gap", type=int, default=300,
                    help="max frame gap between one tracklet ending and another starting "
                         "for a temporal (hand-off / id-switch) link")
    ap.add_argument("--endpoint-thr", type=float, default=2.5,
                    help="world units; max distance between the two tracklet endpoints "
                         "for a temporal link (used for near-zero gaps)")
    ap.add_argument("--pred-thr", type=float, default=1.5,
                    help="world units; max distance between the entering tracklet's start "
                         "and the exiting tracklet's velocity-extrapolated position. This "
                         "motion-continuity test distinguishes 'same person walked through' "
                         "from 'different person arrived at the same spot'.")
    ap.add_argument("--vel-frames", type=int, default=8,
                    help="#endpoint frames used to estimate exit/entry velocity")
    ap.add_argument("--pred-cam-offset", type=int, default=0)
    args = ap.parse_args()

    cal = WarehouseCalibration(args.calib)

    # build tracklets: (cam, ltid) -> {frame: (x,y)}
    traj: dict = defaultdict(dict)
    raw_rows = []  # (cam, frame, ltid)
    for f in sorted(args.export_dir.glob("cam_*_predictions.csv")):
        cam = int(f.stem.split("_")[1]) + args.pred_cam_offset
        if not cal.has(cam):
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            w = cal.foot_to_world(cam, r.left + r.width / 2.0, r.top + r.height)
            raw_rows.append((cam, int(r.frame_no_cam), int(r.local_track_id)))
            if w is not None:
                traj[(cam, int(r.local_track_id))][int(r.frame_no_cam)] = w

    keys = sorted(traj)
    idx = {k: i for i, k in enumerate(keys)}
    meta = []  # per tracklet: (cam, fmin, fmax, frames_set, mean_pos, start_pos, end_pos)
    for k in keys:
        fr = sorted(traj[k]); pts = np.array([traj[k][f] for f in fr])
        vf = min(args.vel_frames, len(fr))
        # exit velocity (per frame) over the last vf frames; entry velocity over the first vf
        exit_v = (pts[-1] - pts[-vf]) / max(1, fr[-1] - fr[-vf]) if vf > 1 else np.zeros(2)
        entry_v = (pts[vf - 1] - pts[0]) / max(1, fr[vf - 1] - fr[0]) if vf > 1 else np.zeros(2)
        meta.append({"cam": k[0], "fmin": fr[0], "fmax": fr[-1], "frames": set(fr),
                     "pos": traj[k], "start": pts[0], "end": pts[-1],
                     "exit_v": exit_v, "entry_v": entry_v})
    uf = UnionFind(len(keys))

    n_spatial = n_temporal = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = meta[i], meta[j]
            # SPATIAL: cross-camera, co-present, coincident
            if a["cam"] != b["cam"]:
                common = a["frames"] & b["frames"]
                if len(common) >= args.min_overlap:
                    ds = [np.hypot(*(np.array(a["pos"][f]) - np.array(b["pos"][f])))
                          for f in common]
                    if np.median(ds) < args.spatial_thr:
                        uf.union(i, j); n_spatial += 1
                        continue
            # TEMPORAL: one ends, the other begins within gap; require EITHER endpoints
            # almost coincident (near-instant hand-off) OR the exiting tracklet's
            # velocity-extrapolated position lands on the entering tracklet's start
            # (the person actually walked there — rejects a different person arriving).
            if a["fmax"] <= b["fmin"]:
                gap = b["fmin"] - a["fmax"]; pred = a["end"] + a["exit_v"] * gap; s1 = b["start"]
            elif b["fmax"] <= a["fmin"]:
                gap = a["fmin"] - b["fmax"]; pred = b["end"] + b["exit_v"] * gap; s1 = a["start"]
            else:
                continue  # they overlap in time -> not a temporal hand-off
            if gap < 0 or gap > args.temporal_gap:
                continue
            endpoint_d = np.hypot(*((a["end"] - b["start"]) if a["fmax"] <= b["fmin"]
                                    else (b["end"] - a["start"])))
            pred_d = np.hypot(*(pred - s1))
            if endpoint_d <= args.endpoint_thr or pred_d <= args.pred_thr:
                uf.union(i, j); n_temporal += 1

    # component -> compact gid
    comp = {}
    for i in range(len(keys)):
        comp.setdefault(uf.find(i), len(comp) + 1)
    tl_gid = {keys[i]: comp[uf.find(i)] for i in range(len(keys))}

    rows = [(c, f, t, tl_gid.get((c, t), -1)) for (c, f, t) in raw_rows]
    out = pd.DataFrame([r for r in rows if r[3] > 0],
                       columns=["cam_id", "frame_no", "local_track_id", "global_id"])
    out.insert(0, "group", args.group)
    out.to_csv(args.out_csv, index=False)
    print(f"[tracklet] {len(keys)} tracklets -> {len(comp)} global ids "
          f"({n_spatial} spatial + {n_temporal} temporal links); wrote {len(out)} rows")


if __name__ == "__main__":
    main()
