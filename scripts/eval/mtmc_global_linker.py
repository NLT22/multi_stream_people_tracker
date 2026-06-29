#!/usr/bin/env python3
"""MTMC global tracklet linker — constrained correlation clustering (geometry-first).

The union-find linker (mtmc_tracklet_linker.py) merges tracklets greedily on first
match and can't undo a bad merge, so it both over-merges (swaps) and under-merges
(fragments). This linker reasons globally over the whole tracklet graph with two
ingredients union-find lacks:

  * MUST-NOT-LINK constraints (hard): two tracklets are provably DIFFERENT people if
      - they share a camera and overlap in time (one camera never sees a person twice), or
      - they are cross-camera, co-present, yet their world positions are far apart.
    No cluster is ever allowed to contain such a pair — this kills the swap errors.
  * AGGREGATE affinity: positive evidence summed over all member pairs (cross-camera
    spatial coincidence + velocity-consistent temporal hand-offs), so a merge is driven
    by the total support between two identities, not a single lucky edge.

Constrained agglomerative clustering: repeatedly merge the cluster pair with the highest
positive affinity that violates no must-not-link constraint, until none remain above
--min-merge. Outputs an assign-csv (group,cam_id,frame_no,local_track_id,global_id).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration


def build_tracklets(export_dir: Path, cal: WarehouseCalibration, cam_offset: int, vel_frames: int):
    traj: dict = defaultdict(dict)
    raw_rows = []
    for f in sorted(export_dir.glob("cam_*_predictions.csv")):
        cam = int(f.stem.split("_")[1]) + cam_offset
        if not cal.has(cam):
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            w = cal.foot_to_world(cam, r.left + r.width / 2.0, r.top + r.height)
            raw_rows.append((cam, int(r.frame_no_cam), int(r.local_track_id)))
            if w is not None:
                traj[(cam, int(r.local_track_id))][int(r.frame_no_cam)] = w
    keys = sorted(traj)
    meta = []
    for k in keys:
        fr = sorted(traj[k]); pts = np.array([traj[k][f] for f in fr])
        vf = min(vel_frames, len(fr))
        exit_v = (pts[-1] - pts[-vf]) / max(1, fr[-1] - fr[-vf]) if vf > 1 else np.zeros(2)
        meta.append({"cam": k[0], "fmin": fr[0], "fmax": fr[-1], "frames": set(fr),
                     "pos": traj[k], "start": pts[0], "end": pts[-1], "exit_v": exit_v})
    return keys, meta, raw_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--group", default="w")
    ap.add_argument("--spatial-thr", type=float, default=1.5,
                    help="cross-cam co-present median dist BELOW which we add positive affinity")
    ap.add_argument("--conflict-thr", type=float, default=3.0,
                    help="cross-cam co-present median dist ABOVE which it is a must-not-link")
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--temporal-gap", type=int, default=150)
    ap.add_argument("--pred-thr", type=float, default=1.0,
                    help="velocity-extrapolation distance for a positive temporal edge")
    ap.add_argument("--reacq-gap", type=int, default=60,
                    help="same-camera re-acquisition: a track ending and another starting within "
                         "this many frames is likely one person after an NvDCF id-switch")
    ap.add_argument("--reacq-thr", type=float, default=2.0,
                    help="endpoint distance for a same-camera re-acquisition edge (velocity may "
                         "flip if the person turned, so this does NOT require motion consistency)")
    ap.add_argument("--temporal-weight", type=float, default=30.0,
                    help="affinity awarded to a velocity-consistent temporal hand-off edge")
    ap.add_argument("--min-merge", type=float, default=1.0,
                    help="stop merging when the best allowed cluster-pair affinity falls below this")
    ap.add_argument("--vel-frames", type=int, default=8)
    ap.add_argument("--pred-cam-offset", type=int, default=0)
    args = ap.parse_args()

    cal = WarehouseCalibration(args.calib)
    keys, meta, raw_rows = build_tracklets(args.export_dir, cal, args.pred_cam_offset, args.vel_frames)
    n = len(keys)

    # pairwise positive affinity + must-not-link constraints
    aff = defaultdict(float)               # (i,j) -> positive weight
    cannot = set()                         # frozenset({i,j}) must-not-link
    n_spatial = n_temporal = n_conflict = 0
    for i, j in combinations(range(n), 2):
        a, b = meta[i], meta[j]
        common = a["frames"] & b["frames"]
        if a["cam"] == b["cam"]:
            if common:                     # same camera, simultaneous -> different people
                cannot.add(frozenset((i, j)))
        else:
            if len(common) >= args.min_overlap:
                ds = [np.hypot(*(np.array(a["pos"][f]) - np.array(b["pos"][f]))) for f in common]
                md = float(np.median(ds))
                if md < args.spatial_thr:
                    aff[(i, j)] += len(common) * (1.0 - md / args.spatial_thr)
                    n_spatial += 1
                elif md > args.conflict_thr:
                    cannot.add(frozenset((i, j))); n_conflict += 1
        # temporal hand-off (either camera): velocity-consistent continuation
        if a["fmax"] <= b["fmin"]:
            gap = b["fmin"] - a["fmax"]; e0 = a["end"]; pred = a["end"] + a["exit_v"] * gap; s1 = b["start"]
        elif b["fmax"] <= a["fmin"]:
            gap = a["fmin"] - b["fmax"]; e0 = b["end"]; pred = b["end"] + b["exit_v"] * gap; s1 = a["start"]
        else:
            gap = None
        if gap is not None and 0 <= gap <= args.temporal_gap:
            if np.hypot(*(pred - s1)) <= args.pred_thr:
                aff[(i, j)] += args.temporal_weight; n_temporal += 1
            elif (a["cam"] == b["cam"] and gap <= args.reacq_gap
                  and np.hypot(*(e0 - s1)) <= args.reacq_thr):
                # same-camera re-acquisition after a brief id-switch: endpoint proximity only
                aff[(i, j)] += args.temporal_weight; n_temporal += 1

    # constrained agglomerative clustering
    members = {i: {i} for i in range(n)}
    active = set(range(n))

    def cluster_blocked(cu, cv):
        for x in members[cu]:
            for y in members[cv]:
                if frozenset((x, y)) in cannot:
                    return True
        return False

    def pair_affinity(cu, cv):
        s = 0.0
        for x in members[cu]:
            for y in members[cv]:
                s += aff.get((x, y) if x < y else (y, x), 0.0)
        return s

    while True:
        best, bw = None, args.min_merge
        al = sorted(active)
        for ai in range(len(al)):
            for aj in range(ai + 1, len(al)):
                cu, cv = al[ai], al[aj]
                w = pair_affinity(cu, cv)
                if w >= bw and not cluster_blocked(cu, cv):
                    best, bw = (cu, cv), w
        if best is None:
            break
        cu, cv = best
        members[cu] |= members[cv]; del members[cv]; active.discard(cv)

    tl_gid = {}
    for gid, cu in enumerate(sorted(active), start=1):
        for m in members[cu]:
            tl_gid[keys[m]] = gid

    rows = [(c, f, t, tl_gid.get((c, t), -1)) for (c, f, t) in raw_rows]
    out = pd.DataFrame([r for r in rows if r[3] > 0],
                       columns=["cam_id", "frame_no", "local_track_id", "global_id"])
    out.insert(0, "group", args.group)
    out.to_csv(args.out_csv, index=False)
    print(f"[global] {n} tracklets -> {len(active)} ids "
          f"({n_spatial} spatial+ / {n_temporal} temporal+ / {n_conflict} cross-cam must-not-link "
          f"+ {sum(1 for c in cannot)} total must-not-link); wrote {len(out)} rows")


if __name__ == "__main__":
    main()
