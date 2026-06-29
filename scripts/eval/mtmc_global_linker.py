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


def build_tracklets(export_dir: Path, cal: WarehouseCalibration, cam_offset: int, vel_frames: int,
                    cam_map: dict | None = None):
    # cam_map: export cam_id -> real calibration camera number (source-list order).
    # Cameras are non-contiguous in W020/W021, so the export index != calibration id.
    traj: dict = defaultdict(dict)
    raw_rows = []
    for f in sorted(export_dir.glob("cam_*_predictions.csv")):
        cam = int(f.stem.split("_")[1]) + cam_offset
        calcam = cam_map.get(cam, cam) if cam_map else cam
        if not cal.has(calcam):
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            w = cal.foot_to_world(calcam, r.left + r.width / 2.0, r.top + r.height)
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


def link(keys, meta, p):
    """Build affinity + must-not-link from tracklet meta and run constrained
    agglomerative clustering. p exposes: min_overlap, spatial_thr, conflict_thr,
    temporal_gap, pred_thr, reacq_gap, reacq_thr, temporal_weight, min_merge.
    Returns (tl_gid: {(cam,ltid): gid}, stats)."""
    n = len(keys)
    cadj = [dict() for _ in range(n)]      # node -> {node: aff}  (cluster adjacency; nodes are reps)
    cforbid = [set() for _ in range(n)]    # node -> set of must-not-link nodes
    n_spatial = n_temporal = n_conflict = 0

    def add_aff(i, j, w):
        cadj[i][j] = cadj[i].get(j, 0.0) + w
        cadj[j][i] = cadj[j].get(i, 0.0) + w

    for i, j in combinations(range(n), 2):
        a, b = meta[i], meta[j]
        common = a["frames"] & b["frames"]
        if a["cam"] == b["cam"]:
            if common:
                cforbid[i].add(j); cforbid[j].add(i)
        else:
            if len(common) >= p.min_overlap:
                ds = [np.hypot(*(np.array(a["pos"][f]) - np.array(b["pos"][f]))) for f in common]
                md = float(np.median(ds))
                if md < p.spatial_thr:
                    add_aff(i, j, len(common) * (1.0 - md / p.spatial_thr)); n_spatial += 1
                elif md > p.conflict_thr:
                    cforbid[i].add(j); cforbid[j].add(i); n_conflict += 1
        if a["fmax"] <= b["fmin"]:
            gap = b["fmin"] - a["fmax"]; e0 = a["end"]; pred = a["end"] + a["exit_v"] * gap; s1 = b["start"]
        elif b["fmax"] <= a["fmin"]:
            gap = a["fmin"] - b["fmax"]; e0 = b["end"]; pred = b["end"] + b["exit_v"] * gap; s1 = a["start"]
        else:
            gap = None
        if gap is not None and 0 <= gap <= p.temporal_gap:
            if np.hypot(*(pred - s1)) <= p.pred_thr:
                add_aff(i, j, p.temporal_weight); n_temporal += 1
            elif (a["cam"] == b["cam"] and gap <= p.reacq_gap and np.hypot(*(e0 - s1)) <= p.reacq_thr):
                add_aff(i, j, p.temporal_weight); n_temporal += 1
    n_cannot = sum(len(s) for s in cforbid) // 2

    # constrained agglomerative clustering on the sparse cluster graph (rep = surviving node)
    members = {i: {i} for i in range(n)}
    active = set(range(n))
    while True:
        best, bw = None, p.min_merge
        for ru in active:
            fu = cforbid[ru]
            for rv, w in cadj[ru].items():
                if ru < rv and w >= bw and rv in active and rv not in fu:
                    best, bw = (ru, rv), w
        if best is None:
            break
        ru, rv = best
        members[ru] |= members[rv]; active.discard(rv)
        for rw, w in cadj[rv].items():
            if rw == ru:
                continue
            cadj[ru][rw] = cadj[ru].get(rw, 0.0) + w
            cadj[rw][ru] = cadj[rw].get(ru, 0.0) + w
            cadj[rw].pop(rv, None)
        cadj[ru].pop(rv, None); cadj[rv] = {}
        cforbid[ru] |= cforbid[rv]
        for x in list(cforbid[rv]):
            cforbid[x].discard(rv); cforbid[x].add(ru)
        cforbid[ru].discard(ru)

    tl_gid = {}
    for gid, cu in enumerate(sorted(active), start=1):
        for m in members[cu]:
            tl_gid[keys[m]] = gid
    return tl_gid, {"n_ids": len(active), "n_spatial": n_spatial, "n_temporal": n_temporal,
                    "n_conflict": n_conflict, "n_cannot": n_cannot}


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
    ap.add_argument("--sources", default=None,
                    help="source list used for the export; maps export cam_N -> real calibration "
                         "camera number (REQUIRED for W020/W021 where camera numbers are non-contiguous)")
    args = ap.parse_args()

    cam_map = None
    if args.sources:
        src = [l.strip() for l in open(args.sources) if l.strip() and not l.strip().startswith("#")]
        cam_map = {i: int(p.split("Camera_")[-1].split(".")[0]) for i, p in enumerate(src)}

    cal = WarehouseCalibration(args.calib)
    keys, meta, raw_rows = build_tracklets(args.export_dir, cal, args.pred_cam_offset, args.vel_frames, cam_map)
    n = len(keys)
    tl_gid, st = link(keys, meta, args)
    print(f"[global] {n} tracklets -> {st['n_ids']} ids "
          f"({st['n_spatial']} spatial+ / {st['n_temporal']} temporal+ / {st['n_conflict']} cross-cam "
          f"must-not-link + {st['n_cannot']} total must-not-link)")

    rows = [(c, f, t, tl_gid.get((c, t), -1)) for (c, f, t) in raw_rows]
    out = pd.DataFrame([r for r in rows if r[3] > 0],
                       columns=["cam_id", "frame_no", "local_track_id", "global_id"])
    out.insert(0, "group", args.group)
    out.to_csv(args.out_csv, index=False)
    print(f"[global] wrote {len(out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
