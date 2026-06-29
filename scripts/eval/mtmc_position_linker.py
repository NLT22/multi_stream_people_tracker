#!/usr/bin/env python3
"""MTMC world-coordinate position linker (camera-topology hand-off, geometry-first).

Appearance-only linking plateaus on the warehouse because deployed crops are small/
occluded and the ReID embeddings are not separable enough (see docs). But the AICity
warehouses ship metric `calibration.json`, and people are 8-16 world units apart while
back-projection error is ~0.15 and per-frame motion ~0.05 — position is ~50x more
discriminative than appearance here.

The linker works in world coordinates, NOT tracker/ReID IDs:

  1. back-project every prediction's foot point (bottom-centre of bbox) to the ground
     plane → world (x, y) via WarehouseCalibration,
  2. per frame, greedily cluster detections within `--cluster-radius` (two cameras
     seeing one person land on the same world point → one instance → cross-camera link),
  3. track instances across frames with a constant-velocity motion model: a track
     predicts its next position, matches the nearest instance within `--gate`
     (Hungarian) and continues its id,
  4. when a track has been unseen for a gap, RE-LINKING it to a new instance also
     requires appearance agreement (`--relink-sim` on the per-track mean embedding) —
     this stops an id from jumping to a *different* person who later walks through the
     same spot, the dominant error mode of pure-position re-linking,
  5. write an assign-csv (group,cam_id,frame_no,local_track_id,global_id) — drop-in for
     score_mtmc_idf1.py's remap, same format as live_buffered --assign-csv.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def greedy_cluster(pts: np.ndarray, cams: list[int], radius: float) -> list[list[int]]:
    """Merge detections of one person seen by multiple cameras into one instance.

    KEY constraint: a cluster holds at most ONE detection per camera. Two detections in
    the SAME camera are always different people (a detector never reports one person
    twice), so they must never merge — even when they cross within `radius`. Without
    this, two people passing within ~1 world unit (which happens constantly in the
    warehouse) collapse into one global id. Points are matched nearest-first."""
    clusters: list[list[int]] = []
    cents: list[np.ndarray] = []
    cl_cams: list[set] = []
    # nearest-first improves stability when several dets are in range
    order = list(range(len(pts)))
    for i in order:
        p = pts[i]; cam = cams[i]
        best, bd = -1, radius
        for ci, c in enumerate(cents):
            if cam in cl_cams[ci]:        # same camera already in this cluster -> skip
                continue
            d = np.hypot(p[0] - c[0], p[1] - c[1])
            if d < bd:
                best, bd = ci, d
        if best < 0:
            clusters.append([i]); cents.append(p.copy()); cl_cams.append({cam})
        else:
            clusters[best].append(i); cl_cams[best].add(cam)
            cents[best] = np.mean(pts[clusters[best]], axis=0)
    return clusters


def load_embeddings(export_dir: Path) -> dict:
    """(cam, frame, ltid) -> L2-normalised embedding, from det_emb_chunk_*.npz."""
    emb: dict = {}
    for p in sorted(export_dir.glob("det_emb_chunk_*.npz")):
        z = np.load(p)
        for c, f, t, e in zip(z["cam_id"], z["frame_no"], z["local_track_id"],
                              z["embeddings"].astype(np.float32)):
            emb[(int(c), int(f), int(t))] = _l2(e)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--group", default="w")
    ap.add_argument("--cluster-radius", type=float, default=2.0,
                    help="world units; merge detections this close into one instance")
    ap.add_argument("--gate", type=float, default=4.0,
                    help="world units; max instance-to-predicted-track distance to continue an id")
    ap.add_argument("--max-age", type=int, default=450,
                    help="frames a track survives without an observation (occlusion/hand-off)")
    ap.add_argument("--relink-gap", type=int, default=15,
                    help="a match to a track stale for > this many frames is a RE-LINK and "
                         "must also pass the appearance gate")
    ap.add_argument("--relink-sim", type=float, default=0.5,
                    help="min cosine sim (track mean emb vs instance emb) to allow a re-link; "
                         "0 disables appearance gating")
    ap.add_argument("--relink-gate", type=float, default=2.5,
                    help="tighter world-distance gate for re-links across a gap")
    ap.add_argument("--vel-decay", type=float, default=0.7,
                    help="EMA factor for the velocity estimate (0=ignore motion)")
    ap.add_argument("--pred-cam-offset", type=int, default=0)
    args = ap.parse_args()

    cal = WarehouseCalibration(args.calib)
    emb = load_embeddings(args.export_dir) if args.relink_sim > 0 else {}

    # load all predictions -> per-frame detection list
    dets: dict[int, list] = defaultdict(list)  # frame -> [(cam, ltid, x, y)]
    n_proj = n_fail = 0
    for f in sorted(args.export_dir.glob("cam_*_predictions.csv")):
        cam = int(f.stem.split("_")[1]) + args.pred_cam_offset
        if not cal.has(cam):
            print(f"[link] no calib for cam {cam}; skipping {f.name}")
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            u = r.left + r.width / 2.0
            v = r.top + r.height            # foot = bottom-centre
            w = cal.foot_to_world(cam, u, v)
            if w is None:
                n_fail += 1; continue
            dets[int(r.frame_no_cam)].append((cam, int(r.local_track_id), w[0], w[1]))
            n_proj += 1
    print(f"[link] projected {n_proj} dets ({n_fail} failed) over {len(dets)} frames")

    # world-space tracking with constant-velocity prediction + appearance-gated re-link
    tracks: list[dict] = []   # {pos, vel, gid, last_frame, emb_sum, emb_n}
    next_gid = 1
    rows = []
    n_relink_rejected = 0
    for frame in sorted(dets):
        items = dets[frame]
        pts = np.array([[x, y] for _, _, x, y in items], float)
        clusters = greedy_cluster(pts, [it[0] for it in items], args.cluster_radius)
        inst_cent = np.array([pts[c].mean(0) for c in clusters])
        # instance mean embedding (for re-link gating)
        inst_emb = []
        for c in clusters:
            es = [emb[(items[di][0], frame, items[di][1])] for di in c
                  if (items[di][0], frame, items[di][1]) in emb]
            inst_emb.append(_l2(np.mean(es, 0)) if es else None)

        active = [t for t in tracks if frame - t["last_frame"] <= args.max_age]
        assign = {}
        if active and len(inst_cent):
            # predicted position = last pos + velocity * gap
            pred = np.array([t["pos"] + t["vel"] * (frame - t["last_frame"]) for t in active])
            cost = np.linalg.norm(inst_cent[:, None, :] - pred[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                tr = active[c]
                gap = frame - tr["last_frame"]
                is_relink = gap > args.relink_gap
                gate = args.relink_gate if is_relink else args.gate
                if cost[r, c] > gate:
                    continue
                if is_relink and args.relink_sim > 0 and inst_emb[r] is not None and tr["emb_n"] > 0:
                    tmean = _l2(tr["emb_sum"] / tr["emb_n"])
                    if float(tmean @ inst_emb[r]) < args.relink_sim:
                        n_relink_rejected += 1
                        continue
                # accept: update motion + appearance
                new_pos = inst_cent[r]
                if gap > 0:
                    v = (new_pos - tr["pos"]) / gap
                    tr["vel"] = args.vel_decay * tr["vel"] + (1 - args.vel_decay) * v
                tr["pos"] = new_pos; tr["last_frame"] = frame
                if inst_emb[r] is not None:
                    tr["emb_sum"] = tr["emb_sum"] + inst_emb[r]; tr["emb_n"] += 1
                assign[r] = tr["gid"]
        for ci, c in enumerate(clusters):
            if ci not in assign:
                e0 = inst_emb[ci]
                tracks.append({"pos": inst_cent[ci], "vel": np.zeros(2), "gid": next_gid,
                               "last_frame": frame,
                               "emb_sum": e0.copy() if e0 is not None else np.zeros(0),
                               "emb_n": 1 if e0 is not None else 0})
                if e0 is None:
                    tracks[-1]["emb_sum"] = np.zeros(emb[next(iter(emb))].shape) if emb else np.zeros(1)
                assign[ci] = next_gid; next_gid += 1
            gid = assign[ci]
            for di in c:
                cam, ltid, _, _ = items[di]
                rows.append((cam, frame, ltid, gid))

    out = pd.DataFrame(rows, columns=["cam_id", "frame_no", "local_track_id", "global_id"])
    out.insert(0, "group", args.group)
    out.to_csv(args.out_csv, index=False)
    print(f"[link] {next_gid - 1} global ids; {n_relink_rejected} re-links rejected by appearance; "
          f"wrote {len(out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
