#!/usr/bin/env python3
"""MTMC post-hoc appearance gid-merge (cross-camera / temporal hand-off linker).

The live_buffered consumer re-clusters each window with a fixed k and stitches
windows 1-to-1 (gated Hungarian). On DISJOINT cameras (MTMC warehouse) a person
who leaves the busiest window or reappears in another camera after a gap spawns a
fresh global id — temporal/cross-camera fragmentation. This pass consolidates that:

  1. aggregate every detection's embedding per buffered global_id (from assign-csv
     + the det_emb_chunk_*.npz the live run wrote),
  2. build one L2-normalised mean embedding per gid (a gid gallery),
  3. agglomeratively merge gids whose mean embeddings are within `--merge-thr`
     cosine distance (average linkage),
  4. rewrite the assign-csv with merged gids.

Pure appearance — no geometry needed (MTMC's 3D calibration adapter is not built
yet). Sweep `--merge-thr`: too low merges distinct people, too high does nothing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--assign-csv", required=True, type=Path,
                    help="live_buffered assign-csv (group,cam_id,frame_no,local_track_id,global_id)")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--merge-thr", type=float, default=0.30,
                    help="cosine-distance ceiling for merging two gid galleries (avg linkage)")
    ap.add_argument("--min-dets", type=int, default=10,
                    help="gids with fewer detections are merged into the nearest gid regardless")
    args = ap.parse_args()

    a = pd.read_csv(args.assign_csv)
    # (cam, frame, ltid) -> gid
    key2gid = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
               for r in a.itertuples()}

    # aggregate embeddings per gid from the chunks
    gid_emb_sum: dict[int, np.ndarray] = {}
    gid_n: dict[int, int] = defaultdict(int)
    for p in sorted(args.export_dir.glob("det_emb_chunk_*.npz")):
        z = np.load(p)
        cam, frm, ltid = z["cam_id"], z["frame_no"], z["local_track_id"]
        emb = z["embeddings"].astype(np.float32)
        for c, f, t, e in zip(cam, frm, ltid, emb):
            gid = key2gid.get((int(c), int(f), int(t)))
            if gid is None:
                continue
            if gid not in gid_emb_sum:
                gid_emb_sum[gid] = np.zeros_like(e)
            gid_emb_sum[gid] += e
            gid_n[gid] += 1

    gids = sorted(gid_emb_sum)
    if len(gids) <= 1:
        a.to_csv(args.out_csv, index=False)
        print(f"[merge] <=1 gid; nothing to merge ({len(gids)})")
        return
    cents = np.stack([_l2(gid_emb_sum[g] / max(1, gid_n[g])) for g in gids])

    # cosine-distance condensed matrix -> average-linkage agglomerative merge
    D = 1.0 - cents @ cents.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=args.merge_thr, criterion="distance")

    # small-gid absorption: a gid below --min-dets dets folds into the nearest other gid
    label_of = {g: int(labels[i]) for i, g in enumerate(gids)}
    for i, g in enumerate(gids):
        if gid_n[g] < args.min_dets:
            order = np.argsort(D[i])
            for j in order:
                if gids[j] != g and gid_n[gids[j]] >= args.min_dets:
                    label_of[g] = label_of[gids[j]]
                    break

    # remap: cluster label -> compact new gid
    uniq = {lab: i + 1 for i, lab in enumerate(sorted(set(label_of.values())))}
    gid_new = {g: uniq[label_of[g]] for g in gids}

    a["global_id"] = a["global_id"].map(lambda g: gid_new.get(int(g), int(g)))
    a.to_csv(args.out_csv, index=False)
    print(f"[merge] {len(gids)} gids -> {len(set(gid_new.values()))} "
          f"(merge-thr={args.merge_thr}, min-dets={args.min_dets})")


if __name__ == "__main__":
    main()
