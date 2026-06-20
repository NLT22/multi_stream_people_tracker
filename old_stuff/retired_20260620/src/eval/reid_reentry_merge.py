"""Step 1 — long-range Global-ID consolidation: fix Global-ID explosion over time.

On long videos the live gallery mints a NEW global id every time a person leaves
and returns (re-entry) or is split across cameras, so N real people accumulate
into many ids. The windowed nearline merge can't span minute-long gaps. This
post-pass consolidates fragments of the SAME person across the WHOLE sequence,
with a hard CANNOT-LINK constraint that is provably safe:

  merge global ids A and B  iff
    * their mean ReID embeddings are very similar (cosine >= --threshold), AND
    * they never share a (camera, frame) — one detection per camera means a
      shared (cam, frame) proves A and B are different people. The same person
      legitimately appears in *different* cameras at the same frame, so
      cross-camera fragments stay mergeable while same-camera co-occurrence is
      forbidden.

This recovers re-entry + cross-camera fragments without the over-merge a loose
whole-sequence cluster causes. It reads a pipeline export and writes a remapped
copy. NOTE: it is a no-op on always-populated scenes (no re-entry) and cannot
separate genuinely look-alike people — that is the appearance ceiling, not a bug.

Run:
    python -m src.eval.reid_reentry_merge --pred-dir output/eval/office10min \
        --out-dir output/eval/office10min_reentry --threshold 0.7
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil

import numpy as np


def _per_gid_embeddings(pred_dir: str):
    tl = {}
    with open(os.path.join(pred_dir, "tracklets.csv")) as f:
        for r in csv.DictReader(f):
            tl[int(r["tracklet_id"])] = (int(r["global_id"]), int(r["num_embeddings"]))
    z = np.load(os.path.join(pred_dir, "tracklet_embeddings.npz"))
    tids, embs = z["tracklet_ids"], z["embeddings"]
    acc, wsum = {}, {}
    for tid, e in zip(tids, embs):
        gid, w = tl.get(int(tid), (-1, 0))
        if gid < 0 or w <= 0:
            continue
        acc[gid] = acc.get(gid, 0.0) + e * w
        wsum[gid] = wsum.get(gid, 0.0) + w
    out = {}
    for gid, v in acc.items():
        m = v / wsum[gid]
        n = np.linalg.norm(m)
        if n > 1e-9:
            out[gid] = m / n
    return out


def _per_gid_frames(pred_dir: str):
    """Returns gid -> set of (cam_id, frame) occupied. Cannot-link uses
    (cam, frame): two ids that share a CAMERA at the same frame are different
    people (one detection per camera), but the same person legitimately appears
    in *different* cameras at the same frame — so cross-camera fragments stay
    mergeable while same-camera co-occurrence is forbidden."""
    cf: dict[int, set] = {}
    for csv_path in glob.glob(os.path.join(pred_dir, "cam_*_predictions.csv")):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                gid = int(float(r["global_id"]))
                if gid < 0:
                    continue
                cf.setdefault(gid, set()).add((int(r["cam_id"]), int(r["frame_no_cam"])))
    return cf


def merge(pred_dir: str, threshold: float):
    emb = _per_gid_embeddings(pred_dir)
    frames = _per_gid_frames(pred_dir)
    gids = sorted(set(emb) & set(frames))

    # union-find with per-component frame set + running mean embedding
    parent = {g: g for g in gids}
    comp_frames = {g: set(frames[g]) for g in gids}
    comp_emb = {g: emb[g].copy() for g in gids}
    comp_w = {g: float(len(frames[g])) for g in gids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # candidate pairs by descending similarity
    pairs = []
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            a, b = gids[i], gids[j]
            s = float(emb[a] @ emb[b])
            if s >= threshold:
                pairs.append((s, a, b))
    pairs.sort(reverse=True)

    merges = []
    for s, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if comp_frames[ra] & comp_frames[rb]:
            continue  # co-occur in some frame -> different people, cannot-link
        # re-check similarity on the merged-so-far component means (avoid drift)
        if float(comp_emb[ra] @ comp_emb[rb]) < threshold:
            continue
        # union (smaller into larger)
        if comp_w[ra] < comp_w[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        m = comp_emb[ra] * comp_w[ra] + comp_emb[rb] * comp_w[rb]
        comp_emb[ra] = m / (np.linalg.norm(m) + 1e-9)
        comp_w[ra] += comp_w[rb]
        comp_frames[ra] |= comp_frames[rb]
        merges.append((a, b, round(s, 3)))

    remap = {g: find(g) for g in gids}
    n_before, n_after = len(gids), len(set(remap.values()))
    return remap, n_before, n_after, merges


def _rewrite(pred_dir: str, out_dir: str, remap: dict):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(pred_dir):
        src, dst = os.path.join(pred_dir, name), os.path.join(out_dir, name)
        if name.endswith(".csv") and (name.startswith("cam_") or name in ("tracklet_bev.csv", "tracklets.csv")):
            with open(src) as f:
                rows = list(csv.reader(f))
            header = rows[0]
            gi = header.index("global_id")
            with open(dst, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for row in rows[1:]:
                    g = int(float(row[gi]))
                    if g in remap:
                        row[gi] = remap[g]
                    w.writerow(row)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Min cosine to merge two re-entry fragments (strict).")
    args = ap.parse_args()

    remap, nb, na, merges = merge(args.pred_dir, args.threshold)
    _rewrite(args.pred_dir, args.out_dir, remap)
    print(f"[reentry] global ids {nb} -> {na}  ({nb - na} re-entry merges, "
          f"threshold={args.threshold})")
    for a, b, s in merges:
        print(f"    merged G{b} -> G{a}  (cos={s})")
    print(f"[reentry] remapped export -> {args.out_dir}")


if __name__ == "__main__":
    main()
