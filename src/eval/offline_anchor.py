"""Anchor-guided offline cross-camera re-association.

Offline post-processing port of the CVPRW2023 AIC23 Track1 winner
(`reference/AIC23_Track1_UWIPL_ETRI`), adapted to this project's export format.

Two stages, both pure post-processing on an existing ``--pred-dir`` export
(no pipeline re-run needed):

  1. ANCHOR CLUSTERING  — agglomerative-cluster per-tracklet mean embeddings with
     full-clip hindsight into ``k`` identities; the cluster label becomes the
     candidate global ID. ``k`` is either the GT person count (``--oracle-k``),
     an explicit ``--num-people``, or estimated via a cosine distance threshold.
  2. STCRA              — spatio-temporal consistency reassignment: build a
     per-identity world trajectory from ``tracklet_bev.csv`` and reassign whole
     tracklets to the geometrically nearest identity, iterated with shrinking
     distance / rising confidence gates. This fixes look-alike ID collisions that
     appearance clustering alone cannot (different people, far apart in world).

Unlike the reference we operate at *tracklet* granularity (our per-camera NvDCF
tracks are strong, ~90% MOTA on industry), so no per-detection embeddings are
needed and the output is a clean ``(cam_id, local_track_id) -> global_id`` remap.

Usage:
    python -m src.eval.offline_anchor \
        --pred-dir output/eval/clean_63am_industry_safety_0 \
        --out-dir  output/eval/clean_63am_industry_safety_0_anchor \
        --short-root dataset/MMPTracking_10minute/train \
        --scene 63am_industry_safety_0 --oracle-k
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ----------------------------------------------------------------------------- IO
def _load_tracklets(pred_dir: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    trk = pd.read_csv(pred_dir / "tracklets.csv")
    data = np.load(pred_dir / "tracklet_embeddings.npz")
    ids = data["tracklet_ids"].astype(np.int64)
    embs = data["embeddings"].astype(np.float32)
    # L2-normalize so euclidean clustering ~ cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return trk, {int(t): embs[i] for i, t in enumerate(ids)}


def _oracle_k(short_root: str, scene: str) -> int:
    from src.dataset.mmp_tracking import MMPTrackingShortDataset
    ds = MMPTrackingShortDataset(str(short_root), scene)
    pids: set[int] = set()
    for cam in ds.get_cam_ids():
        try:
            gt = ds.load_gt(cam)
        except (FileNotFoundError, ValueError):
            continue
        pids.update(int(p) for p in gt["person_id"].unique())
    return len(pids)


# ---------------------------------------------------------------- stage 1: anchors
def estimate_k(X: np.ndarray, k_min: int = 2, k_max: int = 40) -> int:
    """Estimate #identities from the largest gap in the merge distances.

    Weighted by tracklet count: longer agglomerations that suddenly jump in
    distance mark the boundary between within-identity and between-identity
    merges. We cut at the largest relative jump in the upper dendrogram.
    """
    n = len(X)
    if n <= k_min:
        return max(1, n)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.0, compute_distances=True).fit(X)
    d = np.sort(model.distances_)  # ascending, length n-1
    k_hi = min(k_max, n - 1)
    # candidate k corresponds to cutting just below the (n-k)-th merge;
    # score each k by the gap d[n-k] - d[n-k-1] (how decisive the cut is)
    best_k, best_gap = k_min, -1.0
    for k in range(k_min, k_hi + 1):
        idx = n - k  # merges below this index remain as k clusters
        if idx < 1 or idx >= len(d):
            continue
        gap = d[idx] - d[idx - 1]
        if gap > best_gap:
            best_gap, best_k = gap, k
    return best_k


def concurrency_floor(bev: pd.DataFrame, keep_tids: set[int],
                      pct: float = 95.0) -> int:
    """Lower bound on #people: the (robust) peak number of *simultaneously
    detected* people in any single camera. A person is not double-counted within
    one camera, so #identities >= this. Counted per-frame from BEV detections
    (not tracklet intervals, which overlap across a tracklet's whole lifetime),
    and a high percentile rather than the max so brief ID-handoff overlaps and
    ghost detections don't inflate it. Prevents catastrophic under-clustering."""
    sub = bev[bev["tracklet_id"].isin(keep_tids)]
    floor = 0
    for _, cam in sub.groupby("cam_id"):
        per_frame = cam.groupby("frame_no_cam")["local_track_id"].nunique()
        if len(per_frame):
            floor = max(floor, int(np.percentile(per_frame.values, pct)))
    return floor


def cluster_anchors(
    trk: pd.DataFrame,
    emb_by_trk: dict[int, np.ndarray],
    k: int | None,
    min_dets: int,
    conc_floor: int = 0,
) -> tuple[dict[int, int], int]:
    """Return ({tracklet_id -> candidate_gid (>=1)}, k_used)."""
    rows = trk[trk["num_detections"] >= min_dets]
    ids = [int(t) for t in rows["tracklet_id"] if int(t) in emb_by_trk]
    if not ids:
        return {}, 0
    X = np.vstack([emb_by_trk[t] for t in ids])
    if k is None:
        gap_k = estimate_k(X)
        k = max(gap_k, conc_floor)
        print(f"[anchor] auto-k: gap={gap_k} concurrency_floor={conc_floor} "
              f"-> k={k}")
    k = max(1, min(k, len(ids)))
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    return {t: int(lab) + 1 for t, lab in zip(ids, labels)}, k


# ------------------------------------------------------------------ stage 2: STCRA
def _tracklet_world(bev: pd.DataFrame) -> dict[int, dict[int, np.ndarray]]:
    """{tracklet_id -> {frame -> [x, y]}} (per-frame mean world position)."""
    out: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    g = bev.groupby(["tracklet_id", "frame_no_cam"])[["world_x", "world_y"]].mean()
    for (tid, fid), row in g.iterrows():
        out[int(tid)][int(fid)] = np.array([row["world_x"], row["world_y"]])
    return out


def _gid_trajectory(
    tids_of_gid: set[int],
    trk_world: dict[int, dict[int, np.ndarray]],
    exclude: int | None = None,
) -> dict[int, np.ndarray]:
    """Per-frame mean world position over the tracklets assigned to a gid."""
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for tid in tids_of_gid:
        if tid == exclude:
            continue
        for fid, xy in trk_world[tid].items():
            acc[fid].append(xy)
    return {fid: np.mean(v, axis=0) for fid, v in acc.items()}


def _overlap_dist(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> tuple[float, int]:
    common = a.keys() & b.keys()
    if not common:
        return float("inf"), 0
    d = [np.linalg.norm(a[f] - b[f]) for f in common]
    return float(np.median(d)), len(common)


def stcra(
    tid_to_gid: dict[int, int],
    trk_world: dict[int, dict[int, np.ndarray]],
    split_thr: float,
    min_overlap: int,
) -> dict[int, int]:
    """Conservative spatio-temporal correction.

    Only act on a *geometrically impossible* situation: two tracklets carrying
    the same global ID co-occur in time but are far apart in world (so they must
    be different people). Keep the longer tracklet's ID and move the shorter one
    to the appearance-clustered identity whose world trajectory it is closest to
    over the overlap. This refines look-alike collisions without fighting the
    appearance clustering elsewhere.
    """
    tid_to_gid = dict(tid_to_gid)
    moves = 0
    gid_members: dict[int, set[int]] = defaultdict(set)
    for t, g in tid_to_gid.items():
        gid_members[g].add(t)

    for gid in list(gid_members):
        members = [t for t in gid_members[gid] if trk_world.get(t)]
        # longest first -> treat it as the "true" owner of this ID
        members.sort(key=lambda t: -len(trk_world[t]))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if tid_to_gid[b] != gid:
                    continue
                dist, ov = _overlap_dist(trk_world[a], trk_world[b])
                if ov < min_overlap or dist < split_thr:
                    continue
                # b conflicts with a -> reassign b to its nearest OTHER identity
                best_gid, best_dist = None, None
                for cand, cmembers in gid_members.items():
                    if cand == gid:
                        continue
                    traj = _gid_trajectory(cmembers, trk_world)
                    d = [np.linalg.norm(xy - traj[f])
                         for f, xy in trk_world[b].items() if f in traj]
                    if len(d) < min_overlap:
                        continue
                    md = float(np.median(d))
                    if best_dist is None or md < best_dist:
                        best_gid, best_dist = cand, md
                if best_gid is not None and best_dist < split_thr:
                    tid_to_gid[b] = best_gid
                    gid_members[gid].discard(b)
                    gid_members[best_gid].add(b)
                    moves += 1
    print(f"  [stcra] split-conflicts (dist>={split_thr:.0f}): {moves} moves")
    return tid_to_gid


# ------------------------------------------------------------------------- output
def _write(pred_dir: Path, out_dir: Path,
           key_to_gid: dict[tuple[int, int], int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        with open(path, newline="") as src, \
             open(out_dir / path.name, "w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (int(row["cam_id"]), int(row["local_track_id"]))
                if key in key_to_gid:
                    row["global_id"] = key_to_gid[key]
                writer.writerow(row)
    for name in ("tracklets.csv", "tracklet_embeddings.npz", "tracklet_bev.csv"):
        if (pred_dir / name).exists():
            shutil.copy2(pred_dir / name, out_dir / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--oracle-k", action="store_true",
                    help="Set k = GT person count (validation upper bound).")
    ap.add_argument("--num-people", type=int, default=None,
                    help="Explicit k; overrides --oracle-k.")
    ap.add_argument("--min-dets", type=int, default=20,
                    help="Ignore tracklets shorter than this for clustering.")
    ap.add_argument("--split-thr", type=float, default=1500.0,
                    help="STCRA: world dist (mm) above which same-ID co-occurring "
                         "tracklets are deemed different people and split.")
    ap.add_argument("--min-overlap", type=int, default=8,
                    help="Min overlapping frames to trust a STCRA comparison.")
    ap.add_argument("--frag-gate", type=float, default=1.25,
                    help="Re-cluster only if online_ids/k >= this (online "
                         "over-fragmented). Below it, keep online IDs untouched "
                         "to protect already-clean scenes.")
    ap.add_argument("--force", action="store_true",
                    help="Always re-cluster, ignoring the fragmentation gate.")
    ap.add_argument("--relabel", action="store_true",
                    help="Alias for --force (always relabel to cluster IDs).")
    ap.add_argument("--stcra", action="store_true",
                    help="Enable spatio-temporal split correction. Off by "
                         "default: appearance clustering already resolves most "
                         "identities, and STCRA can over-split clean clusters.")
    args = ap.parse_args()

    pred_dir, out_dir = Path(args.pred_dir), Path(args.out_dir)
    trk, emb_by_trk = _load_tracklets(pred_dir)
    bev = pd.read_csv(pred_dir / "tracklet_bev.csv")

    k = args.num_people
    if k is None and args.oracle_k:
        if not args.scene:
            ap.error("--oracle-k requires --scene")
        k = _oracle_k(args.short_root, args.scene)
        print(f"[anchor] oracle k = {k} GT identities")

    keep = set(trk[trk["num_detections"] >= args.min_dets]["tracklet_id"])
    floor = concurrency_floor(bev, keep) if k is None else 0
    tid_to_cluster, k_used = cluster_anchors(
        trk, emb_by_trk, k, args.min_dets, conc_floor=floor)
    print(f"[anchor] clustered {len(tid_to_cluster)} tracklets into "
          f"{len(set(tid_to_cluster.values()))} identities (k={k_used})")

    # tracklets dropped by clustering (too short / no emb) keep their online gid
    full = {int(r.tracklet_id): int(r.global_id) for r in trk.itertuples()}

    # Fragmentation gate: re-cluster only when the online tracker emitted clearly
    # more identities than there are people (it over-fragmented / swapped). When
    # online already produced ~k clean IDs, leave it alone — wholesale relabel
    # only adds appearance-clustering noise to an already-correct scene.
    online_ids = len({full[t] for t in tid_to_cluster if full.get(t, -1) >= 0})
    frag = online_ids / max(1, k_used)
    apply = args.relabel or args.force or frag >= args.frag_gate
    print(f"[anchor] online_ids={online_ids} k={k_used} frag={frag:.2f} "
          f"gate={args.frag_gate} -> {'RELABEL' if apply else 'KEEP online'}")
    if apply:
        base = {**{t: g for t, g in full.items() if g >= 0}, **tid_to_cluster}
    else:
        base = {t: g for t, g in full.items() if g >= 0}

    if args.stcra:
        trk_world = _tracklet_world(bev)
        base = stcra(base, trk_world, args.split_thr, args.min_overlap)
        print(f"[stcra] -> {len(set(base.values()))} identities")

    # build (cam, local_track_id) -> gid via tracklet metadata
    meta = {int(r.tracklet_id): (int(r.cam_id), int(r.local_track_id))
            for r in trk.itertuples()}
    key_to_gid = {meta[t]: g for t, g in base.items() if t in meta}

    _write(pred_dir, out_dir, key_to_gid)
    print(f"[done] wrote remapped predictions -> {out_dir}")


if __name__ == "__main__":
    main()
