"""Faithful reproduction of the AIC23 anchor-guided pipeline (per-DETECTION).

This mirrors `reference/AIC23_Track1_UWIPL_ETRI` as closely as our data allows,
to test whether STCRA hurts because of the *algorithm* or because of our earlier
*tracklet-level* adaptation (`offline_anchor.py`). Differences from that module:

  * anchors built from **per-detection** embeddings (detection_embeddings.npz),
    not tracklet means;
  * **per-frame, per-camera Hungarian** assigns each detection to a distinct
    anchor (mutual exclusion within a camera-frame) — reference `aic_hungarian_*`;
  * **sliding-window (n=15) majority vote** per local track smooths the per-frame
    assignment — reference exactly;
  * **STCRA** then runs at detection granularity (reference `run_stcra.py`):
    per-(gid,frame) camera-weighted world centroid, reassign outlier detections
    to the nearest identity trajectory, 3 passes with shrinking distance and the
    reference confidence gate `conf = 1 - d_best/d_cur`.

Needs an export produced *after* the per-detection-embedding exporter change
(i.e. a re-run pipeline). Usage:

    python -m src.eval.offline_anchor_faithful \
        --pred-dir output/eval/faithful_63am_industry_safety_0 \
        --out-dir  output/eval/faithful_63am_industry_safety_0_out \
        --short-root dataset/MMPTracking_10minute/train \
        --scene 63am_industry_safety_0 --oracle-k [--stcra]
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

WINDOW = 15  # reference sliding-window majority-vote length


# ----------------------------------------------------------------------- load
def _load_det_embeddings(pred_dir: Path):
    z = np.load(pred_dir / "detection_embeddings.npz")
    cam = z["cam_id"].astype(np.int64)
    frame = z["frame_no"].astype(np.int64)
    ltid = z["local_track_id"].astype(np.int64)
    emb = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    emb = emb / n
    return cam, frame, ltid, emb


def _oracle_k(short_root: str, scene: str) -> int:
    from src.dataset.mmp_tracking import MMPTrackingShortDataset
    ds = MMPTrackingShortDataset(str(short_root), scene)
    pids = set()
    for c in ds.get_cam_ids():
        try:
            pids.update(int(p) for p in ds.load_gt(c)["person_id"].unique())
        except (FileNotFoundError, ValueError):
            pass
    return len(pids)


# -------------------------------------------------- stage 1: anchors + assign
def build_anchors(emb: np.ndarray, frame: np.ndarray, k: int,
                  n_anchor_frames: int = 40, bank_cap: int = 64) -> list[np.ndarray]:
    """Cluster embeddings sampled from anchor frames into k anchors; each anchor
    is a FEATURE BANK (paper §3.2): a set of exemplar embeddings (capped), not a
    single mean. Returns a list of (n_j, D) arrays."""
    uframes = np.unique(frame)
    pick = uframes[np.linspace(0, len(uframes) - 1, n_anchor_frames).astype(int)]
    mask = np.isin(frame, pick)
    X = emb[mask]
    if len(X) < k:
        X = emb
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    banks = []
    rng = np.random.default_rng(0)
    for g in range(k):
        bank = X[labels == g]
        if len(bank) > bank_cap:
            bank = bank[rng.choice(len(bank), bank_cap, replace=False)]
        banks.append(bank.astype(np.float32))
    return banks


def _bank_cost(emb: np.ndarray, banks: list[np.ndarray]) -> np.ndarray:
    """Paper eq.(1): cost(d, a_j) = 1 - mean_l cos(d, a_{j,l}) over the bank."""
    cols = []
    for bank in banks:                 # (n_j, D)
        cols.append(1.0 - (emb @ bank.T).mean(axis=1))   # (N,)
    return np.stack(cols, axis=1)      # (N, k)


def assign_per_frame(cam, frame, ltid, emb, banks):
    """Per-camera, per-frame Hungarian to distinct anchors + window-vote smooth.
    Returns {(cam,frame,ltid): gid}."""
    k = len(banks)
    cost = _bank_cost(emb, banks)     # (N,k) avg-cosine-to-bank
    out: dict[tuple[int, int, int], int] = {}

    for c in np.unique(cam):
        cm = cam == c
        cf, cl, cc = frame[cm], ltid[cm], cost[cm]
        idx = np.arange(len(cf))
        # per-frame Hungarian
        raw: dict[tuple[int, int], int] = {}   # (frame,ltid)->anchor
        for f in np.unique(cf):
            fm = cf == f
            rows = idx[fm]
            sub = cc[fm]                       # (m,k)
            r, col = linear_sum_assignment(sub)
            assigned = {}
            for rr, cco in zip(r, col):
                assigned[rr] = cco
            for j, gid_row in enumerate(rows):
                a = assigned.get(j, int(np.argmin(sub[j])))  # extra dets: argmin
                raw[(int(f), int(cl[fm][j]))] = int(a)
        # sliding-window majority vote per local track
        per_track: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (f, t), a in raw.items():
            per_track[t].append((f, a))
        for t, seq in per_track.items():
            seq.sort()
            frames = [f for f, _ in seq]
            anchs = [a for _, a in seq]
            for i, f in enumerate(frames):
                lo = max(0, i - WINDOW // 2)
                hi = min(len(anchs), i + WINDOW // 2 + 1)
                vote = Counter(anchs[lo:hi]).most_common(1)[0][0]
                out[(int(c), int(f), int(t))] = vote + 1   # gid >= 1
    return out


# --------------------------------------------------------------- stage 2: STCRA
def stcra(det_gid, world, passes, conf_thr, min_overlap):
    """Per-detection spatio-temporal reassignment (reference run_stcra spirit).
    det_gid/world keyed by (cam,frame,ltid)."""
    det_gid = dict(det_gid)
    keys = [key for key in det_gid if key in world]
    for dist_thr in passes:
        # per-(gid,frame) world centroid
        cen: dict[tuple[int, int], list] = defaultdict(list)
        for key in keys:
            c, f, t = key
            cen[(det_gid[key], f)].append(world[key])
        centroid = {gf: np.mean(v, 0) for gf, v in cen.items()}
        gids = {det_gid[key] for key in keys}
        moves = 0
        for key in keys:
            c, f, t = key
            g = det_gid[key]
            p = world[key]
            best_g, best_d, cur_d = g, None, None
            for cg in gids:
                # exclude self-frame contribution for current gid handled loosely
                if (cg, f) not in centroid:
                    continue
                d = float(np.linalg.norm(p - centroid[(cg, f)]))
                if cg == g:
                    cur_d = d
                if best_d is None or d < best_d:
                    best_g, best_d = cg, d
            if best_g == g or best_d is None or best_d >= dist_thr:
                continue
            conf = 1.0 - best_d / cur_d if cur_d and cur_d > 0 else 1.0
            if conf >= conf_thr:
                det_gid[key] = best_g
                moves += 1
        print(f"  [stcra] dist<{dist_thr:.0f} conf>={conf_thr}: {moves} moves")
    return det_gid


# ------------------------------------------------------------------- write/main
def _write(pred_dir, out_dir, det_gid):
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        with open(path, newline="") as s, open(out_dir / path.name, "w", newline="") as d:
            r = csv.DictReader(s)
            w = csv.DictWriter(d, fieldnames=r.fieldnames)
            w.writeheader()
            for row in r:
                key = (int(row["cam_id"]), int(row["frame_no_cam"]),
                       int(row["local_track_id"]))
                if key in det_gid:
                    row["global_id"] = det_gid[key]
                w.writerow(row)
    for name in ("tracklets.csv", "tracklet_embeddings.npz", "tracklet_bev.csv"):
        if (pred_dir / name).exists():
            shutil.copy2(pred_dir / name, out_dir / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--oracle-k", action="store_true")
    ap.add_argument("--num-people", type=int, default=None)
    ap.add_argument("--stcra", action="store_true")
    ap.add_argument("--passes", type=float, nargs="+", default=[1500, 1000, 750])
    ap.add_argument("--conf-thr", type=float, default=0.65)
    ap.add_argument("--min-overlap", type=int, default=8)
    args = ap.parse_args()

    pred_dir, out_dir = Path(args.pred_dir), Path(args.out_dir)
    cam, frame, ltid, emb = _load_det_embeddings(pred_dir)
    print(f"[faithful] {len(emb)} per-detection embeddings, "
          f"{len(np.unique(cam))} cams")

    k = args.num_people or (_oracle_k(args.short_root, args.scene)
                            if args.oracle_k else None)
    if k is None:
        raise SystemExit("provide --oracle-k or --num-people")
    print(f"[faithful] k = {k}")

    banks = build_anchors(emb, frame, k)
    print(f"[faithful] anchor bank sizes: {[len(b) for b in banks]}")
    det_gid = assign_per_frame(cam, frame, ltid, emb, banks)
    print(f"[faithful] assigned {len(det_gid)} detections -> "
          f"{len(set(det_gid.values()))} identities")

    if args.stcra:
        bev = pd.read_csv(pred_dir / "tracklet_bev.csv")
        world = {(int(r.cam_id), int(r.frame_no_cam), int(r.local_track_id)):
                 np.array([r.world_x, r.world_y]) for r in bev.itertuples()}
        det_gid = stcra(det_gid, world, args.passes, args.conf_thr,
                        args.min_overlap)
        print(f"[faithful] after STCRA -> {len(set(det_gid.values()))} ids")

    _write(pred_dir, out_dir, det_gid)
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
