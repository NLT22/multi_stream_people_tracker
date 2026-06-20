"""Offline simulation harness for the incremental micro-batch MTMC.

Replays a scene's per-camera tracklets as time-windowed micro-batches through
`IncrementalMTMC`, then writes eval-compatible `cam_*_predictions.csv` so the result
can be scored with `src.eval.metrics_mmp` and compared to the offline anchor-guided
upper bound. This is the offline stand-in for the production perception->bus->service
flow: instead of Kafka, we bin completed tracklets by their end-time into batches.

Run:
  python -m src.mtmc.run_incremental \
      --pred-dir output/eval/heldout_64pm_office_0 \
      --out-dir  output/eval/heldout_64pm_office_0_microbatch \
      --batch-frames 150 --assign-thr 0.35 --merge-thr 0.30 --ttl 900
  python -m src.eval.metrics_mmp --short-root dataset/MMPTracking_10minute/val \
      --scene 64pm_office_0 --pred-dir output/eval/heldout_64pm_office_0_microbatch
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.offline_anchor_faithful import _write
from .incremental_mtmc import IncrementalMTMC
from .tracklet import Tracklet, _l2


def _build_tracklets(pred_dir: Path, k: int = 8) -> list[Tracklet]:
    """Aggregate per-detection embeddings into a per-tracklet BANK of k crops
    (evenly subsampled over the tracklet's frames) — keeps per-crop discrimination
    (vs a collapsed mean) while staying cheap. k<=0 -> use all crops.
    """
    z = np.load(pred_dir / "detection_embeddings.npz")
    cam = z["cam_id"].astype(np.int64)
    frame = z["frame_no"].astype(np.int64)
    ltid = z["local_track_id"].astype(np.int64)
    emb = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1.0
    emb = emb / n

    world: dict[tuple[int, int], list] = defaultdict(list)
    bev_path = pred_dir / "tracklet_bev.csv"
    if bev_path.exists():
        bev = pd.read_csv(bev_path)
        for r in bev.itertuples():
            world[(int(r.cam_id), int(r.local_track_id))].append((r.world_x, r.world_y))

    agg: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"emb": [], "frame": [], "fmin": 1 << 30, "fmax": -1})
    for c, f, t, e in zip(cam, frame, ltid, emb):
        a = agg[(int(c), int(t))]
        a["emb"].append(e); a["frame"].append(int(f))
        a["fmin"] = min(a["fmin"], int(f)); a["fmax"] = max(a["fmax"], int(f))

    tracklets = []
    for (c, t), a in agg.items():
        E = np.stack(a["emb"])                      # crops in frame order
        if k > 0 and len(E) > k:
            sel = np.linspace(0, len(E) - 1, k).astype(int)   # evenly subsample k crops
            E = E[sel]
        fw = np.mean(world[(c, t)], axis=0) if world.get((c, t)) else None
        tracklets.append(Tracklet(sensor_id=c, tracklet_id=t, t_start=a["fmin"],
                                   t_end=a["fmax"], bank=E.astype(np.float32),
                                   foot_world=fw, n_obs=len(a["emb"])))
    return tracklets


def _run_buffered(args) -> None:
    """NEAR-OFFLINE: sliding window of W frames; each window runs the OFFLINE clustering
    (build_anchors + per-frame Hungarian) from scratch, stitched across windows by shared
    tracklets (union-find). Re-clustering the buffer lets it CORRECT recent mistakes with
    more context (unlike greedy). Latency = window; W -> whole clip == offline.
    """
    from src.eval.offline_anchor_faithful import build_anchors, assign_per_frame, _oracle_k
    from collections import Counter
    z = np.load(args.pred_dir / "detection_embeddings.npz")
    cam = z["cam_id"].astype(np.int64); frame = z["frame_no"].astype(np.int64)
    ltid = z["local_track_id"].astype(np.int64); emb = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1.0; emb = emb / n

    if args.num_people:
        k = args.num_people
    elif args.oracle_k:
        k = _oracle_k(args.short_root, args.scene)
    else:
        peak = [np.percentile(pd.read_csv(p).groupby("frame_no_cam")["local_track_id"].nunique(), 95)
                for p in args.pred_dir.glob("cam_*_predictions.csv")]
        k = int(np.ceil(max(peak))) if peak else 7
    from scipy.optimize import linear_sum_assignment
    W = args.window_frames; step = args.window_step or max(1, W // 2)
    fmax = int(frame.max())
    print(f"[buffered] k={k} window={W}f step={step}f -> {fmax//step+1} windows")

    g_cent: list = []          # global identity centroids (one per global gid)
    g_n: list = []             # obs count per global gid (for running mean)
    det_gid: dict = {}
    for ws in range(0, fmax + 1, step):
        m = (frame >= ws) & (frame < ws + W)
        if int(m.sum()) < k:
            continue
        cw, fw, tw, ew = cam[m], frame[m], ltid[m], emb[m]
        banks = build_anchors(ew, fw, k)
        wmap = assign_per_frame(cw, fw, tw, ew, banks, window=args.window)  # (c,f,t)->clu+1
        # window-cluster centroids from the assigned detections
        emap = {(int(c), int(f), int(t)): e for c, f, t, e in zip(cw, fw, tw, ew)}
        cl_emb: dict = defaultdict(list)
        for key, cl in wmap.items():
            cl_emb[cl - 1].append(emap[key])
        clusters = sorted(cl_emb)
        cents = np.stack([_l2(np.mean(cl_emb[c], 0)) for c in clusters])     # (kc, D)
        # match window clusters -> global identities (1-to-1 Hungarian, gated)
        cl2gid = {}
        if g_cent:
            G = np.stack(g_cent)
            cost = 1.0 - cents @ G.T
            rows, cols = linear_sum_assignment(cost)
            taken = set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= args.assign_thr:
                    cl2gid[clusters[r]] = c; taken.add(r)
                    g_cent[c] = _l2(g_cent[c] * g_n[c] + cents[r]); g_n[c] += 1
        for ri, cl in enumerate(clusters):          # unmatched clusters -> new global ids
            if cl not in cl2gid:
                cl2gid[cl] = len(g_cent); g_cent.append(cents[ri]); g_n.append(1)
        # assign every detection in this window its cluster's global gid (overwrites -> refines)
        for key, cl in wmap.items():
            det_gid[key] = cl2gid[cl - 1] + 1
    print(f"[buffered] {len(set(det_gid.values()))} global ids over {len(det_gid)} dets")
    _write(args.pred_dir, args.out_dir, det_gid)
    print(f"[buffered] wrote -> {args.out_dir}")


def _run_perdet(args) -> None:
    """Per-DETECTION micro-batch: keep per-det embeddings + per-cam-per-frame Hungarian
    (offline-style) against PERSISTENT banks, spawning new banks for unmatched dets.
    Heavier than tracklet-mean, but recovers the offline discrimination."""
    import time
    from sklearn.cluster import AgglomerativeClustering
    from src.eval.offline_anchor_faithful import _bank_cost, assign_per_frame

    z = np.load(args.pred_dir / "detection_embeddings.npz")
    cam = z["cam_id"].astype(np.int64); frame = z["frame_no"].astype(np.int64)
    ltid = z["local_track_id"].astype(np.int64); emb = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1.0; emb = emb / n

    wins: dict[int, list[int]] = defaultdict(list)
    for i, f in enumerate(frame):
        wins[int(f) // args.batch_frames].append(i)

    banks: list[np.ndarray] = []          # list of (n, D) exemplar banks
    bank_gid: list[int] = []              # bank index -> global id
    bank_seen: list[int] = []             # bank index -> last window seen
    next_gid = 1
    rng = np.random.default_rng(0)
    det_gid: dict[tuple[int, int, int], int] = {}
    alias: dict[int, int] = {}            # union-find: gid -> merged-into gid

    def _find(g: int) -> int:
        while g in alias:
            g = alias[g]
        return g

    # k prior (opt-in): hard-cap live identities. NOTE measured to HURT — forcing
    # down to k by greedy closest-centroid merge fuses different people (office_0:
    # cap7 -> 0.41 vs threshold-only 0.71). Default = no cap (use --num-people to force).
    k_cap = args.num_people if args.num_people else 10 ** 6
    if args.num_people:
        print(f"[per-det] k_cap = {k_cap} (WARNING: hard cap measured to hurt accuracy)")

    def _consolidate():
        """Merge banks whose centroids drift within consolidate_thr (retroactive gid
        alias), AND hard-merge the closest pair whenever over k_cap — the k prior
        stops streaming per-det from over-spawning, without a threshold knife-edge."""
        changed = True
        while changed and len(banks) > 1:
            changed = False
            cents = np.stack([_l2(b.mean(0)) for b in banks])
            D = 1.0 - cents @ cents.T
            np.fill_diagonal(D, 9.0)
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] < args.consolidate_thr or len(banks) > k_cap:
                lo, hi = (i, j) if bank_gid[i] <= bank_gid[j] else (j, i)
                alias[bank_gid[hi]] = bank_gid[lo]          # union gids
                merged = np.vstack([banks[lo], banks[hi]])
                if len(merged) > args.bank_cap:
                    merged = merged[rng.choice(len(merged), args.bank_cap, replace=False)]
                banks[lo] = merged; bank_seen[lo] = max(bank_seen[lo], bank_seen[hi])
                for arr in (banks, bank_gid, bank_seen):
                    del arr[hi]
                changed = True

    t0 = time.perf_counter()
    for w in sorted(wins):
        idx = np.array(wins[w])
        cw, fw, tw, ew = cam[idx], frame[idx], ltid[idx], emb[idx]
        # 1. spawn new banks for detections far from every existing bank
        minc = _bank_cost(ew, banks).min(1) if banks else np.full(len(ew), 9.0)
        un = minc > args.assign_thr
        if un.any():
            Xun = ew[un]
            labels = (np.zeros(1, int) if len(Xun) == 1 else
                      AgglomerativeClustering(n_clusters=None, distance_threshold=args.merge_thr,
                                              metric="cosine", linkage="average").fit_predict(Xun))
            for g in np.unique(labels):
                bk = Xun[labels == g]
                if len(bk) > args.bank_cap:
                    bk = bk[rng.choice(len(bk), args.bank_cap, replace=False)]
                banks.append(bk.astype(np.float32)); bank_gid.append(next_gid)
                bank_seen.append(w); next_gid += 1
        # 2. assign window dets to banks (per-cam per-frame Hungarian + window vote)
        wmap = assign_per_frame(cw, fw, tw, ew, banks, window=args.window)
        # 3. record gid + collect per-bank members for the bank update
        emap = {(int(c), int(f), int(t)): e for c, f, t, e in zip(cw, fw, tw, ew)}
        members: dict[int, list] = defaultdict(list)
        for key, bi1 in wmap.items():
            bi = bi1 - 1
            det_gid[key] = bank_gid[bi]
            members[bi].append(emap[key]); bank_seen[bi] = w
        # 4. grow + cap banks; TTL-age
        for bi, embs in members.items():
            merged = np.vstack([banks[bi], np.stack(embs).astype(np.float32)])
            if len(merged) > args.bank_cap:
                merged = merged[rng.choice(len(merged), args.bank_cap, replace=False)]
            banks[bi] = merged
        if args.ttl is not None:
            keep = [i for i in range(len(banks)) if (w - bank_seen[i]) * args.batch_frames <= args.ttl]
            banks = [banks[i] for i in keep]; bank_gid = [bank_gid[i] for i in keep]
            bank_seen = [bank_seen[i] for i in keep]
        # 5. consolidate banks that have drifted together (kills over-spawning)
        if not args.no_consolidate:
            _consolidate()
    dt = time.perf_counter() - t0
    # resolve retroactive gid merges (union-find)
    det_gid = {k: _find(g) for k, g in det_gid.items()}
    span = int(frame.max()); vid_s = span / 15.0; n_cam = len(set(cam.tolist()))
    print(f"[per-det] {len(emb)} dets, {len(wins)} batches -> "
          f"{len(set(det_gid.values()))} global ids")
    print(f"[fps] per-det MTMC: {dt*1000:.0f} ms | covers {vid_s:.0f}s of {n_cam}-cam "
          f"-> {vid_s/dt:.0f}x realtime")
    _write(args.pred_dir, args.out_dir, det_gid)
    print(f"[per-det] wrote predictions -> {args.out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--per-det", action="store_true",
                    help="per-detection assignment (offline-style) instead of tracklet-mean")
    ap.add_argument("--window", type=int, default=15, help="per-det sliding-window vote length")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="(per-det) disable bank consolidation — over-spawns identities")
    ap.add_argument("--consolidate-thr", type=float, default=0.15,
                    help="(per-det) merge banks whose centroids are within this cosine "
                         "distance; tighter than --merge-thr (only near-duplicate banks)")
    ap.add_argument("--batch-frames", type=int, default=150,
                    help="micro-batch window size (frames); 150 ~ 10s @15fps")
    ap.add_argument("--merge-thr", type=float, default=0.30)
    ap.add_argument("--assign-thr", type=float, default=0.35)
    ap.add_argument("--bank-cap", type=int, default=64)
    ap.add_argument("--ttl", type=float, default=None,
                    help="drop anchors unseen for this many frames (None = keep)")
    ap.add_argument("--max-anchors", type=int, default=None,
                    help="hard cap on live identities (e.g. concurrency prior)")
    ap.add_argument("--bank-k", type=int, default=8,
                    help="per-tracklet crop-bank size (evenly subsampled); 0 = all crops")
    ap.add_argument("--num-people", type=int, default=None,
                    help="(per-det) hard cap on live identities (k prior); "
                         "default = concurrency-floor estimate from the predictions")
    ap.add_argument("--buffered", action="store_true",
                    help="near-offline: sliding-window offline re-cluster + stitch")
    ap.add_argument("--window-frames", type=int, default=900,
                    help="(buffered) window size in frames (900 ~ 60s @15fps)")
    ap.add_argument("--window-step", type=int, default=0,
                    help="(buffered) window step; 0 = window/2 (50%% overlap)")
    ap.add_argument("--oracle-k", action="store_true", help="use GT person count for k")
    args = ap.parse_args()

    if args.buffered:
        _run_buffered(args)
        return
    if args.per_det:
        _run_perdet(args)
        return

    tracklets = _build_tracklets(args.pred_dir, k=args.bank_k)
    # bin by COMPLETION time (t_end): a tracklet is clustered once it finishes.
    batches: dict[int, list[Tracklet]] = defaultdict(list)
    for tl in tracklets:
        batches[int(tl.t_end) // args.batch_frames].append(tl)

    mtmc = IncrementalMTMC(merge_thr=args.merge_thr, assign_thr=args.assign_thr,
                           bank_cap=args.bank_cap, ttl=args.ttl,
                           max_anchors=args.max_anchors)
    tl_gid: dict[tuple[int, int], int] = {}
    import time
    t0 = time.perf_counter()
    for b in sorted(batches):
        t_now = (b + 1) * args.batch_frames
        out = mtmc.ingest(batches[b], t_now)
        tl_gid.update(out)
    dt = time.perf_counter() - t0
    n_cam = len({tl.sensor_id for tl in tracklets})
    span = max((tl.t_end for tl in tracklets), default=0)   # frames of footage
    # video seconds covered (src 15 fps) vs MTMC wall-time -> realtime factor
    vid_s = span / 15.0
    print(f"[microbatch] {len(tracklets)} tracklets, {len(batches)} batches "
          f"(window={args.batch_frames}f) -> {len(set(tl_gid.values()))} global ids "
          f"({mtmc.num_identities} live anchors)")
    print(f"[fps] MTMC stage: {dt*1000:.0f} ms total | {len(tracklets)/dt:.0f} tracklets/s "
          f"| covers {vid_s:.0f}s of {n_cam}-cam footage -> {vid_s/dt:.0f}x realtime "
          f"({n_cam*vid_s*15/dt:.0f} cam-frames/s equiv)")

    # expand tracklet gid -> per-detection gid for the eval writer
    z = np.load(args.pred_dir / "detection_embeddings.npz")
    cam = z["cam_id"].astype(np.int64)
    frame = z["frame_no"].astype(np.int64)
    ltid = z["local_track_id"].astype(np.int64)
    det_gid = {}
    for c, f, t in zip(cam, frame, ltid):
        g = tl_gid.get((int(c), int(t)))
        if g is not None:
            det_gid[(int(c), int(f), int(t))] = g
    _write(args.pred_dir, args.out_dir, det_gid)
    print(f"[microbatch] wrote predictions -> {args.out_dir}")


if __name__ == "__main__":
    main()
