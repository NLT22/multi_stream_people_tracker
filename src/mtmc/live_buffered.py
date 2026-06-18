"""Live buffered cross-camera MTMC consumer (production_todo §2/§6, option B).

Consumes the per-detection embedding chunks that PredictionExporter writes during
a live run (`--live-buffered-window N` → det_emb_chunk_<NNNN>.npz). Maintains a
rolling window of the most recent chunks, re-clusters each window from scratch
(build_anchors + assign_per_frame — the offline-quality clustering), and stitches
consecutive windows to PERSISTENT global identities via Hungarian matching on
cluster centroids (k↔k, 1-to-1, no transitive union). This is the near-offline
"buffered" path applied incrementally to an unbounded stream.

Runs as a long-lived process alongside the live pipeline:

    python -m src.mtmc.live_buffered --export-dir output/eval/live_run \
        --window-chunks 1 --assign-thr 0.40 --log-csv output/logs/live_buffered.csv

It logs identity health over time (active / total global IDs, window size,
clustering latency) so a long run can be checked for identity stability — global
IDs should PLATEAU, not grow without bound. `--once` processes all existing chunks
then exits (used for verification).
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.eval.offline_anchor_faithful import build_anchors, assign_per_frame
from src.mtmc.tracklet import _l2


def _concurrency_floor(cam, frame, ltid) -> int:
    """Lower bound on #people in the window = max over frames of the busiest
    single camera's distinct-track count (tracks within one camera ≈ distinct
    people; across cameras they overlap, so the per-camera max is a safe floor)."""
    per_cf: dict = defaultdict(set)
    for c, f, t in zip(cam, frame, ltid):
        per_cf[(int(c), int(f))].add(int(t))
    per_frame_max: dict = defaultdict(int)
    for (c, f), tracks in per_cf.items():
        per_frame_max[f] = max(per_frame_max[f], len(tracks))
    return max(per_frame_max.values()) if per_frame_max else 1


class LiveBufferedMTMC:
    def __init__(self, window_chunks: int, assign_thr: float,
                 fixed_k: int | None, anchor_window: int):
        self.window_chunks = max(1, window_chunks)
        self.assign_thr = assign_thr
        self.fixed_k = fixed_k
        self.anchor_window = anchor_window
        self._chunks: deque = deque(maxlen=self.window_chunks)   # recent chunk arrays
        self._g_cent: list = []     # persistent global identity centroids
        self._g_n: list = []        # observation count per global id
        self.track_gid: dict = {}   # (cam, ltid) -> current global id

    def process_chunk(self, cam, frame, ltid, emb) -> dict:
        """Add one chunk, re-cluster the rolling window, return stats."""
        self._chunks.append((cam, frame, ltid, emb))
        cw = np.concatenate([c[0] for c in self._chunks])
        fw = np.concatenate([c[1] for c in self._chunks])
        tw = np.concatenate([c[2] for c in self._chunks])
        ew = np.concatenate([c[3] for c in self._chunks])

        k = self.fixed_k or _concurrency_floor(cw, fw, tw)
        k = max(1, min(k, len(np.unique(tw))))
        banks = build_anchors(ew, fw, k)
        wmap = assign_per_frame(cw, fw, tw, ew, banks, window=self.anchor_window)

        # centroid per window-cluster
        emap = {(int(c), int(f), int(t)): e for c, f, t, e in zip(cw, fw, tw, ew)}
        cl_emb: dict = defaultdict(list)
        for key, cl in wmap.items():
            cl_emb[cl - 1].append(emap[key])
        clusters = sorted(cl_emb)
        if not clusters:
            return {"n_dets": int(len(cw)), "k": k, "n_clusters": 0,
                    "active_gids": 0, "total_gids": len(self._g_cent)}
        cents = np.stack([_l2(np.mean(cl_emb[c], 0)) for c in clusters])

        # stitch window clusters -> persistent global identities (Hungarian, gated)
        cl2gid = {}
        if self._g_cent:
            G = np.stack(self._g_cent)
            cost = 1.0 - cents @ G.T
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.assign_thr:
                    cl2gid[clusters[r]] = c
                    self._g_cent[c] = _l2(self._g_cent[c] * self._g_n[c] + cents[r])
                    self._g_n[c] += 1
        for ri, cl in enumerate(clusters):
            if cl not in cl2gid:
                cl2gid[cl] = len(self._g_cent)
                self._g_cent.append(cents[ri]); self._g_n.append(1)

        # assign each detection's track its window-cluster global id
        active = set()
        for (c, f, t), cl in wmap.items():
            gid = cl2gid[cl - 1] + 1
            self.track_gid[(c, t)] = gid
            active.add(gid)
        return {"n_dets": int(len(cw)), "k": k, "n_clusters": len(clusters),
                "active_gids": len(active), "total_gids": len(self._g_cent)}


def _load_chunk(path: Path):
    z = np.load(path)
    return (z["cam_id"].astype(np.int64), z["frame_no"].astype(np.int64),
            z["local_track_id"].astype(np.int64), z["embeddings"].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--window-chunks", type=int, default=1,
                    help="how many recent chunks form one clustering window")
    ap.add_argument("--assign-thr", type=float, default=0.40,
                    help="max (1 - cosine) to link a window cluster to an existing global id")
    ap.add_argument("--num-people", type=int, default=None,
                    help="fixed k; default = per-window concurrency floor")
    ap.add_argument("--anchor-window", type=int, default=15,
                    help="sliding-window vote length inside assign_per_frame")
    ap.add_argument("--log-csv", default="output/logs/live_buffered.csv")
    ap.add_argument("--gids-csv", default=None,
                    help="optional: write current (cam,local_track_id,global_id) map each step")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after this many seconds (0 = until --max-idle)")
    ap.add_argument("--max-idle", type=float, default=120,
                    help="stop after this many seconds with no new chunk")
    ap.add_argument("--once", action="store_true",
                    help="process all existing chunks once then exit (verification)")
    args = ap.parse_args()

    mtmc = LiveBufferedMTMC(args.window_chunks, args.assign_thr,
                            args.num_people, args.anchor_window)
    log_path = Path(args.log_csv); log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write("ts,elapsed_s,chunk,n_dets,k,n_clusters,active_gids,total_gids,cluster_ms\n")
    log.flush()
    print(f"[live-buffered] watching {args.export_dir} -> {log_path}")

    seen: set = set()
    t0 = time.time(); last_new = time.time()
    while True:
        chunks = sorted(args.export_dir.glob("det_emb_chunk_*.npz"))
        new = [c for c in chunks if c.name not in seen]
        for path in new:
            try:
                cam, frame, ltid, emb = _load_chunk(path)
            except Exception as e:               # half-written / racing; retry next poll
                print(f"[live-buffered] skip {path.name}: {e}"); continue
            t1 = time.time()
            st = mtmc.process_chunk(cam, frame, ltid, emb)
            dt = (time.time() - t1) * 1000
            seen.add(path.name); last_new = time.time()
            idx = len(seen)
            log.write(f"{time.strftime('%FT%T')},{time.time()-t0:.0f},{idx},"
                      f"{st['n_dets']},{st['k']},{st['n_clusters']},"
                      f"{st['active_gids']},{st['total_gids']},{dt:.0f}\n")
            log.flush()
            print(f"[live-buffered] chunk {idx}: dets={st['n_dets']} k={st['k']} "
                  f"clusters={st['n_clusters']} active_gids={st['active_gids']} "
                  f"total_gids={st['total_gids']} ({dt:.0f} ms)")
            if args.gids_csv:
                with open(args.gids_csv, "w") as g:
                    g.write("cam_id,local_track_id,global_id\n")
                    for (c, t), gid in sorted(mtmc.track_gid.items()):
                        g.write(f"{c},{t},{gid}\n")
        if args.once and not new and chunks:
            break
        if args.duration and time.time() - t0 >= args.duration:
            print("[live-buffered] duration reached — stop"); break
        if not args.once and time.time() - last_new >= args.max_idle:
            print(f"[live-buffered] no new chunk for {args.max_idle}s — stop"); break
        time.sleep(args.poll_interval)
    log.close()
    print(f"[live-buffered] done; {len(seen)} chunks, {len(mtmc._g_cent)} global ids total")


if __name__ == "__main__":
    main()
