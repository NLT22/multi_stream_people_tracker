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


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _parse_groups(spec: str | None) -> list[tuple[str, set[int] | None]]:
    """Parse 'office:8-11,retail:16-19' into named camera groups."""
    if not spec:
        return [("all", None)]
    groups: list[tuple[str, set[int] | None]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            hint = ""
            if raw.isdigit():
                # Classic footgun: a caller shell script used a variable named
                # GROUPS (a bash special/readonly array = the user's group IDs),
                # so an empty value leaked the gid (e.g. "1000") into --groups.
                hint = (" — this looks like a leaked shell group id: do NOT use a "
                        "bash variable named GROUPS to build --groups (it is the "
                        "special array of the user's group IDs); rename it, e.g. GRP")
            raise ValueError(
                f"Invalid group spec {raw!r}; expected name:start-end{hint}")
        name, span = raw.split(":", 1)
        name = name.strip()
        cams: set[int] = set()
        for part in span.split("+"):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                cams.update(range(int(lo), int(hi) + 1))
            else:
                cams.add(int(part))
        if not name or not cams:
            raise ValueError(f"Invalid group spec {raw!r}; empty name or cameras")
        groups.append((name, cams))
    return groups or [("all", None)]


def _parse_group_ints(spec: str | None) -> dict[str, int]:
    """Parse 'retail:4,default:1' into per-group integer overrides."""
    out: dict[str, int] = {}
    if not spec:
        return out
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Invalid group override {raw!r}; expected name:value")
        name, value = raw.split(":", 1)
        out[name.strip()] = int(value)
    return out


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
        current_keys = {
            (int(c), int(f), int(t))
            for c, f, t in zip(cam, frame, ltid)
        }
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
        assignments = []
        for (c, f, t), cl in wmap.items():
            gid = cl2gid[cl - 1] + 1
            self.track_gid[(c, t)] = gid
            active.add(gid)
            if (int(c), int(f), int(t)) in current_keys:
                assignments.append((int(c), int(f), int(t), int(gid)))
        return {"n_dets": int(len(cw)), "k": k, "n_clusters": len(clusters),
                "active_gids": len(active), "total_gids": len(self._g_cent),
                "assignments": assignments}


def _load_chunk(path: Path):
    z = np.load(path)
    return (z["cam_id"].astype(np.int64), z["frame_no"].astype(np.int64),
            z["local_track_id"].astype(np.int64), z["embeddings"].astype(np.float32))


def static_track_dropset(export_dir: Path, motion_px: float,
                         min_frames: int) -> set[tuple[int, int]]:
    """Local tracks (cam_id, local_track_id) that barely move and live long.

    These are static false positives — mannequins, posters, shelf clutter — which
    inflate identity count and ID-switches (e.g. retail: precision 0.62 -> 0.88,
    IDF1 +7.7pp once removed). Computed from the per-camera prediction CSVs the
    exporter writes (chunks carry no bbox). Apply only in environments with static
    non-person clutter; seated real people (cafe/office) are static too, so do NOT
    enable this globally — scope it to the relevant camera group.
    """
    import pandas as pd
    drop: set[tuple[int, int]] = set()
    for p in sorted(export_dir.glob("cam_*_predictions.csv")):
        try:
            cam = int(p.stem.split("_")[1])
            df = pd.read_csv(p)
        except (ValueError, IndexError, pd.errors.EmptyDataError):
            continue
        cx = df["left"] + df["width"] / 2.0
        cy = df["top"] + df["height"] / 2.0
        stats = pd.DataFrame({"ltid": df["local_track_id"], "cx": cx, "cy": cy})
        for ltid, grp in stats.groupby("ltid"):
            if len(grp) < min_frames:
                continue
            if float(np.hypot(grp["cx"].std(), grp["cy"].std())) < motion_px:
                drop.add((cam, int(ltid)))
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--window-chunks", type=int, default=1,
                    help="how many recent chunks form one clustering window")
    ap.add_argument("--groups", default=None,
                    help="optional named camera groups, e.g. "
                         "'cafe:0-3,lobby:4-7,office:8-11'. Each group gets "
                         "an independent MTMC state so unrelated environments "
                         "are not merged together.")
    ap.add_argument("--group-window-chunks", default=None,
                    help="optional per-group window override, e.g. "
                         "'retail:4,default:1'")
    ap.add_argument("--assign-thr", type=float, default=0.50,
                    help="max (1 - cosine) to link a window cluster to an existing "
                         "global id. 0.50 is the swept optimum on the full 24-scene "
                         "MMP val set (mean IDF1 0.774@0.40 -> 0.780@0.50, "
                         "non-retail 0.853 -> 0.862); peaks at 0.50 then declines by "
                         "0.55. See scripts/eval/sweep_live_buffered.py.")
    ap.add_argument("--num-people", type=int, default=None,
                    help="fixed k; default = per-window concurrency floor")
    ap.add_argument("--anchor-window", type=int, default=15,
                    help="sliding-window vote length inside assign_per_frame")
    ap.add_argument("--fp-filter", action="store_true",
                    help="drop static long-lived local tracks (mannequins/posters/"
                         "shelf clutter) before clustering. Scope to clutter-prone "
                         "camera groups only (retail) — seated people are static too. "
                         "Retail: IDF1 0.615->0.693, precision 0.62->0.88.")
    ap.add_argument("--fp-motion", type=float, default=8.0,
                    help="center-position std (px) below which a track is 'static'")
    ap.add_argument("--fp-minframes", type=int, default=100,
                    help="only drop static tracks living at least this many frames")
    ap.add_argument("--log-csv", default="output/logs/live_buffered.csv")
    ap.add_argument("--gids-csv", default=None,
                    help="optional: write current (cam,local_track_id,global_id) map each step")
    ap.add_argument("--assign-csv", default=None,
                    help="optional: append per-detection assignments for the "
                         "new chunk as group,cam_id,frame_no,local_track_id,global_id")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after this many seconds (0 = until --max-idle)")
    ap.add_argument("--max-idle", type=float, default=120,
                    help="stop after this many seconds with no new chunk")
    ap.add_argument("--once", action="store_true",
                    help="process all existing chunks once then exit (verification)")
    args = ap.parse_args()

    groups = _parse_groups(args.groups)
    group_windows = _parse_group_ints(args.group_window_chunks)
    default_window = group_windows.get("default", args.window_chunks)
    mtmcs = {
        name: LiveBufferedMTMC(
            group_windows.get(name, default_window),
            args.assign_thr,
            args.num_people,
            args.anchor_window,
        )
        for name, _ in groups
    }
    log_path = Path(args.log_csv); log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write("ts,elapsed_s,chunk,group,n_dets,k,n_clusters,active_gids,total_gids,cluster_ms\n")
    log.flush()
    assign_file = None
    if args.assign_csv:
        assign_path = Path(args.assign_csv)
        assign_path.parent.mkdir(parents=True, exist_ok=True)
        assign_file = open(assign_path, "w")
        assign_file.write("group,cam_id,frame_no,local_track_id,global_id\n")
        assign_file.flush()
    print(f"[live-buffered] watching {args.export_dir} -> {log_path}")
    print("[live-buffered] groups: " + ", ".join(
        f"{name}(cams={'all' if cams is None else min(cams)}"
        f"{'' if cams is None or len(cams) == 1 else '-' + str(max(cams))},"
        f"chunks={mtmcs[name].window_chunks})"
        for name, cams in groups))

    fp_drop: set = set()
    if args.fp_filter:
        fp_drop = static_track_dropset(args.export_dir, args.fp_motion,
                                       args.fp_minframes)
        print(f"[live-buffered] fp-filter: dropping {len(fp_drop)} static track(s) "
              f"(motion<{args.fp_motion}px, >={args.fp_minframes} frames)")

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
            if fp_drop:
                keep = np.array([(int(c), int(t)) not in fp_drop
                                 for c, t in zip(cam, ltid)], dtype=bool)
                cam, frame, ltid, emb = cam[keep], frame[keep], ltid[keep], emb[keep]
                if len(cam) == 0:
                    seen.add(path.name); continue
            seen.add(path.name); last_new = time.time()
            idx = len(seen)
            for group_name, group_cams in groups:
                if group_cams is None:
                    mask = np.ones(len(cam), dtype=bool)
                else:
                    mask = np.isin(cam, list(group_cams))
                if not np.any(mask):
                    continue
                t1 = time.time()
                st = mtmcs[group_name].process_chunk(
                    cam[mask], frame[mask], ltid[mask], emb[mask])
                dt = (time.time() - t1) * 1000
                log.write(f"{time.strftime('%FT%T')},{time.time()-t0:.0f},{idx},"
                          f"{group_name},{st['n_dets']},{st['k']},"
                          f"{st['n_clusters']},{st['active_gids']},"
                          f"{st['total_gids']},{dt:.0f}\n")
                log.flush()
                if assign_file is not None:
                    for c, f, t, gid in st["assignments"]:
                        assign_file.write(f"{group_name},{c},{f},{t},{gid}\n")
                    assign_file.flush()
                print(f"[live-buffered] chunk {idx} group={group_name}: "
                      f"dets={st['n_dets']} k={st['k']} clusters={st['n_clusters']} "
                      f"active_gids={st['active_gids']} total_gids={st['total_gids']} "
                      f"({dt:.0f} ms)")
            if args.gids_csv:
                with open(args.gids_csv, "w") as g:
                    g.write("group,cam_id,local_track_id,global_id\n")
                    for group_name, mtmc in mtmcs.items():
                        for (c, t), gid in sorted(mtmc.track_gid.items()):
                            g.write(f"{group_name},{c},{t},{gid}\n")
        if args.once and not new and chunks:
            break
        if args.duration and time.time() - t0 >= args.duration:
            print("[live-buffered] duration reached — stop"); break
        if not args.once and time.time() - last_new >= args.max_idle:
            print(f"[live-buffered] no new chunk for {args.max_idle}s — stop"); break
        time.sleep(args.poll_interval)
    log.close()
    if assign_file is not None:
        assign_file.close()
    total_gids = sum(len(mtmc._g_cent) for mtmc in mtmcs.values())
    print(f"[live-buffered] done; {len(seen)} chunks, {total_gids} global ids total")


if __name__ == "__main__":
    main()
