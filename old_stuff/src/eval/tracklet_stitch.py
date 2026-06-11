"""Intra-camera tracklet stitching for the near-realtime MTMC merge.

The fast 20cam-capable tracker (NvDCF reidType:0) keeps good detection recall
(~92%) but fragments each person into many short local tracks (15-23 ID switches
per camera vs 1-2 for the expensive in-tracker ReID tracker). Cross-camera
embedding merge alone cannot repair this: appearance is ambiguous, so forcing
the right ID count over-merges different people.

This pass stitches fragmented local tracks *within one camera* using primarily
MOTION continuity (constant-velocity prediction across the temporal gap) gated by
a lenient appearance guard — i.e. it does offline what the in-tracker ReID
re-association does online, but without the per-frame GPU cost.

Output: a new pred-dir where every detection gets a stable per-camera stitched
global_id (by (cam_id, local_track_id)). Feed that to nearline_merge for the
cross-camera step:

    python -m src.eval.tracklet_stitch  --pred-dir IN --out-dir STITCHED [knobs]
    python -m src.eval.nearline_merge   --pred-dir STITCHED --out-dir OUT ...
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval import offline_merge


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intra-camera tracklet stitching")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-gap-frames", type=int, default=45,
                   help="Max temporal gap (frames) between two fragments to stitch")
    p.add_argument("--motion-gate", type=float, default=2.5,
                   help="Max predicted-vs-actual gap distance, in mean bbox heights")
    p.add_argument("--app-floor", type=float, default=0.25,
                   help="Min cosine similarity guard (lenient; motion leads)")
    p.add_argument("--size-ratio", type=float, default=2.5,
                   help="Max mean-height ratio between two stitched fragments")
    p.add_argument("--motion-weight", type=float, default=0.7,
                   help="Weight of motion vs appearance in the stitch score")
    p.add_argument("--vel-window", type=int, default=8,
                   help="Frames used to estimate end/start velocity")
    p.add_argument("--overlap-tolerance", type=int, default=3,
                   help="Allowed temporal overlap (frames) before two fragments conflict")
    return p.parse_args()


def _load_trajectories(pred_dir: Path) -> dict[tuple[int, int], list[tuple]]:
    """(cam_id, local_track_id) -> sorted list of (frame, cx, cy, w, h)."""
    traj: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cam = int(row["cam_id"])
                ltid = int(row["local_track_id"])
                frame = int(float(row["frame_no_cam"]))
                left = float(row["left"]); top = float(row["top"])
                w = float(row["width"]); h = float(row["height"])
                traj[(cam, ltid)].append((frame, left + w / 2.0, top + h / 2.0, w, h))
    for key in traj:
        traj[key].sort(key=lambda r: r[0])
    return traj


def _velocity(points: list[tuple], window: int, at_end: bool) -> tuple[float, float]:
    """Constant-velocity estimate (per frame) from the head or tail of a track."""
    if len(points) < 2:
        return 0.0, 0.0
    seg = points[-window:] if at_end else points[:window]
    f0, x0, y0, *_ = seg[0]
    f1, x1, y1, *_ = seg[-1]
    df = f1 - f0
    if df <= 0:
        return 0.0, 0.0
    return (x1 - x0) / df, (y1 - y0) / df


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smaller root for stable, low global_ids
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int], tol: int) -> bool:
    return a[0] <= b[1] - tol and b[0] <= a[1] - tol


def stitch(pred_dir: Path, args: argparse.Namespace) -> dict[tuple[int, int], int]:
    """Return (cam_id, local_track_id) -> stitched group id."""
    tracklets, emb_by_tracklet = offline_merge._load_tracklets(pred_dir)
    traj = _load_trajectories(pred_dir)

    # Per-tracklet feature record
    recs: list[dict] = []
    for t in tracklets:
        key = (t["cam_id"], t["local_track_id"])
        pts = traj.get(key, [])
        if not pts:
            continue
        emb = emb_by_tracklet.get(t["tracklet_id"])
        mean_h = float(np.mean([p[4] for p in pts])) or 1.0
        recs.append({
            "tid": t["tracklet_id"],
            "cam": t["cam_id"],
            "key": key,
            "start": t["start_frame"],
            "end": t["end_frame"],
            "start_pos": np.array(pts[0][1:3]),
            "end_pos": np.array(pts[-1][1:3]),
            "end_vel": np.array(_velocity(pts, args.vel_window, at_end=True)),
            "mean_h": mean_h,
            "emb": (emb / (np.linalg.norm(emb) + 1e-9)) if emb is not None else None,
        })

    by_cam: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_cam[r["cam"]].append(r)

    uf = _UnionFind()
    for r in recs:
        uf.find(r["tid"])

    candidates: list[tuple[float, int, int]] = []  # (score, tid_a, tid_b)
    for cam, group in by_cam.items():
        group.sort(key=lambda r: r["start"])
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                gap = b["start"] - a["end"]
                if gap < 0 or gap > args.max_gap_frames:
                    if b["start"] > a["end"] + args.max_gap_frames:
                        # group sorted by start; later b only start even later
                        # but b's relative to THIS a; can't break safely -> continue
                        continue
                    continue
                # size consistency
                hr = a["mean_h"] / b["mean_h"] if b["mean_h"] > 0 else 99.0
                if hr > args.size_ratio or hr < 1.0 / args.size_ratio:
                    continue
                # motion: predict a forward across the gap
                predicted = a["end_pos"] + a["end_vel"] * gap
                dist = float(np.linalg.norm(predicted - b["start_pos"]))
                scale = 0.5 * (a["mean_h"] + b["mean_h"])
                dist_norm = dist / max(1.0, scale)
                if dist_norm > args.motion_gate:
                    continue
                motion_score = float(np.exp(-dist_norm))
                # appearance guard
                if a["emb"] is not None and b["emb"] is not None:
                    app = float(np.dot(a["emb"], b["emb"]))
                else:
                    app = args.app_floor
                if app < args.app_floor:
                    continue
                score = args.motion_weight * motion_score + (1.0 - args.motion_weight) * app
                candidates.append((score, a["tid"], b["tid"]))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Members per group (for overlap conflict checks)
    rec_by_tid = {r["tid"]: r for r in recs}
    members: dict[int, list[int]] = {r["tid"]: [r["tid"]] for r in recs}

    def group_members(tid: int) -> list[int]:
        return members[uf.find(tid)]

    for score, a_tid, b_tid in candidates:
        ra, rb = uf.find(a_tid), uf.find(b_tid)
        if ra == rb:
            continue
        # no member of one group may temporally overlap a member of the other
        conflict = False
        for ma in members[ra]:
            ia = (rec_by_tid[ma]["start"], rec_by_tid[ma]["end"])
            for mb in members[rb]:
                ib = (rec_by_tid[mb]["start"], rec_by_tid[mb]["end"])
                if _intervals_overlap(ia, ib, args.overlap_tolerance):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        merged = members[ra] + members[rb]
        uf.union(a_tid, b_tid)
        members[uf.find(a_tid)] = merged

    return {rec_by_tid[r["tid"]]["key"]: uf.find(r["tid"]) for r in recs}


def _write(pred_dir: Path, out_dir: Path,
           key_to_gid: dict[tuple[int, int], int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # predictions: assign global_id by (cam, local_track_id) -> stitched gid
    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        with open(path, newline="") as src, open(out_dir / path.name, "w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (int(row["cam_id"]), int(row["local_track_id"]))
                if key in key_to_gid:
                    row["global_id"] = key_to_gid[key]
                writer.writerow(row)

    # tracklets.csv: rewrite global_id to the stitched gid (nearline aggregates by gid)
    tpath = pred_dir / "tracklets.csv"
    with open(tpath, newline="") as src, open(out_dir / "tracklets.csv", "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            key = (int(row["cam_id"]), int(row["local_track_id"]))
            if key in key_to_gid:
                row["global_id"] = key_to_gid[key]
            writer.writerow(row)

    emb = pred_dir / "tracklet_embeddings.npz"
    if emb.exists():
        shutil.copy2(emb, out_dir / "tracklet_embeddings.npz")


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    key_to_gid = stitch(pred_dir, args)

    n_tracks = len(key_to_gid)
    n_groups = len(set(key_to_gid.values()))
    print(f"[stitch] pred_dir={pred_dir}")
    print(f"[stitch] local tracks={n_tracks} -> stitched groups={n_groups} "
          f"(merged {n_tracks - n_groups})")
    _write(pred_dir, out_dir, key_to_gid)
    print(f"[stitch] wrote {out_dir}")


if __name__ == "__main__":
    main()
