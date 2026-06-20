"""Step 2 — build a global-ID trajectory store from a pipeline export.

Reads <pred-dir>/tracklet_bev.csv (per-detection world foot points keyed by
global_id) and produces the queryable "behaviour / RTLS" store every analytic
reads:

  <out-dir>/global_tracks.csv      one fused world point per (global_id, frame)
                                   global_id, frame, world_x, world_y, n_cams
  <out-dir>/global_id_summary.csv  one row per identity
                                   global_id, first_frame, last_frame,
                                   frames_present, dwell_s, coverage_pct,
                                   n_cameras, path_length_m, mean_speed_mps,
                                   span_x_m, span_y_m

A person seen by several cameras in the same frame yields several world points;
they are fused to ONE point per (global_id, frame) by the per-axis MEDIAN
(robust to a single mis-projected camera). global_id == -1 (unmatched) is dropped.

Run:
    python scripts/eval/global_tracks.py --pred-dir output/eval/lobby0 --fps 25
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def load_bev(pred_dir: str):
    path = os.path.join(pred_dir, "tracklet_bev.csv")
    # gid -> frame -> list[(x, y)] ; gid -> set(cams)
    pts: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    cams: dict[int, set] = defaultdict(set)
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = int(float(r["global_id"]))
            if gid < 0:
                continue
            fr = int(r["frame_no_cam"])
            pts[gid][fr].append((float(r["world_x"]), float(r["world_y"])))
            cams[gid].add(int(r["cam_id"]))
    return pts, cams


def build(pred_dir: str, out_dir: str, fps: float, iqr_k: float = 3.0,
          smooth_win: int = 7) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pts, cams = load_bev(pred_dir)

    def iqr_keep(v: np.ndarray, k: float) -> np.ndarray:
        """Tukey fence: keep points within [Q1-k*IQR, Q3+k*IQR] (drops gross
        BEV-projection outliers that teleport the world point far off-plane)."""
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        return (v >= q1 - k * iqr) & (v <= q3 + k * iqr)

    track_rows, summary_rows = [], []
    for gid in sorted(pts):
        frames = sorted(pts[gid])
        F, X, Y, N = [], [], [], []
        for fr in frames:
            arr = np.array(pts[gid][fr], dtype=np.float64)
            F.append(fr); X.append(np.median(arr[:, 0])); Y.append(np.median(arr[:, 1]))
            N.append(len(arr))
        F, X, Y, N = map(np.asarray, (F, X, Y, N))
        keep = iqr_keep(X, iqr_k) & iqr_keep(Y, iqr_k)
        dropped = int((~keep).sum())
        F, X, Y, N = F[keep], X[keep], Y[keep], N[keep]
        # rolling-median smoothing removes per-frame projection jitter (which
        # otherwise inflates path length / speed) and gives clean journey lines.
        if smooth_win > 1 and len(X) >= 3:
            X = pd.Series(X).rolling(smooth_win, center=True, min_periods=1).median().to_numpy()
            Y = pd.Series(Y).rolling(smooth_win, center=True, min_periods=1).median().to_numpy()
        for fr, x, y, n in zip(F, X, Y, N):
            track_rows.append({"global_id": gid, "frame": int(fr),
                               "world_x": round(float(x), 1), "world_y": round(float(y), 1),
                               "n_cams": int(n)})

        xy = np.column_stack([X, Y])
        seg = np.diff(xy, axis=0)
        path_mm = float(np.sqrt((seg ** 2).sum(axis=1)).sum()) if len(xy) > 1 else 0.0
        first, last = int(F[0]), int(F[-1])
        present = len(F)
        dwell_s = present / fps
        summary_rows.append({
            "global_id": gid,
            "first_frame": first,
            "last_frame": last,
            "frames_present": present,
            "dwell_s": round(dwell_s, 2),
            "coverage_pct": round(100.0 * present / max(1, last - first + 1), 1),
            "n_cameras": len(cams[gid]),
            "path_length_m": round(path_mm / 1000.0, 2),
            "mean_speed_mps": round((path_mm / 1000.0) / dwell_s, 2) if dwell_s > 0 else 0.0,
            "span_x_m": round((xy[:, 0].max() - xy[:, 0].min()) / 1000.0, 2),
            "span_y_m": round((xy[:, 1].max() - xy[:, 1].min()) / 1000.0, 2),
            "outliers_dropped": dropped,
        })

    with open(os.path.join(out_dir, "global_tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["global_id", "frame", "world_x", "world_y", "n_cams"])
        w.writeheader(); w.writerows(track_rows)
    sfields = ["global_id", "first_frame", "last_frame", "frames_present", "dwell_s",
               "coverage_pct", "n_cameras", "path_length_m", "mean_speed_mps",
               "span_x_m", "span_y_m", "outliers_dropped"]
    with open(os.path.join(out_dir, "global_id_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader(); w.writerows(summary_rows)

    print(f"[global_tracks] {len(summary_rows)} global identities, "
          f"{len(track_rows)} fused world points -> {out_dir}/")
    print(f"  {'gid':>4} {'dwell_s':>8} {'cov%':>6} {'cams':>5} {'path_m':>8} {'speed':>6} {'span(x,y)m':>14}")
    for s in summary_rows:
        print(f"  {s['global_id']:>4} {s['dwell_s']:>8} {s['coverage_pct']:>6} "
              f"{s['n_cameras']:>5} {s['path_length_m']:>8} {s['mean_speed_mps']:>6} "
              f"{str((s['span_x_m'], s['span_y_m'])):>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="Default: <pred-dir>")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--iqr-k", type=float, default=3.0,
                    help="Tukey fence width for dropping BEV-projection outliers.")
    ap.add_argument("--smooth-win", type=int, default=7,
                    help="Rolling-median window (frames) to de-jitter trajectories.")
    args = ap.parse_args()
    build(args.pred_dir, args.out_dir or args.pred_dir, args.fps, args.iqr_k, args.smooth_win)


if __name__ == "__main__":
    main()
