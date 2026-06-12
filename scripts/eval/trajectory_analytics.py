"""Step 3 — trajectory analytics on the global-ID store (scripts/eval/global_tracks.py).

Reads <store>/global_tracks.csv (fused world points per global_id) and renders,
in the shared world ground-plane (top-down):

  journey_map.png   per-identity world polyline (one colour per global_id) +
                    start (green) / end (red) markers  -> "who went where"
  dwell_map.png     time-weighted occupancy heatmap (turbo)               -> "where time is spent"
  od_matrix.png/.csv  zone-to-zone transition counts over a GxG grid       -> "flow between zones"
  time_in_zone.csv  seconds each global_id spends in each grid zone

Run:
    python scripts/eval/trajectory_analytics.py --store output/eval/lobby0_store --fps 25
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

TURBO = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def load_tracks(store: str):
    g: dict[int, list] = defaultdict(list)
    with open(os.path.join(store, "global_tracks.csv")) as f:
        for r in csv.DictReader(f):
            g[int(r["global_id"])].append(
                (int(r["frame"]), float(r["world_x"]), float(r["world_y"])))
    for gid in g:
        g[gid].sort()
    return g


def world_bounds(tracks, pad_frac=0.05):
    xs = np.array([p[1] for t in tracks.values() for p in t])
    ys = np.array([p[2] for t in tracks.values() for p in t])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return x0 - px, x1 + px, y0 - py, y1 + py


def make_mapper(bounds, W, H):
    x0, x1, y0, y1 = bounds
    def to_px(x, y):
        u = int((x - x0) / max(1e-6, x1 - x0) * (W - 1))
        v = int((y - y0) / max(1e-6, y1 - y0) * (H - 1))
        return np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    return to_px


def color_for(gid: int):
    return tuple(int(c) for c in np.random.RandomState(gid * 9 + 5).randint(60, 256, 3))


def draw_scale(img, bounds, W, H):
    x0, x1, y0, y1 = bounds
    cv2.putText(img, f"world {(x1-x0)/1000:.1f}m x {(y1-y0)/1000:.1f}m  (top-down)",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def journey_map(tracks, bounds, W, H, out):
    img = np.full((H, W, 3), 28, np.uint8)
    for gid, t in sorted(tracks.items()):
        c = color_for(gid)
        pts = [make_mapper(bounds, W, H)(x, y) for _, x, y in t]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, c, 2, cv2.LINE_AA)
        cv2.circle(img, pts[0], 5, (0, 220, 0), -1)   # start
        cv2.circle(img, pts[-1], 5, (0, 0, 230), -1)  # end
        cv2.putText(img, f"G{gid}", (pts[-1][0] + 4, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
    draw_scale(img, bounds, W, H)
    cv2.imwrite(out, img)


def dwell_map(tracks, bounds, W, H, fps, sigma, out):
    acc = np.zeros((H, W), np.float32)
    to_px = make_mapper(bounds, W, H)
    for t in tracks.values():
        for _, x, y in t:
            u, v = to_px(x, y)
            acc[v, u] += 1.0 / fps   # each sample = 1 frame = 1/fps seconds
    dens = cv2.GaussianBlur(acc, (0, 0), sigma)
    vmax = np.percentile(dens[dens > 0], 99) if (dens > 0).any() else 1.0
    norm = np.clip(dens / (vmax + 1e-9), 0, 1) ** 0.5
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), TURBO)
    bg = np.full((H, W, 3), 255, np.uint8)
    a = norm[..., None]
    img = (bg * (1 - a) + heat * a).astype(np.uint8)
    draw_scale(img, bounds, W, H)
    cv2.imwrite(out, img)


def od_and_zones(tracks, bounds, grid, fps, store):
    x0, x1, y0, y1 = bounds
    def zone(x, y):
        cx = min(grid - 1, max(0, int((x - x0) / max(1e-6, x1 - x0) * grid)))
        cy = min(grid - 1, max(0, int((y - y0) / max(1e-6, y1 - y0) * grid)))
        return cy * grid + cx
    nz = grid * grid
    od = np.zeros((nz, nz), np.int64)
    tiz = defaultdict(lambda: np.zeros(nz))   # gid -> seconds per zone
    for gid, t in tracks.items():
        zs = [zone(x, y) for _, x, y in t]
        for z in zs:
            tiz[gid][z] += 1.0 / fps
        for a, b in zip(zs, zs[1:]):
            if a != b:
                od[a, b] += 1
    # write OD matrix + time-in-zone
    with open(os.path.join(store, "od_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_zone\\to_zone"] + list(range(nz)))
        for i in range(nz):
            w.writerow([i] + od[i].tolist())
    with open(os.path.join(store, "time_in_zone.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id"] + [f"zone{z}_s" for z in range(nz)])
        for gid in sorted(tiz):
            w.writerow([gid] + [round(v, 1) for v in tiz[gid]])
    # OD heatmap image
    img = cv2.applyColorMap(
        (np.clip(od / (od.max() + 1e-9), 0, 1) ** 0.5 * 255).astype(np.uint8), TURBO)
    img = cv2.resize(img, (nz * 36, nz * 36), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(store, "od_matrix.png"), img)
    return int(od.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="dir with global_tracks.csv")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--w", type=int, default=900)
    ap.add_argument("--h", type=int, default=700)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--grid", type=int, default=4, help="GxG zones for OD / time-in-zone")
    args = ap.parse_args()

    tracks = load_tracks(args.store)
    bounds = world_bounds(tracks)
    journey_map(tracks, bounds, args.w, args.h, os.path.join(args.store, "journey_map.png"))
    dwell_map(tracks, bounds, args.w, args.h, args.fps, args.sigma,
              os.path.join(args.store, "dwell_map.png"))
    transitions = od_and_zones(tracks, bounds, args.grid, args.fps, args.store)
    print(f"[analytics] {len(tracks)} identities -> {args.store}/")
    print(f"  journey_map.png  dwell_map.png  od_matrix.{{csv,png}}  time_in_zone.csv")
    print(f"  grid={args.grid}x{args.grid} zones, {transitions} zone transitions")


if __name__ == "__main__":
    main()
