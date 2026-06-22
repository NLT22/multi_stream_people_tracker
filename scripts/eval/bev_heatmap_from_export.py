#!/usr/bin/env python3
"""Bird's-eye-view (floor-plan) occupancy heatmap merging cameras into one ground
map (production_todo §6). Offline, no GPU.

Uses src/reid/geometry.py to project each detection's foot point (bbox bottom-centre,
in 640x360 calibration space) to world XY (mm, Z=0 floor) via the scene calibration,
then accumulates ALL the scene's cameras into a single ground-plane density map.

  python scripts/eval/bev_heatmap_from_export.py \
      --export-dir output/runs/<ts>/export --cams 16 17 18 19 \
      --calib dataset/MMPTracking/MMPTracking_validation/validation/calibrations/retail/calibrations.json \
      --out output/heatmap/retail_bev.png

`--cams` are the global cam indices for this scene in the export, in source order;
they map in order to the sorted calibration CameraIds (override with --cam-ids).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.reid.geometry import GroundPlaneGeometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path, help="calibrations.json for the scene env")
    ap.add_argument("--cams", required=True, nargs="+", type=int,
                    help="global cam indices in the export (source order)")
    ap.add_argument("--cam-ids", nargs="+", type=int, default=None,
                    help="calibration CameraIds matching --cams (default: sorted calib ids)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calib-w", type=float, default=640.0)
    ap.add_argument("--calib-h", type=float, default=360.0)
    ap.add_argument("--pred-w", type=float, default=640.0)
    ap.add_argument("--pred-h", type=float, default=360.0)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=2.0)
    args = ap.parse_args()

    calib = json.loads(args.calib.read_text())
    geo = GroundPlaneGeometry(calib)
    calib_ids = sorted(c["CameraId"] for c in calib.get("Cameras", []))
    cam_ids = args.cam_ids or calib_ids[: len(args.cams)]
    if len(cam_ids) != len(args.cams):
        raise SystemExit(f"--cams ({len(args.cams)}) and CameraIds ({len(cam_ids)}) length mismatch")

    sx, sy = args.calib_w / args.pred_w, args.calib_h / args.pred_h
    xs, ys = [], []
    for g, cid in zip(args.cams, cam_ids):
        f = args.export_dir / f"cam_{g}_predictions.csv"
        if not f.exists():
            print(f"[bev] skip missing {f.name}")
            continue
        if not geo.has_camera(cid):
            print(f"[bev] calib has no CameraId {cid}; skip cam {g}")
            continue
        df = pd.read_csv(f)
        u = (df["left"] + df["width"] / 2.0).to_numpy() * sx
        v = (df["top"] + df["height"]).to_numpy() * sy
        n = 0
        for uu, vv in zip(u, v):
            w = geo.foot_to_world(cid, float(uu), float(vv))
            if w is not None:
                xs.append(w[0]); ys.append(w[1]); n += 1
        print(f"[bev] cam {g} (CameraId {cid}): {n}/{len(df)} foot points projected")

    if not xs:
        raise SystemExit("no world points projected — check --cams/--cam-ids/calib")
    xs = np.asarray(xs) / 1000.0  # mm -> m
    ys = np.asarray(ys) / 1000.0

    # robust bounds (ignore 1% outliers)
    xlo, xhi = np.percentile(xs, [0.5, 99.5])
    ylo, yhi = np.percentile(ys, [0.5, 99.5])
    H, xe, ye = np.histogram2d(xs, ys, bins=args.bins,
                               range=[[xlo, xhi], [ylo, yhi]])
    H = gaussian_filter(H.T, sigma=args.sigma)  # .T so rows=Y

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8 * (yhi - ylo) / max(xhi - xlo, 1e-6)))
    plt.imshow(H, origin="lower", extent=[xlo, xhi, ylo, yhi],
               cmap="jet", aspect="equal")
    plt.colorbar(label="occupancy density")
    plt.xlabel("world X (m)"); plt.ylabel("world Y (m)")
    plt.title(f"BEV floor heatmap — {len(xs)} foot points, {len(args.cams)} cams")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    plt.close()
    print(f"[bev] wrote {args.out}  (world extent X[{xlo:.1f},{xhi:.1f}] Y[{ylo:.1f},{yhi:.1f}] m)")


if __name__ == "__main__":
    main()
