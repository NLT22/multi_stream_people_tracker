#!/usr/bin/env python3
"""Occupancy heatmaps from a pipeline export (production_todo §6 — offline, no GPU).

Reads the per-camera prediction CSVs (`cam_*_predictions.csv`) that the production
pipeline already writes, accumulates each detection's foot point (bbox bottom-centre)
into a 2D density grid per camera, renders a colour heatmap PNG, tiles all cameras
into one montage, and dumps an occupancy grid CSV + a stats JSON (peak hotspot,
occupied fraction, detections) per camera.

Per-camera occupancy depends only on detection quality, not on cross-camera identity,
so the heatmap is meaningful even where Global IDF1 is weak (e.g. retail).

  python scripts/eval/heatmap_from_export.py \
      --export-dir output/runs/<ts>_<preset>/export --out-dir output/heatmap/<name>
  # optionally restrict to some cameras / change blur or grid:
  python scripts/eval/heatmap_from_export.py --export-dir <dir> --out-dir <dir> \
      --cams 16 17 18 19 --grid-h 180 --grid-w 320 --sigma 4
"""
from __future__ import annotations
import argparse, glob, json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image


def _cam_id(path: str) -> int:
    m = re.search(r"cam_(\d+)_predictions", Path(path).name)
    return int(m.group(1)) if m else -1


def _density(df: pd.DataFrame, pw: float, ph: float, gw: int, gh: int, sigma: float):
    # foot point = bbox bottom-centre, the convention nvdsanalytics also uses.
    fx = (df["left"] + df["width"] / 2.0).to_numpy()
    fy = (df["top"] + df["height"]).to_numpy()
    gx = np.clip((fx / pw * gw).astype(int), 0, gw - 1)
    gy = np.clip((fy / ph * gh).astype(int), 0, gh - 1)
    grid = np.zeros((gh, gw), dtype=np.float64)
    np.add.at(grid, (gy, gx), 1.0)
    return gaussian_filter(grid, sigma=sigma), grid


def _render(dens: np.ndarray, out_png: Path, title: str):
    norm = dens / dens.max() if dens.max() > 0 else dens
    rgba = matplotlib.colormaps["jet"](norm)
    rgba[..., 3] = np.clip(norm * 1.4, 0, 1)  # transparent where empty
    plt.figure(figsize=(6, 6 * dens.shape[0] / max(dens.shape[1], 1)))
    plt.imshow(rgba)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def _stats(raw: np.ndarray, dens: np.ndarray) -> dict:
    peak = np.unravel_index(int(dens.argmax()), dens.shape) if dens.max() > 0 else (0, 0)
    occupied = float((dens > dens.max() * 0.05).mean()) if dens.max() > 0 else 0.0
    return {
        "detections": int(raw.sum()),
        "peak_cell_rc": [int(peak[0]), int(peak[1])],
        "occupied_fraction": round(occupied, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cams", nargs="+", type=int, default=None)
    ap.add_argument("--pred-w", type=float, default=640.0)
    ap.add_argument("--pred-h", type=float, default=360.0)
    ap.add_argument("--grid-w", type=int, default=320)
    ap.add_argument("--grid-h", type=int, default=180)
    ap.add_argument("--sigma", type=float, default=4.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(args.export_dir / "cam_*_predictions.csv")), key=_cam_id)
    if not files:
        raise SystemExit(f"no cam_*_predictions.csv in {args.export_dir}")

    pngs, stats = [], {}
    for f in files:
        cid = _cam_id(f)
        if args.cams is not None and cid not in args.cams:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        dens, raw = _density(df, args.pred_w, args.pred_h, args.grid_w, args.grid_h, args.sigma)
        png = args.out_dir / f"cam_{cid}_heatmap.png"
        _render(dens, png, f"cam {cid}  ({int(raw.sum())} dets)")
        np.savetxt(args.out_dir / f"cam_{cid}_grid.csv", raw.astype(int), fmt="%d", delimiter=",")
        stats[f"cam_{cid}"] = _stats(raw, dens)
        pngs.append(png)
        print(f"[heatmap] cam {cid}: {int(raw.sum())} dets -> {png.name}")

    (args.out_dir / "occupancy_stats.json").write_text(json.dumps(stats, indent=2))

    # montage: tile all camera heatmaps into one image
    if pngs:
        imgs = [Image.open(p) for p in pngs]
        n = len(imgs)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        w = max(im.width for im in imgs)
        h = max(im.height for im in imgs)
        canvas = Image.new("RGB", (cols * w, rows * h), "white")
        for i, im in enumerate(imgs):
            canvas.paste(im, ((i % cols) * w, (i // cols) * h))
        canvas.save(args.out_dir / "montage.png")
        print(f"[heatmap] montage -> {args.out_dir/'montage.png'}  ({n} cams)")
    print(f"[heatmap] stats -> {args.out_dir/'occupancy_stats.json'}")


if __name__ == "__main__":
    main()
