"""Per-camera occupancy heat-map analytic for MMPTracking.

For each camera in a scene, accumulate where people are over the whole clip and
render a density heat-map overlaid on a scene frame. Two density modes:
  foot   - foot point (bottom-centre of each box): "where people stand/walk"
  dwell  - full bounding box: "where people spend time / body coverage"

Source of detections:
  GT       (default): dataset/MMPTracking_short/<scene>/gt_cam<N>.csv
  pipeline (--pred-dir): <dir>/cam_<N>_predictions.csv  (from --export-predictions)

Run:
    python scripts/eval/camera_heatmap.py --scene lobby_0
    python scripts/eval/camera_heatmap.py --scene lobby_0 --mode dwell
    python scripts/eval/camera_heatmap.py --scene lobby_0 --pred-dir output/eval/exp_lobby_0
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

TURBO = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def load_cameras(args) -> list[tuple[int, pd.DataFrame, str]]:
    """Return [(cam_id, df[left,top,width,height], video_path), ...]."""
    scene_dir = os.path.join(args.short_root, args.scene)
    out = []
    if args.pred_dir:
        # pred cam_id is the 0-indexed source_id; videos are cam1..camN (1-indexed).
        # Map source i -> the i-th sorted video so backgrounds line up.
        videos = sorted(glob.glob(os.path.join(scene_dir, "cam*.mp4")),
                        key=lambda p: int(re.search(r"cam(\d+)\.mp4", p).group(1)))
        csvs = sorted(glob.glob(os.path.join(args.pred_dir, "cam_*_predictions.csv")),
                      key=lambda p: int(re.search(r"cam_(\d+)_", os.path.basename(p)).group(1)))
        for csv in csvs:
            src = int(re.search(r"cam_(\d+)_", os.path.basename(csv)).group(1))
            df = pd.read_csv(csv).rename(columns={"local_track_id": "tid"})
            video = videos[src] if src < len(videos) else _video(scene_dir, src)
            m = re.search(r"cam(\d+)\.mp4", os.path.basename(video))
            cam = int(m.group(1)) if m else src  # real camera number, so it matches GT labels
            out.append((cam, df[["left", "top", "width", "height", "tid"]], video))
    else:
        for csv in sorted(glob.glob(os.path.join(scene_dir, "gt_cam*.csv"))):
            tail = os.path.basename(csv).replace("gt_cam", "").replace(".csv", "")
            if "_" in tail or not tail.isdigit():
                continue
            cam = int(tail)
            df = pd.read_csv(csv).rename(columns={"person_id": "tid"})
            out.append((cam, df[["left", "top", "width", "height", "tid"]], _video(scene_dir, cam)))
    return out


def _video(scene_dir: str, cam: int) -> str:
    return os.path.join(scene_dir, f"cam{cam}.mp4")


def background(video: str, W: int, H: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return np.full((H, W, 3), 60, np.uint8)
    return cv2.resize(fr, (W, H))


def accumulate(df: pd.DataFrame, W: int, H: int, mode: str, visit_cell: int = 8) -> np.ndarray:
    """foot  = +1 per detection at the foot point (dwell-weighted point density)
    dwell = +1 per detection over the whole box (body coverage, dwell-weighted)
    visit = +1 per TRACK per coarse cell it ever steps on (movement/coverage,
            dwell-independent: a person sitting still adds 1, not N frames)."""
    acc = np.zeros((H, W), np.float32)
    L = df["left"].to_numpy(); T = df["top"].to_numpy()
    Wd = df["width"].to_numpy(); Hd = df["height"].to_numpy()
    if mode == "dwell":
        for l, t, w, h in zip(L, T, Wd, Hd):
            x1, y1 = max(0, int(l)), max(0, int(t))
            x2, y2 = min(W, int(l + w)), min(H, int(t + h))
            if x2 > x1 and y2 > y1:
                acc[y1:y2, x1:x2] += 1.0
    elif mode == "visit":
        fx = np.clip((L + Wd / 2).astype(int), 0, W - 1)
        fy = np.clip((T + Hd).astype(int), 0, H - 1)
        cx, cy = fx // visit_cell, fy // visit_cell
        uniq = pd.DataFrame({"t": df["tid"].to_numpy(), "cx": cx, "cy": cy}).drop_duplicates()
        cnt = uniq.groupby(["cy", "cx"]).size().reset_index(name="n")
        px = np.clip(cnt["cx"].to_numpy() * visit_cell + visit_cell // 2, 0, W - 1)
        py = np.clip(cnt["cy"].to_numpy() * visit_cell + visit_cell // 2, 0, H - 1)
        np.add.at(acc, (py, px), cnt["n"].to_numpy().astype(np.float32))
    else:  # foot point
        fx = np.clip((L + Wd / 2).astype(int), 0, W - 1)
        fy = np.clip((T + Hd).astype(int), 0, H - 1)
        np.add.at(acc, (fy, fx), 1.0)
    return acc


def smooth(acc: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(acc, (0, 0), sigma)


def grid_stats(acc: np.ndarray, grid_w: int, fps: float, mode: str) -> tuple[np.ndarray, dict]:
    """Downsample the full-res accumulator to a grid_w-wide occupancy grid and
    compute per-camera summary stats — the structured 'information' a real
    analytics store keeps (vs the rendered picture).

    Returns (grid[gh,gw], stats). Grid cell value = summed occupancy in that
    cell. For mode=foot the value is #person-detections (≈ person-frames); at
    `fps` that converts to dwell-seconds."""
    H, W = acc.shape
    gw = max(1, grid_w)
    gh = max(1, round(gw * H / W))
    grid = cv2.resize(acc, (gw, gh), interpolation=cv2.INTER_AREA) * (W / gw) * (H / gh)
    total = float(acc.sum())
    flat_i = int(grid.argmax())
    py, px = divmod(flat_i, gw)
    occupied = int((grid > grid.max() * 0.05).sum()) if grid.max() > 0 else 0
    # dwell-seconds only meaningful for foot mode (cell value = person-frames).
    # foot: count = person-frames; visit: count = unique track-visits; dwell: box-pixel coverage.
    foot = mode == "foot"
    stats = {
        "mode": mode,
        "grid_w": gw, "grid_h": gh,
        "total_detections": round(total, 1),
        "peak_cell_xy": [px, py],
        "peak_cell_px": [int((px + 0.5) * W / gw), int((py + 0.5) * H / gh)],
        "peak_value": round(float(grid.max()), 1),
        "peak_dwell_seconds": round(float(grid.max()) / fps, 1) if (foot and fps) else None,
        "occupied_cells": occupied,
        "occupied_fraction": round(occupied / (gw * gh), 3),
    }
    return grid, stats


def colorize(dens: np.ndarray, gamma: float, vmax_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (heat_bgr, alpha[0..1]) from a smoothed density map."""
    pos = dens[dens > 0]
    vmax = np.percentile(pos, vmax_pct) if pos.size else 1.0
    norm = np.clip(dens / (vmax + 1e-9), 0, 1) ** gamma
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), TURBO)
    return heat, norm


def render(bg: np.ndarray, heat: np.ndarray, alpha: np.ndarray, max_alpha: float) -> np.ndarray:
    a = (alpha * max_alpha)[..., None]
    return (bg * (1 - a) + heat * a).astype(np.uint8)


def density_metrics(gt_d: np.ndarray, pred_d: np.ndarray) -> tuple[float, float, float]:
    """Compare two smoothed density maps (saliency-map metrics).
    CC = Pearson corr (1=identical shape); SIM = histogram intersection [0..1];
    KL = KL(gt||pred) [0=identical, higher=worse]."""
    g = gt_d.flatten().astype(np.float64); p = pred_d.flatten().astype(np.float64)
    cc = float(np.corrcoef(g, p)[0, 1]) if g.std() > 1e-9 and p.std() > 1e-9 else 0.0
    gp = g / (g.sum() + 1e-12); pp = p / (p.sum() + 1e-12)
    sim = float(np.minimum(gp, pp).sum())
    kl = float(np.sum(gp * np.log((gp + 1e-12) / (pp + 1e-12))))
    return cc, sim, kl


def diff_overlay(bg: np.ndarray, gt_d: np.ndarray, pred_d: np.ndarray, max_alpha: float) -> np.ndarray:
    """Red where pred over-counts (hallucination), blue where pred under-counts (miss)."""
    m = max(gt_d.max(), pred_d.max(), 1e-9)
    d = np.clip(pred_d / m - gt_d / m, -1, 1)
    heat = np.zeros((*d.shape, 3), np.uint8)
    heat[..., 2] = (np.clip(d, 0, 1) * 255).astype(np.uint8)    # red  = pred > gt
    heat[..., 0] = (np.clip(-d, 0, 1) * 255).astype(np.uint8)   # blue = pred < gt
    a = (np.abs(d) * max_alpha)[..., None]
    return (bg * (1 - a) + heat * a).astype(np.uint8)


def compare_gt(args) -> None:
    """Render GT vs pred heatmaps + diff map per camera and print CC/SIM/KL."""
    if not args.pred_dir:
        print("[ERROR] --compare-gt requires --pred-dir"); return
    scene_dir = os.path.join(args.short_root, args.scene)
    pred = load_cameras(args)  # pred cameras, source-id order
    gt_files = sorted(
        (f for f in glob.glob(os.path.join(scene_dir, "gt_cam*.csv"))
         if os.path.basename(f).replace("gt_cam", "").replace(".csv", "").isdigit()),
        key=lambda p: int(re.search(r"gt_cam(\d+)\.csv", p).group(1)))
    gt_dfs = [pd.read_csv(f).rename(columns={"person_id": "tid"})
              [["left", "top", "width", "height", "tid"]] for f in gt_files]

    rows, ms = [], []
    for i, (cam, pdf, video) in enumerate(pred):
        if i >= len(gt_dfs):
            break
        gdf = gt_dfs[i]
        cap = cv2.VideoCapture(video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        cap.release()
        bg = background(video, W, H)
        gd = smooth(accumulate(gdf, W, H, args.mode, args.visit_cell), args.sigma)
        pdn = smooth(accumulate(pdf, W, H, args.mode, args.visit_cell), args.sigma)
        cc, sim, kl = density_metrics(gd, pdn)
        ms.append((cc, sim, kl))
        gpan = render(bg, *colorize(gd, args.gamma, args.vmax_pct), args.alpha)
        ppan = render(bg, *colorize(pdn, args.gamma, args.vmax_pct), args.alpha)
        dpan = diff_overlay(bg, gd, pdn, args.alpha)
        for img, t in [(gpan, "GT"), (ppan, "PRED"), (dpan, "DIFF  blue=miss red=halluc")]:
            cv2.putText(img, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(gpan, f"cam{cam}  CC={cc:.2f} SIM={sim:.2f} KL={kl:.2f}",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([gpan, ppan, dpan]))
        print(f"[cam{cam}] CC={cc:.3f}  SIM={sim:.3f}  KL={kl:.3f}  "
              f"(gt {len(gdf)} vs pred {len(pdf)} det)")
    if ms:
        a = np.mean(ms, axis=0)
        print(f"[avg ] CC={a[0]:.3f}  SIM={a[1]:.3f}  KL={a[2]:.3f}   "
              "(CC/SIM higher=better, KL lower=better)")
    if rows:
        p = os.path.join(args.out_dir, f"{args.scene}_{args.mode}_compare.png")
        cv2.imwrite(p, np.vstack(rows))
        print(f"[montage] {p}  (columns: GT | PRED | DIFF)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-root", default="dataset/MMPTracking_short")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pred-dir", default=None, help="Use pipeline predictions instead of GT.")
    ap.add_argument("--mode", choices=["foot", "dwell", "visit"], default="foot")
    ap.add_argument("--visit-cell", type=int, default=8,
                    help="Cell size (px) for --mode visit dedup (per track per cell).")
    ap.add_argument("--sigma", type=float, default=12.0, help="Gaussian smoothing (px).")
    ap.add_argument("--gamma", type=float, default=0.5, help="<1 boosts low-density areas.")
    ap.add_argument("--vmax-pct", type=float, default=99.0)
    ap.add_argument("--alpha", type=float, default=0.65, help="Max overlay opacity.")
    ap.add_argument("--compare-gt", action="store_true",
                    help="With --pred-dir: render GT vs pred + diff map and print CC/SIM/KL per camera.")
    ap.add_argument("--dump-grid", type=int, default=0, metavar="GRID_W",
                    help="Also write structured heatmap data: per-camera occupancy "
                         "grid CSV (GRID_W wide) + a stats JSON. 0 = off.")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Frame rate, to convert occupancy counts -> dwell seconds in --dump-grid.")
    ap.add_argument("--out-dir", default="output/heatmap")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare_gt:
        compare_gt(args)
        return

    cams = load_cameras(args)
    if not cams:
        print("[ERROR] no cameras found"); return
    src = "pred" if args.pred_dir else "gt"
    panels = []
    all_stats = {}
    for cam, df, video in cams:
        cap = cv2.VideoCapture(video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        cap.release()
        bg = background(video, W, H)
        acc = accumulate(df, W, H, args.mode, args.visit_cell)
        if args.dump_grid:
            grid, st = grid_stats(acc, args.dump_grid, args.fps, args.mode)
            np.savetxt(os.path.join(args.out_dir, f"{args.scene}_cam{cam}_{args.mode}_grid.csv"),
                       grid, fmt="%.2f", delimiter=",")
            all_stats[f"cam{cam}"] = st
            secs = f" ({st['peak_dwell_seconds']}s dwell)" if st["peak_dwell_seconds"] else ""
            print(f"[cam{cam}] grid {st['grid_w']}x{st['grid_h']} -> peak {st['peak_value']}{secs} "
                  f"at cell {st['peak_cell_xy']}, {st['occupied_fraction']*100:.0f}% occupied")
        heat, alpha = colorize(smooth(acc, args.sigma), args.gamma, args.vmax_pct)
        out = render(bg, heat, alpha, args.alpha)
        cv2.putText(out, f"{args.scene} cam{cam} ({args.mode}, {src}, {len(df)} det)",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        p = os.path.join(args.out_dir, f"{args.scene}_cam{cam}_{args.mode}_{src}.png")
        cv2.imwrite(p, out)
        print(f"[cam{cam}] {len(df)} detections -> {p}")
        panels.append(out)

    if args.dump_grid and all_stats:
        import json
        sp = os.path.join(args.out_dir, f"{args.scene}_{args.mode}_{src}_stats.json")
        with open(sp, "w") as f:
            json.dump({"scene": args.scene, "mode": args.mode, "source": src,
                       "fps": args.fps, "cameras": all_stats}, f, indent=2)
        print(f"[data] {sp}  (per-camera occupancy grids + stats)")

    # group montage: every camera of the scene tiled in one picture
    import math as _m
    cols = max(1, int(_m.ceil(_m.sqrt(len(panels)))))
    h, w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    th = 26
    mont = np.full((th + rows * h, cols * w, 3), 20, np.uint8)
    cv2.putText(mont, f"{args.scene}   mode={args.mode}   {src}   {len(panels)} cameras",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, pim in enumerate(panels):
        r, c = divmod(i, cols)
        mont[th + r * h:th + (r + 1) * h, c * w:(c + 1) * w] = cv2.resize(pim, (w, h))
    mp = os.path.join(args.out_dir, f"{args.scene}_{args.mode}_{src}_group.png")
    cv2.imwrite(mp, mont)
    print(f"[group] {mp}  ({len(panels)} cameras, {cols}x{rows} grid)")


if __name__ == "__main__":
    main()
