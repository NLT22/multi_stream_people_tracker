#!/usr/bin/env python3
"""Offline people-analytics visualizations for one environment, from a pipeline export.

Produces (all from cam_*_predictions.csv + the camera videos + scene calibration):
  1. Per-camera heatmaps blended over a real camera frame, 3 modes:
        foot  = occupancy (every detection's foot point)
        dwell = time-weighted lingering (stationary detections weighted up)
        visit = coverage (count of distinct tracks per cell)
  2. Top-down BEV floor view:
        bev_trajectory.png = each person's path on the floor plane (world XY)
        bev_heatmap.png    = floor-plane occupancy density
  3. Time-lapse videos (heatmap builds up over time):
        timelapse_cam.mp4  = per-camera heatmaps over background, tiled
        timelapse_bev.mp4  = top-down floor density growing over time

  PYTHONPATH=. python scripts/eval/venv_visualize.py \
      --export-dir output/demo/64pm_office_0/export \
      --video-dir  dataset/MMPTracking_10minute/val/64pm_office_0 \
      --calib dataset/MMPTracking/MMPTracking_validation/validation/calibrations/office/calibrations.json \
      --cams 0 1 2 3 --out-dir output/demo/64pm_office_0/viz
"""
from __future__ import annotations
import argparse, glob, re, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image, ImageDraw
import colorsys

from src.reid.geometry import GroundPlaneGeometry

JET = colormaps["jet"]


# ----- clean top-down "map" helpers (no matplotlib chart chrome) -------------
def _bev_dims(xlo, xhi, ylo, yhi, px_w=760):
    aspect = (yhi - ylo) / max(xhi - xlo, 1e-6)
    H = int(np.clip(px_w * aspect, 240, 1500))
    return px_w - (px_w % 2), H - (H % 2)   # even dims (libx264/yuv420p requires it)


def _w2px(X, Y, b):
    xlo, xhi, ylo, yhi, W, H = b
    px = (np.asarray(X, float) - xlo) / max(xhi - xlo, 1e-6) * (W - 1)
    py = (1 - (np.asarray(Y, float) - ylo) / max(yhi - ylo, 1e-6)) * (H - 1)  # up = +Y
    return px, py


def _bev_base(b, grid_m=2.0) -> Image.Image:
    """Dark floor canvas with a faint metric grid — a map look, not a chart."""
    xlo, xhi, ylo, yhi, W, H = b
    img = Image.new("RGB", (W, H), (26, 28, 32))
    d = ImageDraw.Draw(img)
    x = np.ceil(xlo / grid_m) * grid_m
    while x <= xhi:
        px, _ = _w2px(x, ylo, b); d.line([(px, 0), (px, H)], fill=(44, 47, 53)); x += grid_m
    y = np.ceil(ylo / grid_m) * grid_m
    while y <= yhi:
        _, py = _w2px(xlo, y, b); d.line([(0, py), (W, py)], fill=(44, 47, 53)); y += grid_m
    return img


def _palette(i, n):
    r, g, bl = colorsys.hsv_to_rgb((i / max(n, 1)) % 1.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(bl * 255)


def _caption(img: Image.Image, text: str):
    ImageDraw.Draw(img).text((8, 6), text, fill=(235, 235, 235))


def _iou_ltwh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def _dedup_boxes(dets, iou_thr=0.6):
    """Per-frame cleanup of fragmented detections: (1) keep one box per global ID
    (largest = closest/most reliable), (2) NMS overlapping boxes across IDs."""
    best = {}                                   # gid -> (area, det)
    for d in dets:
        g, l, t, w, h = d
        a = w * h
        if g not in best or a > best[g][0]:
            best[g] = (a, d)
    kept = sorted((v[1] for v in best.values()), key=lambda d: -d[3] * d[4])
    out = []
    for d in kept:
        if all(_iou_ltwh(d[1:], o[1:]) <= iou_thr for o in out):
            out.append(d)
    return out


def _cam_tracking_video(preds, cam_order, gid_of, video_dir, W, H, fps, n_frames,
                        out_path, label):
    """Camera-view tracking video with AUTHORITATIVE BUFFERED IDs.

    Labels each box by its track's buffered Global ID (majority vote over the
    track's per-detection assignments from live_buffered), so every box of a known
    person is labelled — no live 'GID:?'. Tracks never assigned a buffered ID
    (too short / no embedding ever) are dropped. Tiles all cameras; offline cv2.
    """
    import math
    from collections import deque, Counter
    try:
        import cv2
    except ImportError:
        print("[viz] cam_tracking needs opencv (cv2); skipped")
        return False
    if not gid_of:
        print("[viz] cam_tracking needs --assign-csv (buffered IDs); skipped")
        return False

    # track-level buffered gid (majority vote per (cam, local_track_id))
    votes: dict = {}
    for (c, f, lt), g in gid_of.items():
        if g >= 0:
            votes.setdefault((c, lt), Counter())[g] += 1
    track_gid = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    gids = sorted(set(track_gid.values()))
    if not gids:
        return False
    color = {g: _palette(i, len(gids)) for i, g in enumerate(gids)}  # RGB

    # boxes[c][frame] = [(gid, l, t, w, h)] for tracks that have a buffered ID
    boxes: dict = {}
    for c in cam_order:
        bf: dict = {}
        for r in preds[c].itertuples():
            g = track_gid.get((c, int(r.local_track_id)))
            if g is None:
                continue
            bf.setdefault(int(r.frame_no_cam), []).append(
                (g, float(r.left), float(r.top), float(r.width), float(r.height)))
        boxes[c] = bf

    caps = {c: cv2.VideoCapture(str(_video_for(video_dir, j)))
            for j, c in enumerate(cam_order)}
    total = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) or 0 for c in cam_order)
    if total <= 0:
        for cap in caps.values():
            cap.release()
        return False
    step = max(1, total // max(1, n_frames))
    cols = math.ceil(math.sqrt(len(cam_order)))
    rows = math.ceil(len(cam_order) / cols)
    trails = {(c, g): deque(maxlen=22) for c in cam_order for g in gids}
    tmpd = Path(tempfile.mkdtemp(prefix="camtrk_"))

    fidx = out_idx = 0
    while True:
        rets, ok = {}, True
        for c in cam_order:
            r, fr = caps[c].read()
            if not r:
                ok = False
                break
            rets[c] = fr
        if not ok:
            break
        if fidx % step == 0:
            tiles = []
            for c in cam_order:
                img = rets[c]
                dets = _dedup_boxes(boxes[c].get(fidx, []))
                present = {g for (g, *_b) in dets}
                for g, l, t, w, h in dets:
                    bgr = color[g][::-1]
                    cv2.rectangle(img, (int(l), int(t)), (int(l + w), int(t + h)), bgr, 2)
                    cv2.putText(img, f"ID:{g}", (int(l), max(12, int(t) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
                    trails[(c, g)].append((int(l + w / 2), int(t + h)))
                for g in gids:
                    if g not in present:
                        trails[(c, g)].clear()
                        continue
                    pts = list(trails[(c, g)])
                    for k in range(1, len(pts)):
                        cv2.line(img, pts[k - 1], pts[k], color[g][::-1], 1, cv2.LINE_AA)
                tiles.append(img)
            while len(tiles) < rows * cols:
                tiles.append(np.zeros((H, W, 3), np.uint8))
            canvas = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                                for r in range(rows)])
            cv2.putText(canvas, f"{label}  -  buffered-ID tracking ({len(gids)} people)",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(tmpd / f"f{out_idx:04d}.png"), canvas)
            out_idx += 1
        fidx += 1
    for cap in caps.values():
        cap.release()
    if out_idx == 0:
        return False
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmpd / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                   capture_output=True)
    return True


def _bev_tracking_video(per_frame, b, out_path, fps, n_frames, label):
    """AI-City-style top-down: each buffered ID = a stable colored marker moving on
    the floor map with a trail + a live people count. Positions are binned per
    video frame, then each track is gap-interpolated across short occlusions so
    markers move continuously (no ghost flicker) instead of blinking on/off."""
    from collections import deque
    frames = sorted(per_frame)
    if len(frames) < 2:
        return False
    bins = [list(x) for x in np.array_split(np.array(frames), min(n_frames, len(frames)))]
    nb = len(bins)

    # per-bin averaged position for each gid
    pos = {}
    for s, grp in enumerate(bins):
        for f in grp:
            for g, xy in per_frame[f].items():
                pos.setdefault(g, {}).setdefault(s, []).append(xy)
    for g in pos:
        pos[g] = {s: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v])))
                  for s, v in pos[g].items()}
    # keep only IDs present in >=4% of bins (drop ultra-brief spurious tracks)
    gids = sorted(g for g in pos if len(pos[g]) >= max(3, 0.04 * nb))
    color = {g: _palette(i, len(gids)) for i, g in enumerate(gids)}

    # densify: interpolate across gaps <= maxgap bins (no ghosting); leave longer
    # gaps empty (person genuinely left view)
    maxgap = max(6, nb // 40)
    dense = {}
    for g in gids:
        bs = sorted(pos[g]); dd = {}
        for i, s0 in enumerate(bs):
            dd[s0] = pos[g][s0]
            if i + 1 < len(bs):
                s1 = bs[i + 1]
                if 1 < s1 - s0 <= maxgap:
                    (x0, y0), (x1, y1) = pos[g][s0], pos[g][s1]
                    for s in range(s0 + 1, s1):
                        t = (s - s0) / (s1 - s0)
                        dd[s] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        dense[g] = dd

    base = _bev_base(b)
    trails = {g: deque(maxlen=24) for g in gids}
    maxseg = 0.12 * max(b[4], b[5])
    tmpd = Path(tempfile.mkdtemp(prefix="bevtrk_"))
    for s in range(nb):
        img = base.copy(); d = ImageDraw.Draw(img); active = 0
        for g in gids:
            col = color[g]; here = dense[g].get(s)
            if here is not None:
                px, py = _w2px(here[0], here[1], b)
                tr = trails[g]
                if tr and np.hypot(px - tr[-1][0], py - tr[-1][1]) > maxseg:
                    tr.clear()
                tr.append((float(px), float(py))); active += 1
            else:
                trails[g].clear()                     # gone -> no stale trail
            pts = list(trails[g])
            if len(pts) >= 4:
                a = np.array(pts, dtype=float)
                # endpoint-preserving smoothing (mode='nearest'); np.convolve 'same'
                # zero-pads the ends and drags the last point — where the dot sits —
                # toward the corner, detaching the marker from its trail.
                a[:, 0] = uniform_filter1d(a[:, 0], size=3, mode="nearest")
                a[:, 1] = uniform_filter1d(a[:, 1], size=3, mode="nearest")
                pts = a.tolist()
            for k in range(1, len(pts)):
                if np.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]) > maxseg:
                    continue
                fr = k / len(pts)
                d.line([pts[k - 1], pts[k]], fill=tuple(int(c * (0.25 + 0.75 * fr)) for c in col), width=2)
            if here is not None:
                # draw the marker at the RAW current position (trail end), never the
                # smoothed value, so the dot always sits on the person's real spot.
                px, py = trails[g][-1]
                d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=col)
                d.text((px + 8, py - 6), str(g), fill=(255, 255, 255))
        d.rectangle([0, 0, 250, 22], fill=(0, 0, 0))
        d.text((8, 6), f"{label}  -  top-down tracking   |   people: {active}", fill=(240, 240, 240))
        img.save(tmpd / f"f{s:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmpd / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], capture_output=True)
    return True


# ----------------------------- helpers ---------------------------------------
def _extract_bg(video: Path, w: int, h: int) -> np.ndarray:
    """Grab a mid-clip frame as the heatmap background (RGB, h x w)."""
    tmp = Path(tempfile.mktemp(suffix=".png"))
    subprocess.run(["ffmpeg", "-y", "-ss", "60", "-i", str(video), "-frames:v", "1",
                    str(tmp)], capture_output=True)
    if not tmp.exists():
        return np.full((h, w, 3), 40, np.uint8)
    img = Image.open(tmp).convert("RGB").resize((w, h), Image.BILINEAR)
    tmp.unlink(missing_ok=True)
    return np.asarray(img)


def _foot(df: pd.DataFrame):
    return (df["left"] + df["width"] / 2.0).to_numpy(), (df["top"] + df["height"]).to_numpy()


def accumulate(df: pd.DataFrame, W: int, H: int, mode: str, visit_cell: int = 8) -> np.ndarray:
    """Three genuinely distinct maps (ported from the archived camera_heatmap):
      foot  = +1 per detection at the FOOT POINT  -> sharp where people stand
      dwell = +1 per detection over the whole BOX -> broad body-area coverage
      visit = +1 per unique (track, coarse cell)  -> movement/coverage, time-INDEPENDENT
              (a person sitting still adds 1, not N frames; transit areas light up)."""
    acc = np.zeros((H, W), np.float32)
    L = df["left"].to_numpy(); T = df["top"].to_numpy()
    Wd = df["width"].to_numpy(); Hd = df["height"].to_numpy()
    if mode == "dwell":
        for l, t, w, h in zip(L, T, Wd, Hd):
            x1, y1 = max(0, int(l)), max(0, int(t))
            x2, y2 = min(W, int(l + w)), min(H, int(t + h))
            if x2 > x1 and y2 > y1:
                acc[y1:y2, x1:x2] += 1.0
        sigma = 5
    elif mode == "visit":
        fx = np.clip((L + Wd / 2).astype(int), 0, W - 1)
        fy = np.clip((T + Hd).astype(int), 0, H - 1)
        cx, cy = fx // visit_cell, fy // visit_cell
        ids = df["local_track_id"].to_numpy()
        uniq = pd.DataFrame({"t": ids, "cx": cx, "cy": cy}).drop_duplicates()
        for cxx, cyy in zip(uniq["cx"].to_numpy(), uniq["cy"].to_numpy()):
            x, y = cxx * visit_cell, cyy * visit_cell
            acc[y:min(H, y + visit_cell), x:min(W, x + visit_cell)] += 1.0
        sigma = 6
    else:  # foot
        fx = np.clip((L + Wd / 2).astype(int), 0, W - 1)
        fy = np.clip((T + Hd).astype(int), 0, H - 1)
        np.add.at(acc, (fy, fx), 1.0)
        sigma = 6
    return gaussian_filter(acc, sigma)


def _overlay(bg: np.ndarray, grid: np.ndarray, alpha_max=0.7, gamma=0.45) -> np.ndarray:
    """Blend a density grid over bg. gamma<1 lifts low densities so EVERY person
    shows (not just the 2-3 brightest dwell spots that otherwise saturate)."""
    h, w = bg.shape[:2]
    g = grid / grid.max() if grid.max() > 0 else grid
    g = np.power(g, gamma)
    gi = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    rgb = JET(gi)[..., :3] * 255.0
    a = np.clip(gi * alpha_max, 0, 1)[..., None]
    return (bg * (1 - a) + rgb * a).astype(np.uint8)


def _save(img: np.ndarray, path: Path, title=None):
    im = Image.fromarray(img)
    im.save(path)


def _cam_files(export_dir: Path, cams):
    out = {}
    for f in sorted(glob.glob(str(export_dir / "cam_*_predictions.csv"))):
        c = int(re.search(r"cam_(\d+)_", Path(f).name).group(1))
        if cams is None or c in cams:
            out[c] = pd.read_csv(f)
    return out


def _video_for(video_dir: Path, local_idx: int) -> Path:
    vids = sorted(video_dir.glob("cam*.mp4"))
    return vids[local_idx] if local_idx < len(vids) else vids[0]


# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--cams", nargs="+", type=int, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pred-w", type=float, default=640.0)
    ap.add_argument("--pred-h", type=float, default=360.0)
    ap.add_argument("--video-steps", type=int, default=120, help="frames in each time-lapse")
    ap.add_argument("--video-fps", type=int, default=15)
    ap.add_argument("--assign-csv", type=Path, default=None,
                    help="live_buffered _eval_assign.csv for clean buffered IDs in the top-down")
    ap.add_argument("--track-frames", type=int, default=600, help="frames in the top-down tracking video")
    ap.add_argument("--cam-tracking", action="store_true",
                    help="also render a camera-view tracking video with buffered IDs "
                         "(boxes+ID+trail on real footage; needs --assign-csv)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    W, H = int(args.pred_w), int(args.pred_h)

    preds = _cam_files(args.export_dir, args.cams)
    if not preds:
        raise SystemExit(f"no cam_*_predictions.csv in {args.export_dir}")
    cam_order = sorted(preds)

    # ---- 1. per-camera heatmaps over background (foot / dwell / visit) ----
    bgs = {}
    for j, c in enumerate(cam_order):
        df = preds[c]
        bg = _extract_bg(_video_for(args.video_dir, j), W, H)
        bgs[c] = bg
        modes = {
            "foot": accumulate(df, W, H, "foot"),
            "dwell": accumulate(df, W, H, "dwell"),
            "visit": accumulate(df, W, H, "visit"),
        }
        for name, g in modes.items():
            _save(_overlay(bg, g), args.out_dir / f"cam_{c}_{name}.png")
        print(f"[viz] cam {c}: foot/dwell/visit over background ({len(df)} dets)")

    # ---- camera-view tracking video with buffered IDs (no calibration needed) ----
    if args.cam_tracking:
        gid_of_cam = {}
        if args.assign_csv and args.assign_csv.exists():
            am = pd.read_csv(args.assign_csv)
            gid_of_cam = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
                          for r in am.itertuples()}
        if _cam_tracking_video(preds, cam_order, gid_of_cam, args.video_dir, W, H,
                               args.video_fps, args.track_frames,
                               args.out_dir / "cam_tracking.mp4", args.video_dir.name):
            print(f"[viz] cam_tracking.mp4 (buffered IDs, {len(cam_order)} cams)")

    # ---- BEV projection (needs calibration) ----
    geo = world = None
    if args.calib and args.calib.exists():
        import json
        calib = json.loads(args.calib.read_text())
        geo = GroundPlaneGeometry(calib)
        calib_ids = sorted(c["CameraId"] for c in calib.get("Cameras", []))
        # buffered IDs (clean, ~7-10 people) from live_buffered, if provided
        gid_of = {}
        if args.assign_csv and args.assign_csv.exists():
            am = pd.read_csv(args.assign_csv)
            gid_of = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
                      for r in am.itertuples()}
        # Per (gid, frame) keep ONLY the nearest-camera estimate (largest bbox =
        # closest = most accurate foot->world). Far cameras give grazing rays that
        # project to wildly wrong far distances; averaging them ruins the position.
        best = {}            # (gid, frame) -> (X, Y, area)
        for j, c in enumerate(cam_order):
            cid = calib_ids[j] if j < len(calib_ids) else None
            if cid is None or not geo.has_camera(cid):
                continue
            df = preds[c]
            u, v = _foot(df)
            area = (df["width"] * df["height"]).to_numpy()
            gcol = df["global_id"].to_numpy(); fr = df["frame_no_cam"].to_numpy()
            ltid = df["local_track_id"].to_numpy()
            for uu, vv, g, f, lt, ar in zip(u, v, gcol, fr, ltid, area):
                wxy = geo.foot_to_world(cid, float(uu), float(vv))
                if wxy is None:
                    continue
                if gid_of:
                    key = gid_of.get((c, int(f), int(lt)), -1)
                    if key < 0:
                        continue
                else:
                    key = int(g) if g and g > 0 else (c * 100000 + int(lt))
                X, Y = wxy[0] / 1000.0, wxy[1] / 1000.0
                k = (key, int(f))
                if k not in best or ar > best[k][2]:
                    best[k] = (X, Y, float(ar))
        # Reject far-field projection outliers: keep the robust room footprint
        # (median +/- IQR). This both cleans the map and makes its aspect match
        # the real room (no 19 m phantom depth from grazing rays).
        bx = np.array([v[0] for v in best.values()]); by = np.array([v[1] for v in best.values()])

        def _rb(a, k=1.6):
            q1, q3 = np.percentile(a, [25, 75]); iqr = max(q3 - q1, 1e-3)
            return max(float(a.min()), q1 - k * iqr), min(float(a.max()), q3 + k * iqr)
        rxlo, rxhi = _rb(bx); rylo, ryhi = _rb(by)
        raw = {}
        for (key, f), (X, Y, _ar) in best.items():
            if rxlo <= X <= rxhi and rylo <= Y <= ryhi:
                raw.setdefault(key, []).append((f, X, Y))
        # Temporal smoothing per track (median kills projection spikes, mean smooths)
        # so markers move smoothly and trails are clean instead of bouncing.
        world, per_frame, allX, allY = {}, {}, [], []
        for key, recs in raw.items():
            recs.sort()
            fr = np.array([r[0] for r in recs]); X = np.array([r[1] for r in recs]); Y = np.array([r[2] for r in recs])
            if len(X) >= 5:
                msz = min(9, len(X) if len(X) % 2 else len(X) - 1)
                X = median_filter(X, size=msz); Y = median_filter(Y, size=msz)
                X = uniform_filter1d(X, size=5); Y = uniform_filter1d(Y, size=5)
            world[key] = list(zip(fr.tolist(), X.tolist(), Y.tolist()))
            for f, x, y in zip(fr.tolist(), X.tolist(), Y.tolist()):
                per_frame.setdefault(int(f), {})[key] = (float(x), float(y))
                allX.append(float(x)); allY.append(float(y))
        if allX:
            # Footprint = the area people actually occupied, at TRUE metric aspect
            # (X vs Y metres). MMPTracking has no wall geometry, and camera-FOV
            # projection runs to the horizon (no walls to stop it), so the walked
            # area + a small margin is the most honest "room" we can show.
            xlo, xhi = np.percentile(allX, [2, 98]); ylo, yhi = np.percentile(allY, [2, 98])
            mx = 0.12 * (xhi - xlo); my = 0.12 * (yhi - ylo)
            xlo, xhi, ylo, yhi = xlo - mx, xhi + mx, ylo - my, yhi + my
            BW, BH = _bev_dims(xlo, xhi, ylo, yhi)
            b = (xlo, xhi, ylo, yhi, BW, BH)
            label = args.video_dir.name
            # ---- 2a. BEV trajectory: clean floor map, paths split at teleport jumps ----
            traj = _bev_base(b)
            d = ImageDraw.Draw(traj, "RGBA")
            keys = [k for k, p in world.items() if len(p) >= 15]
            drawn = 0
            for i, key in enumerate(sorted(keys)):
                pts = sorted(world[key])
                fr = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
                inb = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
                jump = np.hypot(np.diff(xs), np.diff(ys)) > 1.5
                gap = np.diff(fr) > 20
                cut = np.where(jump | gap | ~inb[1:] | ~inb[:-1])[0] + 1
                col = _palette(i, len(keys))
                px, py = _w2px(xs, ys, b)
                for seg in np.split(np.arange(len(xs)), cut):
                    if len(seg) >= 5:
                        d.line([(px[k], py[k]) for k in seg[::2]], fill=col + (190,), width=2)
                        drawn += 1
            _caption(traj, f"{label}  -  top-down trajectories ({drawn} paths)")
            traj.save(args.out_dir / "bev_trajectory.png")
            # ---- 2b. BEV heatmap: clean floor map ----
            px, py = _w2px(allX, allY, b)
            g = np.zeros((BH, BW), float)
            np.add.at(g, (py.astype(int).clip(0, BH - 1), px.astype(int).clip(0, BW - 1)), 1.0)
            g = gaussian_filter(g, 4.0)
            heat = Image.fromarray(_overlay(np.asarray(_bev_base(b)), g, alpha_max=0.9))
            _caption(heat, f"{label}  -  top-down occupancy")
            heat.save(args.out_dir / "bev_heatmap.png")
            print(f"[viz] BEV: trajectory + heatmap ({len(world)} tracks, "
                  f"{len(allX)} points, extent X[{xlo:.1f},{xhi:.1f}] Y[{ylo:.1f},{yhi:.1f}] m)")
            # ---- 2c. animated top-down tracking (AI-City style) ----
            if _bev_tracking_video(per_frame, b, args.out_dir / "bev_tracking.mp4",
                                   args.video_fps, args.track_frames, label):
                print(f"[viz] bev_tracking.mp4 ({len(gid_of) and 'buffered' or 'greedy'} IDs)")

    # ---- 3. time-lapse videos (accumulating heatmap over time) ----
    maxf = max(int(df["frame_no_cam"].max()) for df in preds.values())
    steps = args.video_steps
    edges = np.linspace(0, maxf, steps + 1)

    # 3a. per-camera time-lapse (tiled, over background)
    cols = int(np.ceil(np.sqrt(len(cam_order)))); rows = int(np.ceil(len(cam_order) / cols))
    tmpd = Path(tempfile.mkdtemp(prefix="tlcam_"))
    accU = {c: ([], []) for c in cam_order}
    cam_dfs = {c: preds[c].sort_values("frame_no_cam") for c in cam_order}
    for s in range(steps):
        hi = edges[s + 1]
        tiles = []
        for c in cam_order:
            df = cam_dfs[c]; sub = df[df["frame_no_cam"] <= hi]
            g = accumulate(sub, W, H, "foot") if len(sub) else np.zeros((H, W))
            tiles.append(_overlay(bgs[c], g))
        canvas = np.zeros((rows * H, cols * W, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, cc = divmod(i, cols); canvas[r*H:(r+1)*H, cc*W:(cc+1)*W] = t
        Image.fromarray(canvas).save(tmpd / f"f{s:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.video_fps), "-i", str(tmpd / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out_dir / "timelapse_cam.mp4")],
                   capture_output=True)
    print(f"[viz] timelapse_cam.mp4 ({steps} frames)")

    # 3b. BEV time-lapse
    if geo is not None and world:
        # rebuild per-frame world points (flatten)
        wf = []  # (frame, X, Y)
        for pts in world.values():
            wf.extend(pts)
        wf = np.array(sorted(wf))  # (N,3) frame,X,Y
        xlo, xhi = np.percentile(wf[:, 1], [2, 98]); ylo, yhi = np.percentile(wf[:, 2], [2, 98])
        BW, BH = _bev_dims(xlo, xhi, ylo, yhi); b = (xlo, xhi, ylo, yhi, BW, BH)
        base = np.asarray(_bev_base(b))
        pxa, pya = _w2px(wf[:, 1], wf[:, 2], b)
        pxa = pxa.astype(int).clip(0, BW - 1); pya = pya.astype(int).clip(0, BH - 1)
        tmpb = Path(tempfile.mkdtemp(prefix="tlbev_"))
        for s in range(steps):
            m = wf[:, 0] <= edges[s + 1]
            g = np.zeros((BH, BW), float); np.add.at(g, (pya[m], pxa[m]), 1.0)
            g = gaussian_filter(g, 4.0)
            frame = Image.fromarray(_overlay(base, g, alpha_max=0.9))
            _caption(frame, args.video_dir.name + "  -  top-down (time-lapse)")
            frame.save(tmpb / f"f{s:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.video_fps), "-i", str(tmpb / "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out_dir / "timelapse_bev.mp4")],
                       capture_output=True)
        print(f"[viz] timelapse_bev.mp4 ({steps} frames)")
    print(f"[viz] DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
