#!/usr/bin/env python3
"""MTMC density heatmaps (occupancy / footfall / dwell) on the real top-down floor map
+ per-camera, matching the webui HeatmapView asset naming:
  <out>/bev_{occupancy,footfall,dwelltime}.png  and  cam_<N>_{...}.png

occupancy = presence (detection-frames) per cell; footfall = distinct global ids per
cell; dwelltime = occupancy / footfall (avg frames a person lingers). World positions
come from the calibration back-projection (BEV) or bbox foot points (per-camera).
"""
import argparse, importlib.util
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration
import json


def colorize(acc, base_bgr, alpha_max=0.85, gamma=0.45, title=None):
    """Match the MMP demo heatmap look (venv_visualize._overlay): JET with a gamma<1
    lift so EVERY density level shows, and a CONTINUOUS alpha (no hard mask) that fades
    smoothly into the background — not isolated patches. The floor map is dimmed so the
    heat reads like MMP's dark canvas while keeping the real warehouse structure."""
    g = cv2.GaussianBlur(acc.astype(np.float32), (0, 0), sigmaX=max(2, base_bgr.shape[1] // 200))
    if g.max() > 0:
        g = np.power(g / g.max(), gamma)
    g = np.clip(g, 0, 1)
    cm = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    a = (g * alpha_max)[..., None]
    bg = base_bgr.astype(np.float32)                # keep the REAL floor map at full brightness
    out = (bg * (1 - a) + cm * a).astype(np.uint8)
    if title:                                       # dark strip behind title for readability on bright map
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(out, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def save_set(out_dir, prefix, occ_display, foot_occ, foot_ids, base_bgr, label=""):
    """occ_display = the occupancy density to render (FULL-BOX area for per-camera, like MMP;
    foot-point→world for BEV). foot_occ = foot-point occupancy used for the dwell ratio
    (dwell = foot_occ / footfall, MMP's definition). foot_ids = distinct-id sets per cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    foot = np.array([[len(s) for s in row] for row in foot_ids], np.float32)
    dwell = np.where(foot > 0, foot_occ / np.maximum(foot, 1), 0)
    titles = {"occupancy": "top-down Occupancy (area density)",
              "footfall": "Footfall (distinct people)", "dwelltime": "Dwell time (occupancy / footfall)"}
    for name, acc in [("occupancy", occ_display), ("footfall", foot), ("dwelltime", dwell)]:
        img = colorize(cv2.resize(acc, (base_bgr.shape[1], base_bgr.shape[0])), base_bgr,
                       title=f"{label} - {titles[name]}" if label else titles[name])
        cv2.imwrite(str(out_dir / f"{prefix}_{name}.png"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/MTMC_Tracking_2026/val")
    ap.add_argument("--warehouse", default="Warehouse_022")
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frame", type=int, default=1799)
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--cell", type=int, default=12, help="map heatmap cell size (px)")
    args = ap.parse_args()

    whdir = Path(args.root) / args.warehouse
    cal = WarehouseCalibration(whdir / "calibration.json")
    calj = json.load(open(whdir / "calibration.json")); s0 = calj["sensors"][0]
    sf = s0["scaleFactor"]; tx = s0["translationToGlobalCoordinates"]["x"]; ty = s0["translationToGlobalCoordinates"]["y"]
    mp = cv2.imread(str(whdir / "map.png")); MH, MW = mp.shape[:2]
    if args.sources:
        src = [l.strip() for l in open(args.sources) if l.strip() and not l.strip().startswith("#")]
        cam_calib = {i: int(p.split("Camera_")[-1].split(".")[0]) for i, p in enumerate(src)}
    else:
        cam_calib = {}

    a = pd.read_csv(args.assign)
    gid_of = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id) for r in a.itertuples()}

    cell = args.cell
    GW, GH = MW // cell, MH // cell
    bev_occ = np.zeros((GH, GW), np.float32)          # BEV occupancy = foot->world (MMP BEV is foot-based)
    bev_ids = [[set() for _ in range(GW)] for _ in range(GH)]
    # per-camera image-space accumulators (MMP definitions):
    cam_box, cam_foot, cam_ids = {}, {}, {}           # box=full-bbox occupancy, foot=foot pt, ids=distinct
    CW, CH = 1920 // cell, 1080 // cell

    for f in sorted(Path(args.export_dir).glob("cam_*_predictions.csv")):
        ec = int(f.stem.split("_")[1]); cc = cam_calib.get(ec, ec)
        d = pd.read_csv(f)
        if ec not in cam_box:
            cam_box[ec] = np.zeros((CH, CW), np.float32); cam_foot[ec] = np.zeros((CH, CW), np.float32)
            cam_ids[ec] = [[set() for _ in range(CW)] for _ in range(CH)]
        for r in d.itertuples():
            fr = int(r.frame_no_cam)
            if fr > args.max_frame:
                continue
            gid = gid_of.get((ec, fr, int(r.local_track_id)), -1)
            # occupancy = FULL BOUNDING BOX area (MMP 'dwell' accumulation -> broad body coverage)
            x1 = max(0, int(r.left // cell)); y1 = max(0, int(r.top // cell))
            x2 = min(CW, int((r.left + r.width) // cell) + 1); y2 = min(CH, int((r.top + r.height) // cell) + 1)
            if x2 > x1 and y2 > y1:
                cam_box[ec][y1:y2, x1:x2] += 1
            # foot point: for the dwell ratio + distinct-id footfall
            fu, fv = r.left + r.width / 2.0, r.top + r.height
            ix, iy = int(fu // cell), int(fv // cell)
            if 0 <= ix < CW and 0 <= iy < CH:
                cam_foot[ec][iy, ix] += 1
                if gid >= 0: cam_ids[ec][iy][ix].add(gid)
            # BEV world-space (foot -> world; a box has no world rectangle)
            w = cal.foot_to_world(cc, fu, fv)
            if w is None: continue
            px = (w[0] + tx) * sf; py = (w[1] + ty) * sf
            if args.flip_y: py = MH - py
            gx, gy = int(px // cell), int(py // cell)
            if 0 <= gx < GW and 0 <= gy < GH:
                bev_occ[gy, gx] += 1
                if gid >= 0: bev_ids[gy][gx].add(gid)

    out = Path(args.out_dir)
    save_set(out, "bev", bev_occ, bev_occ, bev_ids, mp, label=args.warehouse)
    # bev_heatmap.png (alias of occupancy, matches MMP folder) + bev_trajectory.png (world paths)
    cv2.imwrite(str(out / "bev_heatmap.png"), colorize(cv2.resize(bev_occ, (MW, MH)), mp))
    traj = mp.copy()
    gid_path = defaultdict(list)
    for f in sorted(Path(args.export_dir).glob("cam_*_predictions.csv")):
        ec = int(f.stem.split("_")[1]); cc = cam_calib.get(ec, ec)
        d = pd.read_csv(f); d = d[d.frame_no_cam <= args.max_frame] if "frame_no_cam" in d else d
        for r in d.itertuples():
            gid = gid_of.get((ec, int(r.frame_no_cam), int(r.local_track_id)), -1)
            if gid < 0:
                continue
            w = cal.foot_to_world(cc, r.left + r.width / 2.0, r.top + r.height)
            if w is None:
                continue
            px = (w[0] + tx) * sf; py = (w[1] + ty) * sf
            if args.flip_y:
                py = MH - py
            gid_path[gid].append((int(px), int(py), int(r.frame_no_cam)))
    for gid, pts in gid_path.items():
        pts.sort(key=lambda x: x[2]); col = tuple(int(v) for v in
            cv2.cvtColor(np.uint8([[[(gid * 47) % 180, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        for k in range(1, len(pts)):
            if abs(pts[k][2] - pts[k - 1][2]) <= 15:
                cv2.line(traj, pts[k - 1][:2], pts[k][:2], col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "bev_trajectory.png"), traj)
    for ec in sorted(cam_box):
        base = cv2.imread(str(whdir / "videos" / f"Camera_{cam_calib.get(ec, ec):04d}.mp4")) if False else \
               cv2.resize(np.full((1080, 1920, 3), 30, np.uint8), (1920, 1080))
        # use a real frame as the per-camera backdrop
        cap = cv2.VideoCapture((src[ec] if args.sources else str(whdir / "videos" / f"Camera_{ec:04d}.mp4")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(args.max_frame, 900)); ok, frm = cap.read(); cap.release()
        if ok: base = frm
        save_set(out, f"cam_{ec}", cam_box[ec], cam_foot[ec], cam_ids[ec], base, label=f"{args.warehouse} cam{ec}")
    print(f"[heatmap] wrote bev + {len(cam_box)} per-camera x3 metrics -> {out}")


if __name__ == "__main__":
    main()
