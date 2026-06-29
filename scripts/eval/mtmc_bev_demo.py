#!/usr/bin/env python3
"""MTMC split demo: one camera's OSD (left) | bird's-eye floor map (right), synced.

The MTMC differentiator MMP cannot show: a metric top-down view. Left panel is a real
camera with detection boxes + the geometry global-linker's stable cross-camera IDs;
right panel is the warehouse floor (`map.png`) with every person plotted at their
back-projected world position + a fading trail, the SAME ID/colour as the camera.

World→map pixel transform (from calibration.json — global, identical across cameras):
    map_px = (world_xy + translationToGlobalCoordinates) * scaleFactor
(Y orientation is verified with --probe-frame before rendering the full clip.)

Usage:
    # validate one composite frame first
    mtmc_bev_demo.py --warehouse Warehouse_022 --cam 2 --assign <gl.csv> --probe-frame 300 --out probe.png
    # then the clip
    mtmc_bev_demo.py --warehouse Warehouse_022 --cam 2 --assign <gl.csv> --out demo.mp4 --max-frame 1799
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration


def color_for(gid: int) -> tuple[int, int, int]:
    h = (gid * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/MTMC_Tracking_2026/val")
    ap.add_argument("--warehouse", default="Warehouse_022")
    ap.add_argument("--export-dir", default="output/eval/mtmc_w022_1280")
    ap.add_argument("--assign", required=True, help="global-linker assign-csv")
    ap.add_argument("--cam", type=int, default=2, help="export cam id for the camera panel")
    ap.add_argument("--mode", choices=["split", "bev", "camera"], default="split",
                    help="split = camera|BEV side-by-side; bev = full-frame top-down map only "
                         "(MMP-style tracking-BEV video, on the real floor map); camera = camera OSD only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frame", type=int, default=1799)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--trail", type=int, default=45, help="BEV trail length (frames)")
    ap.add_argument("--flip-y", action="store_true", help="flip BEV Y (set after probing)")
    ap.add_argument("--probe-frame", type=int, default=-1, help="if >=0 write a single composite PNG and exit")
    ap.add_argument("--gt", action="store_true", help="overlay GT 3d-location as white X on BEV (validation)")
    args = ap.parse_args()

    whdir = Path(args.root) / args.warehouse
    cal = WarehouseCalibration(whdir / "calibration.json")
    import json
    calj = json.load(open(whdir / "calibration.json"))
    s0 = calj["sensors"][0]
    sf = s0["scaleFactor"]; tx = s0["translationToGlobalCoordinates"]["x"]; ty = s0["translationToGlobalCoordinates"]["y"]
    cam_world = {int(s["id"].split("_")[-1]): (s["coordinates"]["x"], s["coordinates"]["y"]) for s in calj["sensors"]}
    mp = cv2.imread(str(whdir / "map.png"))
    MH, MW = mp.shape[:2]

    def to_map(wx, wy):
        px = (wx + tx) * sf; py = (wy + ty) * sf
        if args.flip_y:
            py = MH - py
        return int(round(px)), int(round(py))

    # optional GT 3d locations per frame (validation overlay)
    gt_by_frame = {}
    if args.gt:
        gtj = json.load(open(whdir / "ground_truth.json"))
        for fs, objs in gtj.items():
            gt_by_frame[int(fs)] = [(o["3d location"][0], o["3d location"][1])
                                    for o in objs if o.get("object type") == "Person"]

    # global ids
    a = pd.read_csv(args.assign)
    gid_of = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id) for r in a.itertuples()}

    # per-frame detections per camera: frame -> [(cam, ltid, l,t,w,h, gid)]
    byframe = defaultdict(list)
    for f in sorted(Path(args.export_dir).glob("cam_*_predictions.csv")):
        c = int(f.stem.split("_")[1])
        d = pd.read_csv(f)
        for r in d.itertuples():
            fr = int(r.frame_no_cam)
            if fr > args.max_frame:
                continue
            gid = gid_of.get((c, fr, int(r.local_track_id)))
            if gid is None:
                continue
            byframe[fr].append((c, int(r.local_track_id), r.left, r.top, r.width, r.height, gid))

    # camera video for the camera panel (not needed in bev-only mode)
    need_cam_outer = args.mode in ("split", "camera")
    cap = None
    if need_cam_outer:
        cap = cv2.VideoCapture(str(whdir / "videos" / f"Camera_{args.cam:04d}.mp4"))

    trails = defaultdict(lambda: deque(maxlen=args.trail))  # gid -> deque of (px,py)

    need_cam = args.mode in ("split", "camera")

    def render(frame_idx, cam_img):
        # LEFT: camera OSD (only when the camera panel is shown)
        left = cam_img.copy() if cam_img is not None else None
        if left is not None:
            for (c, t, l, tp, w, h, gid) in byframe.get(frame_idx, []):
                if c != args.cam:
                    continue
                col = color_for(gid)
                cv2.rectangle(left, (int(l), int(tp)), (int(l + w), int(tp + h)), col, 2)
                cv2.putText(left, f"ID{gid}", (int(l), int(tp) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # BEV: one world point per gid this frame (mean over cameras), + trail
        bev = mp.copy()
        for cid, (cx, cy) in cam_world.items():
            mx, my = to_map(cx, cy)
            cv2.drawMarker(bev, (mx, my), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 16, 2)
            cv2.putText(bev, f"C{cid}", (mx + 8, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if args.gt:
            for (gx, gy) in gt_by_frame.get(frame_idx, []):
                mx, my = to_map(gx, gy)
                cv2.drawMarker(bev, (mx, my), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        gid_world = defaultdict(list)
        for (c, t, l, tp, w, h, gid) in byframe.get(frame_idx, []):
            wpt = cal.foot_to_world(c, l + w / 2.0, tp + h)
            if wpt:
                gid_world[gid].append(wpt)
        for gid, pts in gid_world.items():
            wx = float(np.mean([p[0] for p in pts])); wy = float(np.mean([p[1] for p in pts]))
            px, py = to_map(wx, wy); col = color_for(gid)
            trails[gid].append((px, py))
            tl = list(trails[gid])
            for i in range(1, len(tl)):
                cv2.line(bev, tl[i - 1], tl[i], col, 2)
            cv2.circle(bev, (px, py), 7, col, -1)
            cv2.circle(bev, (px, py), 7, (255, 255, 255), 1)
            cv2.putText(bev, f"ID{gid}", (px + 9, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        def banner(img, text, col):
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            return img

        if args.mode == "bev":
            B = bev.copy()
            return banner(B, f"{args.warehouse}  bird's-eye tracking  (frame {frame_idx})", (255, 130, 124))
        OUT_H = 720
        def fit(img):
            h, w = img.shape[:2]; return cv2.resize(img, (int(w * OUT_H / h), OUT_H))
        if args.mode == "camera":
            return banner(fit(left), f"Camera {args.cam}  (frame {frame_idx})", (40, 230, 200))
        L = banner(fit(left), f"Camera {args.cam}  (frame {frame_idx})", (40, 230, 200))
        B = banner(fit(bev), f"{args.warehouse}  bird's-eye (world meters)", (255, 130, 124))
        return np.hstack([L, np.full((OUT_H, 4, 3), 60, np.uint8), B])

    def read_frame(idx):
        if cap is None:
            return True, None
        return cap.read()

    if args.probe_frame >= 0:
        img = None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.probe_frame)
            ok, img = cap.read()
            if not ok:
                print("cannot read probe frame"); return
        cv2.imwrite(args.out, render(args.probe_frame, img))
        print(f"[probe] wrote {args.out} (frame {args.probe_frame}); check alignment / Y orientation")
        if cap:
            cap.release()
        return

    writer = None
    idx = 0
    while idx <= args.max_frame:
        ok, img = read_frame(idx)
        if not ok:
            break
        comp = render(idx, img)
        if writer is None:
            h, w = comp.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(comp)
        idx += 1
    if writer:
        writer.release()
    if cap:
        cap.release()
    print(f"[bev-demo] wrote {args.out} ({idx} frames @ {args.fps} fps, mode={args.mode})")


if __name__ == "__main__":
    main()
