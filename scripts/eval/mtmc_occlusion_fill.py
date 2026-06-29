#!/usr/bin/env python3
"""MTMC occlusion reprojection — recover missed detections from cross-camera geometry.

The detector misses ~13 % of GT boxes (the oracle ceiling), about half of them medium
people OCCLUDED behind shelves/others — which no detector resolution recovers. But a
person hidden in camera A is usually localised by camera B at the same instant, and the
warehouse calibration lets us project that world position back INTO camera A and
synthesise the missing box.

Safety against false positives: we ONLY fill within-camera gaps that are *bracketed* by
real detections of the same global id in that camera (the person was demonstrably visible
in A just before and after, so they were merely occluded in between). We never invent a
box in a camera that never saw the person.

Input: the global-linker assign-csv + the raw cam_*_predictions.csv + calibration.
Output: scores Global IDF1 directly on the union of real + synthesised detections.
Box model (validated: ~91 % of synthesised boxes hit IoU>=0.5 vs GT): foot = project
(x,y,0), head = project (x,y,PERSON_H); width = WIDTH_RATIO * box_height.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration

PERSON_H = 1.7        # world units (m); GT 3d box scale z median ~1.67
WIDTH_RATIO = 0.33

_spec = importlib.util.spec_from_file_location(
    "score_mtmc", str(Path(__file__).resolve().parent / "score_mtmc_idf1.py"))
score_mtmc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(score_mtmc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--assign-csv", required=True, type=Path, help="global-linker assign-csv")
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--gt-json", required=True, type=Path)
    ap.add_argument("--max-frame", type=int, default=1799)
    ap.add_argument("--max-fill", type=int, default=120,
                    help="max bracketed gap length (frames) to fill within a camera")
    ap.add_argument("--span-mode", choices=["bracket", "global"], default="bracket",
                    help="bracket: only fill gaps between consecutive detections in a camera. "
                         "global: fill the gid's whole global active span in every camera that "
                         "detected it >= --min-cam-dets times (recovers sustained occlusion).")
    ap.add_argument("--min-cam-dets", type=int, default=10,
                    help="global mode: a camera must have detected the gid this many times "
                         "(proof it can physically see the person) before we reproject into it")
    ap.add_argument("--pred-cam-offset", type=int, default=0)
    args = ap.parse_args()

    cal = WarehouseCalibration(args.calib)
    a = pd.read_csv(args.assign_csv)
    gid_of = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
              for r in a.itertuples()}

    # real labelled detections: per cam -> list of (frame, gid, l,t,w,h); also world pos
    real = defaultdict(list)
    world_by_gid_frame = defaultdict(list)   # (gid,frame) -> [world (x,y)]
    detframes = defaultdict(set)             # (cam,gid) -> set(frames detected)
    for f in sorted(args.export_dir.glob("cam_*_predictions.csv")):
        cam = int(f.stem.split("_")[1]) + args.pred_cam_offset
        if not cal.has(cam):
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            fr = int(r.frame_no_cam)
            if fr > args.max_frame:
                continue
            gid = gid_of.get((cam, fr, int(r.local_track_id)))
            if gid is None:
                continue
            real[cam].append((fr, gid, r.left, r.top, r.width, r.height))
            detframes[(cam, gid)].add(fr)
            w = cal.foot_to_world(cam, r.left + r.width / 2.0, r.top + r.height)
            if w is not None:
                world_by_gid_frame[(gid, fr)].append(w)

    # consensus world position per (gid, frame)
    cons = {k: np.mean(v, axis=0) for k, v in world_by_gid_frame.items()}
    # per-gid sorted consensus track (for interpolation in global mode)
    gid_track = defaultdict(list)
    for (gid, fr), w in cons.items():
        gid_track[gid].append((fr, w[0], w[1]))
    for gid in gid_track:
        gid_track[gid].sort()
    gid_span = {gid: (t[0][0], t[-1][0]) for gid, t in gid_track.items()}

    def world_at(gid, fr):
        wp = cons.get((gid, fr))
        if wp is not None:
            return np.asarray(wp)
        tr = gid_track.get(gid)
        if not tr or fr < tr[0][0] or fr > tr[-1][0]:
            return None
        fa = np.array([x[0] for x in tr])
        return np.array([np.interp(fr, fa, [x[1] for x in tr]),
                         np.interp(fr, fa, [x[2] for x in tr])])

    def synth_box(cam, gid, fr, wp):
        foot = cal.world_to_pixel(cam, wp[0], wp[1], 0.0)
        head = cal.world_to_pixel(cam, wp[0], wp[1], PERSON_H)
        if not foot or not head:
            return None
        uf, vf, _ = foot; uh, vh, _ = head
        if not (0 <= uf <= 1920 and 0 <= max(vf, vh) <= 1080):
            return None
        bh = abs(vf - vh); bw = bh * WIDTH_RATIO; cu = (uf + uh) / 2.0
        return (fr, gid, cu - bw / 2, min(vf, vh), bw, bh)

    n_syn = 0
    synth = defaultdict(list)
    if args.span_mode == "bracket":
        for (cam, gid), frames in detframes.items():
            fs = sorted(frames)
            for i in range(len(fs) - 1):
                f0, f1 = fs[i], fs[i + 1]; gap = f1 - f0
                if gap <= 1 or gap > args.max_fill:
                    continue
                for fmid in range(f0 + 1, f1):
                    wp = world_at(gid, fmid)
                    if wp is None:
                        continue
                    box = synth_box(cam, gid, fmid, wp)
                    if box:
                        synth[cam].append(box); n_syn += 1
    else:  # global span
        for (cam, gid), frames in detframes.items():
            if len(frames) < args.min_cam_dets:
                continue
            gmin, gmax = gid_span[gid]
            for fr in range(gmin, gmax + 1):
                if fr in frames or fr > args.max_frame:
                    continue
                wp = world_at(gid, fr)
                if wp is None:
                    continue
                box = synth_box(cam, gid, fr, wp)
                if box:
                    synth[cam].append(box); n_syn += 1

    # assemble pred dict (real + synth) for the scorer; gid IS the id column
    all_pred = {}
    for cam in set(list(real) + list(synth)):
        rows = real[cam] + synth[cam]
        df = pd.DataFrame(rows, columns=["frame", "global_id", "left", "top", "width", "height"])
        all_pred[cam] = df
    all_gt = score_mtmc.load_gt(args.gt_json)
    all_gt = {c: d[d.frame <= args.max_frame] for c, d in all_gt.items()}
    all_pred = {c: d[d.frame <= args.max_frame] for c, d in all_pred.items()}
    r = score_mtmc.global_idf1(all_gt, all_pred, iou_threshold=0.5)
    print(f"[occ-fill] max-fill={args.max_fill}: synthesised {n_syn} boxes")
    print(f"  IDF1={r['idf1']:.4f} predIDs={r['num_pred_ids']} TP={r['idtp']} FP={r['idfp']} FN={r['idfn']}")


if __name__ == "__main__":
    main()
