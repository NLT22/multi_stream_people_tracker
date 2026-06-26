"""Characterize retail false-positive detections to find a no-training filter.

For each retail scene/camera: match predicted boxes to clean GT per frame (IoU>=0.5).
Unmatched predictions = false positives. Group FPs by local_track_id and ask:
  - Are FP tracks STATIC (mannequins/posters)?  -> center motion over lifetime
  - Are they long-lived or transient?            -> num frames
  - Where are they / how big?                     -> mean pos + size
Compare against TRUE-positive (GT-matched) tracks so a filter can separate them.
"""
from __future__ import annotations
import glob, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

REPO = "/media/pc/c88ba509-53f0-4c97-9e44-e33483754b08/multi_stream_people_tracker"
EXPORT = REPO + "/output/eval/full_mmp_val"
VAL = REPO + "/dataset/MMPTracking_10minute/val"


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, 0][:, None], b[:, 0][None]); iy1 = np.maximum(a[:, 1][:, None], b[:, 1][None])
    ix2 = np.minimum(ax2[:, None], bx2[None]); iy2 = np.minimum(ay2[:, None], by2[None])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]; area_b = (b[:, 2] * b[:, 3])[None]
    return inter / (area_a + area_b - inter + 1e-9)


def analyze_scene(scene):
    sdir = f"{EXPORT}/{scene}"
    vdir = f"{VAL}/{scene}"
    src_ids = sorted(int(os.path.basename(p).split("_")[1])
                     for p in glob.glob(f"{sdir}/cam_*_predictions.csv"))
    gt_cams = sorted(int(os.path.basename(p)[3:-4].split("_")[0].replace("cam", "") or 0)
                     for p in glob.glob(f"{vdir}/cam*.mp4"))
    gt_cams = sorted(int(os.path.basename(p)[3:].split(".")[0]) for p in glob.glob(f"{vdir}/cam*.mp4"))
    rows = []
    for src, gtc in zip(src_ids, gt_cams):
        pred = pd.read_csv(f"{sdir}/cam_{src}_predictions.csv")
        gtp = f"{vdir}/gt_cam{gtc}_clean.csv"
        if not os.path.exists(gtp):
            gtp = f"{vdir}/gt_cam{gtc}.csv"
        gt = pd.read_csv(gtp)
        # per-frame match
        tp_flag = {}  # (frame,ltid) -> matched?
        for f in pred["frame_no_cam"].unique():
            p = pred[pred["frame_no_cam"] == f]
            g = gt[gt["frame"] == f]
            pb = p[["left", "top", "width", "height"]].values.astype(float)
            gb = g[["left", "top", "width", "height"]].values.astype(float)
            M = iou_matrix(pb, gb)
            matched = (M.max(axis=1) >= 0.5) if len(gb) else np.zeros(len(pb), bool)
            for ltid, m in zip(p["local_track_id"].values, matched):
                tp_flag[(f, int(ltid))] = tp_flag.get((f, int(ltid)), False) or bool(m)
        # per local track stats
        for ltid, grp in pred.groupby("local_track_id"):
            cx = (grp["left"] + grp["width"] / 2).values
            cy = (grp["top"] + grp["height"] / 2).values
            nfr = len(grp)
            match_frac = np.mean([tp_flag.get((int(f), int(ltid)), False)
                                  for f in grp["frame_no_cam"].values])
            motion = float(np.hypot(cx.std(), cy.std()))  # px spread of center
            span = float(np.hypot(cx.max() - cx.min(), cy.max() - cy.min()))
            rows.append({"scene": scene, "cam": src, "ltid": int(ltid),
                         "nframes": nfr, "match_frac": match_frac,
                         "motion_std": motion, "span": span,
                         "mean_h": float(grp["height"].mean())})
    return rows


def main():
    scenes = sorted(os.path.basename(p) for p in glob.glob(f"{EXPORT}/64pm_retail_*"))
    allrows = []
    for s in scenes:
        allrows += analyze_scene(s)
    df = pd.DataFrame(allrows)
    # Classify tracks: TP-track (mostly matches GT) vs FP-track (rarely matches)
    df["is_fp"] = df["match_frac"] < 0.2
    fp, tp = df[df.is_fp], df[~df.is_fp]
    print(f"retail tracks: {len(df)}  FP-tracks(match<0.2)={len(fp)} ({100*len(fp)/len(df):.0f}%)  TP-tracks={len(tp)}")
    print(f"\n{'group':12}{'n':>6}{'med_nframes':>12}{'med_motion_std':>15}{'med_span':>10}{'med_h':>8}")
    for nm, g in [("FP", fp), ("TP", tp)]:
        print(f"{nm:12}{len(g):>6}{g.nframes.median():>12.0f}"
              f"{g.motion_std.median():>15.1f}{g.span.median():>10.1f}{g.mean_h.median():>8.1f}")
    # How many detections (not tracks) come from FP tracks? (that's what hurts precision)
    fp_dets = fp.nframes.sum(); tp_dets = tp.nframes.sum()
    print(f"\ndetections: FP={fp_dets} ({100*fp_dets/(fp_dets+tp_dets):.0f}%)  TP={tp_dets}")
    # Static-filter potential: FP tracks that are static & long-lived
    for mstd in [5, 10, 15, 20]:
        for minfr in [50, 100, 200]:
            staticish = df[(df.motion_std < mstd) & (df.nframes >= minfr)]
            sf = staticish[staticish.is_fp]; st = staticish[~staticish.is_fp]
            if len(staticish):
                print(f"  filter motion_std<{mstd},nframes>={minfr}: removes {len(staticish)} tracks "
                      f"({sf.nframes.sum()} FP-dets, {st.nframes.sum()} TP-dets) "
                      f"-> {'GOOD' if sf.nframes.sum() > 3*max(1,st.nframes.sum()) else 'risky'}")
    df.to_csv("/tmp/retail_track_stats.csv", index=False)
    print("\nsaved /tmp/retail_track_stats.csv")


if __name__ == "__main__":
    main()
