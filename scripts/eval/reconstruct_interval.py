#!/usr/bin/env python3
"""Offline reconstruction of a detector-interval export to recover IDF1.

When the detector runs every (interval+1) frames, the tracker fills the skipped
frames with predicted boxes (which can drift) and the SGIE embeds those drifted
crops. Because global IDs are computed OFFLINE (live_buffered), we can repair the
export after capture:
  --interp-boxes : per (cam, track) linearly interpolate left/top/width/height on
                   non-detector frames between consecutive detector frames.
  --filter-emb   : keep only detector-frame embeddings in the chunks, so the
                   clustering uses clean crops (re-run live_buffered on the output).

Detector frames are taken as frame_no % (interval+1) == offset.

  python scripts/eval/reconstruct_interval.py --src <export> --dst <out> \
      --interval 2 --interp-boxes --filter-emb
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _interp_cam(df: pd.DataFrame, step: int, offset: int) -> pd.DataFrame:
    out = []
    for _tid, g in df.groupby("local_track_id"):
        g = g.sort_values("frame_no_cam").copy()
        fr = g["frame_no_cam"].to_numpy()
        det = (fr % step) == offset
        if det.sum() >= 2:
            dfr = fr[det]
            lo, hi = dfr.min(), dfr.max()
            inside = (fr >= lo) & (fr <= hi)
            for col in ("left", "top", "width", "height"):
                interp = np.interp(fr, dfr, g.loc[det, col].to_numpy())
                g[col] = np.where(inside, interp, g[col].to_numpy())
        out.append(g)
    return pd.concat(out).sort_values(["frame_no_cam", "local_track_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--interval", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0, help="detector-frame phase (frame %% step == offset)")
    ap.add_argument("--interp-boxes", action="store_true")
    ap.add_argument("--filter-emb", action="store_true")
    args = ap.parse_args()
    step = args.interval + 1
    args.dst.mkdir(parents=True, exist_ok=True)

    n_box = 0
    for csv in sorted(glob.glob(str(args.src / "cam_*_predictions.csv"))):
        df = pd.read_csv(csv)
        if args.interp_boxes:
            df = _interp_cam(df, step, args.offset)
            n_box += 1
        df.to_csv(args.dst / Path(csv).name, index=False)

    n_emb_in = n_emb_out = 0
    for npz in sorted(glob.glob(str(args.src / "det_emb_chunk_*.npz"))):
        d = np.load(npz)
        if args.filter_emb:
            mask = (d["frame_no"] % step) == args.offset
            n_emb_in += len(d["frame_no"]); n_emb_out += int(mask.sum())
            np.savez(args.dst / Path(npz).name,
                     cam_id=d["cam_id"][mask], frame_no=d["frame_no"][mask],
                     local_track_id=d["local_track_id"][mask],
                     embeddings=d["embeddings"][mask])
        else:
            shutil.copy(npz, args.dst / Path(npz).name)

    print(f"[recon] step={step} offset={args.offset} interp_boxes={args.interp_boxes} "
          f"filter_emb={args.filter_emb}: {n_box} cam csvs"
          + (f", emb {n_emb_out}/{n_emb_in} kept" if args.filter_emb else ""))


if __name__ == "__main__":
    main()
