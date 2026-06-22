#!/usr/bin/env python3
"""Honest per-scene IDF1 from a long_run export, using the buffered per-detection
assignments (_eval_assign.csv from src.mtmc.live_buffered --groups).

For each env group: remap its global cam-ids to scene-local, set each detection's
global_id from the buffered assignment (fallback -1), and run metrics_mmp against
that scene's GT. Scores FULL GT by default (honest, untrimmed); --processed-only
restricts GT to frames the pipeline actually produced predictions for (reproduces
the optimistic 'processed-segment' number).

  python scripts/eval/score_longrun_idf1.py --export-dir output/eval/long_run \
    --map "cafe=64pm_cafe_shop_0:0-3,lobby=64pm_lobby_0:4-7,office=64pm_office_0:8-11,\
industry=64pm_industry_safety_0:12-15,retail=64pm_retail_0:16-19" \
    --val-root dataset/MMPTracking_10minute/val
"""
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd


def _expand(rng):
    out = []
    for seg in rng.split("+"):
        if "-" in seg:
            a, b = seg.split("-"); out += list(range(int(a), int(b) + 1))
        elif seg:
            out.append(int(seg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--map", required=True, help="group=scene:cams,...")
    ap.add_argument("--val-root", default="dataset/MMPTracking_10minute/val")
    ap.add_argument("--processed-only", action="store_true",
                    help="restrict GT to frames with predictions (optimistic processed-segment)")
    args = ap.parse_args()

    am = pd.read_csv(args.export_dir / "_eval_assign.csv")
    gid_of = {(r.group, int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
              for r in am.itertuples()}

    entries = []
    for part in args.map.split(","):
        grp, rest = part.split("="); scene, rng = rest.split(":")
        entries.append((grp.strip(), scene.strip(), _expand(rng)))

    results = {}
    for grp, scene, gcams in entries:
        local = {g: j for j, g in enumerate(gcams)}
        tmp = Path(tempfile.mkdtemp(prefix=f"score_{scene}_"))
        for g in gcams:
            src = args.export_dir / f"cam_{g}_predictions.csv"
            if not src.exists():
                continue
            df = pd.read_csv(src)
            df["global_id"] = [gid_of.get((grp, g, int(f), int(t)), -1)
                               for f, t in zip(df["frame_no_cam"], df["local_track_id"])]
            df["cam_id"] = local[g]
            if args.processed_only:
                df = df[df["global_id"] >= 0]
            df.to_csv(tmp / f"cam_{local[g]}_predictions.csv", index=False)
        gt = ["--gt-suffix", "_clean"] if "retail" in scene else []
        r = subprocess.run([sys.executable, "-m", "src.eval.metrics_mmp",
                            "--short-root", args.val_root, "--scene", scene,
                            "--pred-dir", str(tmp), *gt], capture_output=True, text=True)
        m = re.search(r"Global IDF1:\s*([0-9.]+)", r.stdout)
        results[scene] = float(m.group(1)) if m else None
        print(f"[score] {scene:28s} IDF1={results[scene]}")
    vals = [v for v in results.values() if v is not None]
    print(f"\nMEAN ({'processed-only' if args.processed_only else 'full GT'}): "
          f"{sum(vals)/len(vals):.4f}" if vals else "no results")


if __name__ == "__main__":
    main()
