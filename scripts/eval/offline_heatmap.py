#!/usr/bin/env python3
"""Offline per-camera heatmap pipeline (batch analytics).

Real-project offline flow for per-camera occupancy analytics:

  1. EXPORT   batch-run the production pipeline (clean YOLO detector + NvDCF +
              clean Swin ReID + gallery) over a recorded scene → per-camera
              detections with global IDs  (skipped if --pred-dir already exists).
  2. HEATMAP  accumulate every detection over the whole clip into a per-camera
              occupancy density (foot / dwell / visit), overlay on a scene frame,
              and tile all cameras into one montage.
  3. DATA     emit the structured 'information': per-camera occupancy GRID CSV +
              a stats JSON (peak hotspot, dwell-seconds, occupied fraction).
  4. VALIDATE (optional, --compare-gt) GT-vs-pred density CC/SIM/KL per camera.

Per-camera occupancy only depends on detection quality (not identity), so the
heatmap is accurate even where cross-camera IDF1 is not.

Usage:
  # full pipeline (export + heatmaps) for a 10-min scene
  python scripts/eval/offline_heatmap.py --scene 63am_lobby_3 \
      --short-root dataset/MMPTracking_10minute/train --out-dir output/heatmap/lobby3

  # reuse an existing export (skip step 1)
  python scripts/eval/offline_heatmap.py --scene 63am_lobby_3 \
      --short-root dataset/MMPTracking_10minute/train \
      --pred-dir output/eval/clean_63am_lobby_3 --out-dir output/heatmap/lobby3
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

DETECTOR = "configs/models/nvinfer_yolov11_10min_clean.yml"
TRACKER = "configs/tracker/nvdcf_accuracy_mmp_recall_clean.yaml"
PIPELINE = "configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml"


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--pred-dir", default=None, help="Existing export; if omitted, run the export.")
    ap.add_argument("--out-dir", default="output/heatmap")
    ap.add_argument("--modes", nargs="+", default=["foot", "dwell", "visit"])
    ap.add_argument("--grid-w", type=int, default=64, help="Occupancy-grid width for data dump.")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--geo-weight", type=float, default=0.35)
    ap.add_argument("--no-compare-gt", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pred_dir = args.pred_dir or f"output/eval/heatmap_{args.scene}"

    # 1. EXPORT (batch) — skip if the export already exists
    if args.pred_dir and Path(args.pred_dir, "cam_0_predictions.csv").exists():
        print(f"[1/4] export: reusing {args.pred_dir}")
    else:
        print(f"[1/4] export: running production pipeline → {pred_dir}")
        run([sys.executable, "-m", "src.main", "--config", PIPELINE,
             "--nvinfer-config", DETECTOR, "--tracker-config", TRACKER,
             "--geo-weight", str(args.geo_weight),
             "--mmp-short-dataset", f"{args.short_root}:{args.scene}",
             "--no-display", "--no-sync", "--export-predictions", pred_dir])

    # 2 + 3. HEATMAP + DATA for each mode
    for mode in args.modes:
        print(f"[2/4] heatmap+data: mode={mode}")
        run([sys.executable, "scripts/eval/camera_heatmap.py",
             "--short-root", args.short_root, "--scene", args.scene,
             "--pred-dir", pred_dir, "--mode", mode,
             "--dump-grid", str(args.grid_w), "--fps", str(args.fps),
             "--out-dir", args.out_dir])

    # 4. VALIDATE vs GT
    if not args.no_compare_gt:
        print("[4/4] validate vs GT (foot)")
        run([sys.executable, "scripts/eval/camera_heatmap.py",
             "--short-root", args.short_root, "--scene", args.scene,
             "--pred-dir", pred_dir, "--mode", "foot", "--compare-gt",
             "--out-dir", args.out_dir])

    print(f"\n[done] per-camera heatmaps + data + stats in {args.out_dir}/")


if __name__ == "__main__":
    main()
