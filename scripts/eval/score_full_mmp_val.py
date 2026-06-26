"""
Score full MMPTracking val predictions with buffered IDs.

For each scene under export_root:
  1. Run live_buffered --once  → _eval_assign.csv
  2. Remap global_id from assignments into predictions
  3. Score with _eval_global_idf1 → metrics.json

Skips scenes already scored (metrics.json with valid idf1 exists).

Usage:
    python scripts/eval/score_full_mmp_val.py \
        --export-root output/eval/full_mmp_val \
        --val-root dataset/MMPTracking_10minute/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd


def run_live_buffered(export_dir: Path, assign_csv: Path,
                      window_chunks: int = 1, assign_thr: float = 0.50,
                      fp_filter: bool = False, fp_motion: float = 8.0,
                      fp_minframes: int = 100) -> None:
    # Import directly (avoids subprocess python path issues with venv)
    from src.mtmc.live_buffered import main as lb_main
    sys.argv = [
        "live_buffered",
        "--export-dir", str(export_dir),
        "--window-chunks", str(window_chunks),
        "--assign-thr", str(assign_thr),
        "--assign-csv", str(assign_csv),
        "--once",
    ]
    if fp_filter:
        sys.argv += ["--fp-filter", "--fp-motion", str(fp_motion),
                     "--fp-minframes", str(fp_minframes)]
    lb_main()


def score_scene(scene_dir: Path, val_root: Path, scene: str,
                per_camera: bool = True, with_hota: bool = True) -> float | None:
    from src.eval.mmp_metrics.core import (
        _eval_global_idf1, compute_per_camera_metrics,
    )

    assign_csv = scene_dir / "_eval_assign.csv"
    assign = pd.read_csv(assign_csv)
    gid_map = {
        (int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
        for r in assign.itertuples()
    }

    cam_files = sorted(scene_dir.glob("cam_*_predictions.csv"))
    source_ids = [int(p.stem.split("_")[1]) for p in cam_files]

    scene_data = val_root / scene
    gt_cam_ids = sorted(int(p.stem[3:]) for p in scene_data.glob("cam*.mp4"))

    all_gt: dict = {}
    all_pred: dict = {}

    for src_id, gt_cam_id in zip(source_ids, gt_cam_ids):
        pred_path = scene_dir / f"cam_{src_id}_predictions.csv"
        gt_path = scene_data / f"gt_cam{gt_cam_id}_clean.csv"
        if not gt_path.exists():
            gt_path = scene_data / f"gt_cam{gt_cam_id}.csv"
        if not pred_path.exists() or not gt_path.exists():
            continue

        pred = pd.read_csv(pred_path)
        gt = pd.read_csv(gt_path)

        pred = pred.copy()
        pred["global_id"] = [
            gid_map.get((src_id, int(f), int(t)), -1)
            for f, t in zip(pred["frame_no_cam"], pred["local_track_id"])
        ]
        pred = pred[pred["global_id"] >= 0].copy()
        pred = pred.rename(columns={"frame_no_cam": "frame"})

        all_gt[gt_cam_id] = gt
        all_pred[gt_cam_id] = pred

    if not all_gt:
        return None

    result = _eval_global_idf1(all_gt, all_pred, iou_threshold=0.5)
    idf1 = result.get("idf1") or result.get("global_idf1") or result.get("mean_idf1")

    out = {"scene": scene, "idf1": idf1, **result}

    if per_camera:
        per_cam = compute_per_camera_metrics(
            all_gt, all_pred, iou_threshold=0.5,
            pred_id_col="global_id", with_hota=with_hota,
        )
        out["per_camera"] = per_cam
        # scene-level means of the per-camera tracking metrics
        if per_cam:
            keys = [k for k in per_cam[0] if k != "camera"]
            out["per_camera_mean"] = {
                k: round(sum(c.get(k, 0.0) for c in per_cam) / len(per_cam), 4)
                for k in keys
            }

    (scene_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return idf1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", default="output/eval/full_mmp_val")
    ap.add_argument("--val-root", default="dataset/MMPTracking_10minute/val")
    ap.add_argument("--window-chunks", type=int, default=1)
    ap.add_argument("--assign-thr", type=float, default=0.50)  # swept optimum (see sweep_live_buffered.py)
    ap.add_argument("--retail-fp-filter", action="store_true",
                    help="opt-in: apply the static-FP filter to retail (OBSOLETE with "
                         "the retail-clean detector; only useful for the old amodal detector)")
    ap.add_argument("--fp-motion", type=float, default=8.0)
    ap.add_argument("--fp-minframes", type=int, default=100)
    ap.add_argument("--rerun-scoring", action="store_true",
                    help="Re-run scoring even if metrics.json already exists.")
    ap.add_argument("--no-per-camera", action="store_true",
                    help="Skip per-camera MOTA/IDF1/HOTA (Global IDF1 only).")
    ap.add_argument("--no-hota", action="store_true",
                    help="Compute per-camera MOTA/IDF1 but skip HOTA (faster).")
    args = ap.parse_args()
    per_camera = not args.no_per_camera
    with_hota = not args.no_hota

    export_root = Path(args.export_root)
    val_root = Path(args.val_root)

    scene_dirs = sorted(d for d in export_root.glob("64pm_*") if d.is_dir())
    if not scene_dirs:
        print(f"No scene dirs found under {export_root}")
        sys.exit(1)

    results: dict[str, float] = {}

    for scene_dir in scene_dirs:
        scene = scene_dir.name

        # Skip if no predictions (pipeline not done)
        if not list(scene_dir.glob("cam_*_predictions.csv")):
            print(f"\n[{scene}] no predictions, skipping")
            continue

        # Load existing score if valid (and complete — has per_camera if asked)
        metrics_path = scene_dir / "metrics.json"
        if metrics_path.exists() and not args.rerun_scoring:
            d = json.loads(metrics_path.read_text())
            v = d.get("idf1") or d.get("global_idf1") or d.get("mean_idf1")
            complete = v is not None and (not per_camera or "per_camera" in d)
            if complete:
                print(f"\n[{scene}] already scored: IDF1={v:.4f}")
                results[scene] = float(v)
                continue

        print(f"\n── {scene} " + "─" * 30)

        # Step 1: live_buffered --once.
        # NOTE: the static-FP filter is now OBSOLETE — the deployed retail-clean
        # detector removes the shelf/phantom boxes at the source, so the filter
        # only clips real static people and HURTS (retail 0.661 -> 0.636). It is
        # therefore OFF by default. Opt back in (e.g. to evaluate the old amodal
        # detector) with --retail-fp-filter.
        fp_on = ("retail" in scene) and args.retail_fp_filter
        assign_csv = scene_dir / "_eval_assign.csv"
        if not assign_csv.exists():
            print(f"  live_buffered --once ... (fp_filter={fp_on})")
            run_live_buffered(scene_dir, assign_csv, args.window_chunks,
                              args.assign_thr, fp_filter=fp_on,
                              fp_motion=args.fp_motion, fp_minframes=args.fp_minframes)
            print(f"  → {assign_csv.name}")
        else:
            print("  _eval_assign.csv exists, skipping live_buffered.")

        # Step 2: score
        print("  scoring ...", flush=True)
        try:
            idf1 = score_scene(scene_dir, val_root, scene,
                               per_camera=per_camera, with_hota=with_hota)
            if idf1 is not None:
                print(f"  IDF1 = {idf1:.4f}")
                results[scene] = idf1
            else:
                print("  IDF1 = N/A (no GT/pred overlap)")
        except Exception as e:
            print(f"  scoring failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n==========================================")
    print(" Results summary")
    print("==========================================")
    print(f"{'Scene':<32}  {'IDF1':>6}")
    print(f"{'─' * 32}  {'──────'}")

    env_sums: dict[str, list[float]] = {}
    for scene in sorted(results):
        v = results[scene]
        print(f"{scene:<32}  {v:.4f}")
        env = scene.removeprefix("64pm_").rsplit("_", 1)[0]
        env_sums.setdefault(env, []).append(v)

    if results:
        print()
        for env in sorted(env_sums):
            vals = env_sums[env]
            print(f"  {env:<22}  {sum(vals)/len(vals):.4f}  ({len(vals)} scenes)")
        overall = sum(results.values()) / len(results)
        print(f"\n{'─' * 32}  {'──────'}")
        print(f"{'MEAN (' + str(len(results)) + ' scenes)':<32}  {overall:.4f}")


if __name__ == "__main__":
    main()
