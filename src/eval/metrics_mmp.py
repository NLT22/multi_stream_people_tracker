"""
Offline evaluation for MMPTracking_short predictions.

Wrapper around src.eval.metrics — same MOTA/MOTP/IDF1/HOTA logic,
but loads GT from MMPTracking_short CSV files instead of MTA CSVs.

Usage:
    # Single scene
    python -m src.eval.metrics_mmp \\
        --short-root dataset/MMPTracking_short \\
        --scene      lobby_0 \\
        --pred-dir   output/eval/baseline_mmp/lobby_0

    # All scenes in one run (aggregates per-scene then prints grand summary)
    python -m src.eval.metrics_mmp \\
        --short-root dataset/MMPTracking_short \\
        --pred-root  output/eval/baseline_mmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Re-use all eval logic from metrics.py
from src.eval.metrics import (
    _filter_boxes,
    _load_pred,
    _iou_matrix,
    _eval_camera_motmetrics,
    _eval_global_idf1,
    _TRACKEVAL_AVAILABLE,
)
if _TRACKEVAL_AVAILABLE:
    from src.eval.metrics import _eval_hota_camera

try:
    import motmetrics as mm
except ImportError:
    sys.exit("[eval] motmetrics not found. pip install motmetrics")

from src.dataset.mmp_tracking import MMPTrackingShortDataset

# MMPTracking_short resolution
_IMG_W = 640
_IMG_H = 360

# Difficulty filter — indoor 640×360, persons larger than MTA outdoor
_DEFAULT_MIN_H   = 20.0
_DEFAULT_MIN_W   = 8.0
_DEFAULT_MIN_VIS = 0.30


def _eval_scene(
    scene: str,
    short_root: Path,
    pred_dir: Path,
    iou_thr: float,
    min_h: float,
    min_w: float,
    min_vis: float,
    cam_ids: list[int] | None,
) -> dict:
    """Evaluate one scene. Returns dict with per-cam accs and global GT/pred."""
    try:
        ds = MMPTrackingShortDataset(str(short_root), scene)
    except FileNotFoundError as e:
        print(f"[{scene}] ERROR: {e}")
        return {}

    scene_cam_ids = cam_ids if cam_ids else ds.get_cam_ids()

    all_gt:        dict[int, pd.DataFrame] = {}
    all_pred:      dict[int, pd.DataFrame] = {}
    per_cam_accs:  dict[int, mm.MOTAccumulator] = {}
    hota_results:  dict[int, dict | None] = {}

    print(f"\n[{scene}] cameras={scene_cam_ids}")

    for cam_id in scene_cam_ids:
        try:
            gt_df = ds.load_gt(cam_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [cam{cam_id}] GT not found: {e} — skipping")
            continue

        try:
            pred_df = _load_pred(pred_dir, cam_id)
        except FileNotFoundError as e:
            print(f"  [cam{cam_id}] Predictions not found: {e} — skipping")
            continue

        if min_h > 0 or min_w > 0 or min_vis > 0:
            gt_raw, pred_raw = len(gt_df), len(pred_df)
            gt_df   = _filter_boxes(gt_df,   min_h, min_w, min_vis,
                                    frame_w=_IMG_W, frame_h=_IMG_H)
            pred_df = _filter_boxes(pred_df, min_h, min_w, min_vis,
                                    frame_w=_IMG_W, frame_h=_IMG_H)
            print(f"  [cam{cam_id}] GT {gt_raw}→{len(gt_df)}  "
                  f"Pred {pred_raw}→{len(pred_df)}")
        else:
            print(f"  [cam{cam_id}] GT={len(gt_df)}  Pred={len(pred_df)}")

        all_gt[cam_id]   = gt_df
        all_pred[cam_id] = pred_df

        acc = _eval_camera_motmetrics(gt_df, pred_df, iou_thr)
        per_cam_accs[cam_id] = acc

        if _TRACKEVAL_AVAILABLE:
            hota_results[cam_id] = _eval_hota_camera(gt_df, pred_df, cam_id)

    return {
        "per_cam_accs": per_cam_accs,
        "hota_results": hota_results,
        "all_gt":       all_gt,
        "all_pred":     all_pred,
    }


def _print_scene_summary(scene: str, result: dict) -> None:
    if not result or not result["per_cam_accs"]:
        print(f"[{scene}] No cameras evaluated.")
        return

    per_cam_accs  = result["per_cam_accs"]
    hota_results  = result["hota_results"]
    all_gt        = result["all_gt"]
    all_pred      = result["all_pred"]

    mh = mm.metrics.create()
    metric_names = ["num_frames", "mota", "motp", "idf1",
                    "num_switches", "num_fragmentations",
                    "num_misses", "num_false_positives",
                    "precision", "recall"]
    summary = mh.compute_many(
        list(per_cam_accs.values()),
        metrics=metric_names,
        names=[f"cam{c}" for c in per_cam_accs],
        generate_overall=True,
    )

    print(f"\n{'─'*60}")
    print(f"  {scene} — Per-Camera")
    print(f"{'─'*60}")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))

    if len(per_cam_accs) > 1:
        global_idf1 = _eval_global_idf1(all_gt, all_pred)
        print(f"  Global IDF1: {global_idf1:.4f}")

    if _TRACKEVAL_AVAILABLE and hota_results:
        valid = {c: r for c, r in hota_results.items() if r}
        if valid:
            print(f"\n  HOTA per-camera:")
            for cam_id, r in sorted(valid.items()):
                print(f"    cam{cam_id}: HOTA={r['HOTA']:.2f}  "
                      f"DetA={r['DetA']:.2f}  AssA={r['AssA']:.2f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval MOTA/IDF1/HOTA on MMPTracking_short predictions")
    p.add_argument("--short-root", default="dataset/MMPTracking_short",
                   help="MMPTracking_short root directory")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene", default=None,
                      help="Single scene name, e.g. lobby_0")
    mode.add_argument("--pred-root", default=None,
                      help="Evaluate all scenes found under this directory "
                           "(each sub-dir name = scene name)")

    p.add_argument("--pred-dir", default=None,
                   help="Prediction directory for --scene mode. "
                        "Default: output/eval/<scene>")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Camera IDs to evaluate. Default: all in scene")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--min-height",    type=float, default=_DEFAULT_MIN_H)
    p.add_argument("--min-width",     type=float, default=_DEFAULT_MIN_W)
    p.add_argument("--min-visibility",type=float, default=_DEFAULT_MIN_VIS)
    p.add_argument("--no-filter",     action="store_true",
                   help="Disable difficulty filter")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    short_root = Path(args.short_root)

    min_h   = 0.0 if args.no_filter else args.min_height
    min_w   = 0.0 if args.no_filter else args.min_width
    min_vis = 0.0 if args.no_filter else args.min_visibility

    print(f"[eval] short-root : {short_root}")
    print(f"[eval] IoU threshold: {args.iou_threshold}")
    if min_h > 0 or min_w > 0 or min_vis > 0:
        print(f"[eval] Filter: min_h={min_h}  min_w={min_w}  min_vis={min_vis:.0%}")
    else:
        print("[eval] Filter: disabled")

    # ── Single scene ──────────────────────────────────────────────────────────
    if args.scene:
        pred_dir = Path(args.pred_dir) if args.pred_dir \
                   else Path("output/eval") / args.scene
        result = _eval_scene(
            args.scene, short_root, pred_dir,
            args.iou_threshold, min_h, min_w, min_vis, args.cameras,
        )
        _print_scene_summary(args.scene, result)
        return

    # ── All scenes under pred-root ────────────────────────────────────────────
    pred_root = Path(args.pred_root)
    scenes = sorted(
        d.name for d in pred_root.iterdir()
        if d.is_dir() and (short_root / d.name).exists()
    )
    if not scenes:
        sys.exit(f"[eval] No matching scenes found under {pred_root}")

    print(f"[eval] Evaluating {len(scenes)} scenes: {scenes}\n")

    all_accs_flat: list[mm.MOTAccumulator] = []
    all_names_flat: list[str] = []
    grand_idf1_gt:   dict[str, pd.DataFrame] = {}
    grand_idf1_pred: dict[str, pd.DataFrame] = {}

    for scene in scenes:
        pred_dir = pred_root / scene
        result = _eval_scene(
            scene, short_root, pred_dir,
            args.iou_threshold, min_h, min_w, min_vis, args.cameras,
        )
        _print_scene_summary(scene, result)

        if result and result["per_cam_accs"]:
            for cam_id, acc in result["per_cam_accs"].items():
                all_accs_flat.append(acc)
                all_names_flat.append(f"{scene}_cam{cam_id}")
            # Namespace GT/pred keys by scene to avoid cam_id collision
            for cam_id, df in result["all_gt"].items():
                grand_idf1_gt[f"{scene}_cam{cam_id}"]   = df
            for cam_id, df in result["all_pred"].items():
                grand_idf1_pred[f"{scene}_cam{cam_id}"] = df

    if not all_accs_flat:
        sys.exit("[eval] No cameras evaluated across all scenes.")

    # Grand summary across all scenes
    mh = mm.metrics.create()
    metric_names = ["mota", "motp", "idf1", "num_switches",
                    "num_misses", "num_false_positives", "precision", "recall"]
    grand = mh.compute_many(
        all_accs_flat,
        metrics=metric_names,
        names=all_names_flat,
        generate_overall=True,
    )
    print(f"\n{'═'*60}")
    print(f"  GRAND SUMMARY — {len(scenes)} scenes")
    print(f"{'═'*60}")
    print(mm.io.render_summary(
        grand.loc[["OVERALL"]],
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))
    global_idf1 = _eval_global_idf1(grand_idf1_gt, grand_idf1_pred)
    print(f"  Grand Global IDF1: {global_idf1:.4f}")


if __name__ == "__main__":
    main()
