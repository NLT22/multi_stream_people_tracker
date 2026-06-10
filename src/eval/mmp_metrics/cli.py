"""
Offline evaluation — MOTA / IDF1 / HOTA for MMPTracking_short predictions.

Khác biệt so với src.eval.metrics (MTA):
  1. GT load từ MMPTracking_short CSV (640×360).
  2. Prediction cam_id = source_id (0-based index theo thứ tự get_cam_ids()),
     không phải camera ID thật (1-based). Mapping tự động.
  3. Tọa độ prediction có thể ở source/tile/mux space. Mặc định auto-detect
     từ bbox extents rồi scale về GT space 640×360 trước khi so sánh.

Usage:
    # Single scene
    python -m src.eval.metrics_mmp \\
        --short-root dataset/MMPTracking_short \\
        --scene      lobby_0 \\
        --pred-dir   output/eval/baseline_mmp/lobby_0

    # All scenes
    python -m src.eval.metrics_mmp \\
        --short-root dataset/MMPTracking_short \\
        --pred-root  output/eval/baseline_mmp
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    import motmetrics as mm
except ImportError:
    sys.exit("[eval] motmetrics not found.  pip install motmetrics")

from src.eval.mmp_metrics.core import (
    GT_W, GT_H, PRED_W, PRED_H,
    _DEFAULT_MIN_HEIGHT, _DEFAULT_MIN_WIDTH, _DEFAULT_MIN_VIS,
    _eval_global_idf1, _eval_scene, _eval_scene_job,
    _load_exclude_ids, _print_scene_summary,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline evaluation: MOTA/IDF1 for MMPTracking_short predictions")
    p.add_argument("--short-root", default="dataset/MMPTracking_short",
                   help="MMPTracking_short root directory")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene",     default=None,
                      help="Single scene name, e.g. lobby_0")
    mode.add_argument("--pred-root", default=None,
                      help="Root dir containing one sub-dir per scene. "
                           "Evaluates all matching scenes.")

    p.add_argument("--pred-dir", default=None,
                   help="Prediction directory for --scene mode. "
                        "Default: output/eval/<scene>")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Camera IDs to evaluate (real IDs, e.g. 1 2 3 4). "
                        "Default: all cameras in scene")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for GT-prediction matching (default: 0.5)")
    p.add_argument("--min-height", type=float, default=_DEFAULT_MIN_HEIGHT,
                   help=f"Minimum box height in GT pixels "
                        f"(default: {_DEFAULT_MIN_HEIGHT}). Set 0 to disable.")
    p.add_argument("--min-width", type=float, default=_DEFAULT_MIN_WIDTH,
                   help=f"Minimum box width in GT pixels "
                        f"(default: {_DEFAULT_MIN_WIDTH}). Set 0 to disable.")
    p.add_argument("--min-visibility", type=float, default=_DEFAULT_MIN_VIS,
                   help=f"Minimum fraction of box area inside frame "
                        f"(default: {_DEFAULT_MIN_VIS}). Set 0 to disable.")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable all difficulty filters (evaluate on raw GT).")
    p.add_argument("--pred-width", type=float, default=None,
                   help=f"Prediction coordinate width before scaling to GT "
                        f"(default: auto; current tile width is {PRED_W}).")
    p.add_argument("--pred-height", type=float, default=None,
                   help=f"Prediction coordinate height before scaling to GT "
                        f"(default: auto; current tile height is {PRED_H}).")
    p.add_argument("--gt-suffix", default="", metavar="SUFFIX",
                   help="GT file suffix, e.g. '_clean' loads gt_cam1_clean.csv (default: '')")
    p.add_argument("--exclude-person-ids", default=None, metavar="FILE",
                   help="Path to a text file with person_ids (one per line) to "
                        "remove from GT before evaluation. Use detect_phantom_gt.py "
                        "to generate this file for retail scenes with occluded persons.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of scenes to evaluate in parallel for --pred-root mode "
                        "(default: 1).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    short_root = Path(args.short_root)

    if args.no_filter:
        min_height = min_width = min_visibility = 0.0
    else:
        min_height     = args.min_height
        min_width      = args.min_width
        min_visibility = args.min_visibility

    exclude_person_ids = _load_exclude_ids(args.exclude_person_ids)
    if exclude_person_ids:
        print(f"[eval] Exclude IDs   : {len(exclude_person_ids)} person_ids "
              f"from {args.exclude_person_ids}")

    print(f"[eval] short-root    : {short_root}")
    print(f"[eval] IoU threshold : {args.iou_threshold}")
    if args.pred_width is None or args.pred_height is None:
        print(f"[eval] Pred space    : auto-detect per scene → GT {GT_W}×{GT_H}")
    else:
        scale_x = GT_W / args.pred_width
        scale_y = GT_H / args.pred_height
        print(f"[eval] Pred space    : {args.pred_width:g}×{args.pred_height:g} "
              f"→ scale ×{scale_x:.4f}/×{scale_y:.4f} → GT {GT_W}×{GT_H}")
    if min_height > 0 or min_width > 0 or min_visibility > 0:
        print(f"[eval] Filter        : min_height={min_height}px  "
              f"min_width={min_width}px  min_visibility={min_visibility:.0%}  "
              f"(--no-filter to disable)")
    else:
        print("[eval] Filter        : disabled")

    # ── Single scene ──────────────────────────────────────────────────────────
    if args.scene:
        pred_dir = Path(args.pred_dir) if args.pred_dir \
                   else Path("output/eval") / args.scene
        result = _eval_scene(
            args.scene, short_root, pred_dir,
            args.iou_threshold, min_height, min_width, min_visibility,
            args.cameras,
            args.pred_width, args.pred_height,
            exclude_person_ids=exclude_person_ids,
            gt_suffix=args.gt_suffix,
        )
        _print_scene_summary(args.scene, result)
        return

    # ── All scenes ────────────────────────────────────────────────────────────
    pred_root = Path(args.pred_root)
    scenes = sorted(
        d.name for d in pred_root.iterdir()
        if d.is_dir() and (short_root / d.name).exists()
    )
    if not scenes:
        sys.exit(f"[eval] No matching scenes found under {pred_root}")

    print(f"[eval] Evaluating {len(scenes)} scenes: {scenes}\n")
    jobs = max(1, args.jobs)
    if jobs > 1:
        jobs = min(jobs, len(scenes), os.cpu_count() or jobs)
        print(f"[eval] Parallel jobs  : {jobs}")

    all_accs_flat:  list[mm.MOTAccumulator] = []
    all_names_flat: list[str] = []
    grand_gt:   dict[str, pd.DataFrame] = {}
    grand_pred: dict[str, pd.DataFrame] = {}

    results_by_scene: dict[str, dict] = {}
    if jobs > 1:
        params = [
            (
                scene,
                str(short_root),
                str(pred_root / scene),
                args.iou_threshold,
                min_height,
                min_width,
                min_visibility,
                args.cameras,
                args.pred_width,
                args.pred_height,
                exclude_person_ids,
                args.gt_suffix,
            )
            for scene in scenes
        ]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_eval_scene_job, p) for p in params]
            for i, fut in enumerate(as_completed(futures), start=1):
                scene, result = fut.result()
                results_by_scene[scene] = result
                print(f"[eval] completed {i}/{len(scenes)}: {scene}")
    else:
        for scene in scenes:
            pred_dir = pred_root / scene
            result = _eval_scene(
                scene, short_root, pred_dir,
                args.iou_threshold, min_height, min_width, min_visibility,
                args.cameras,
                args.pred_width, args.pred_height,
                exclude_person_ids=exclude_person_ids,
                gt_suffix=args.gt_suffix,
            )
            results_by_scene[scene] = result

    for scene in scenes:
        result = results_by_scene.get(scene, {})
        _print_scene_summary(scene, result)

        if result and result["per_cam_accs"]:
            for cam_id, acc in result["per_cam_accs"].items():
                all_accs_flat.append(acc)
                all_names_flat.append(f"{scene}_cam_{cam_id}")
            # Namespace keys theo scene để tránh trùng cam_id
            for cam_id, df in result["all_gt"].items():
                grand_gt[f"{scene}_cam_{cam_id}"]   = df
            for cam_id, df in result["all_pred"].items():
                grand_pred[f"{scene}_cam_{cam_id}"] = df

    if not all_accs_flat:
        sys.exit("[eval] No cameras evaluated across all scenes.")

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

    g = _eval_global_idf1(grand_gt, grand_pred, args.iou_threshold)
    print(f"  Grand Global IDF1 : {g['idf1']:.4f}  "
          f"(IDTP={g['idtp']}  IDFP={g['idfp']}  IDFN={g['idfn']}  "
          f"GT IDs={g['num_gt_ids']}  Pred IDs={g['num_pred_ids']})")




if __name__ == "__main__":
    main()
