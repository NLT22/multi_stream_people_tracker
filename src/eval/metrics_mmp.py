"""
Offline evaluation — MOTA / IDF1 / HOTA for MMPTracking_short predictions.

Hai điểm khác MTA eval:
  1. GT load từ MMPTracking_short CSV (640×360).
  2. Prediction cam_id = source_id (0-based index), không phải camera ID thật
     (1-based). Mapping: source_id N → cam_ids[N] theo thứ tự get_cam_ids().
  3. Tọa độ prediction ở không gian 1920×1080 (nvstreammux input) → scale về
     640×360 trước khi so với GT.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
except ImportError:
    sys.exit("[eval] motmetrics not found.  pip install motmetrics")

try:
    import trackeval as _te
    _TRACKEVAL_AVAILABLE = True
except ImportError:
    _TRACKEVAL_AVAILABLE = False

from src.dataset.mmp_tracking import MMPTrackingShortDataset

# MMPTracking_short source resolution
GT_W, GT_H = 640, 360

# Pipeline mux resolution (nvstreammux default)
PRED_W, PRED_H = 1920, 1080

# Scale factors: prediction → GT space
SCALE_X = GT_W / PRED_W   # 1/3
SCALE_Y = GT_H / PRED_H   # 1/3

# Default difficulty filter (GT space, 640×360)
_DEFAULT_MIN_H   = 20.0
_DEFAULT_MIN_W   = 8.0
_DEFAULT_MIN_VIS = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pred(pred_dir: Path, source_id: int) -> pd.DataFrame:
    """Load cam_<source_id>_predictions.csv and scale coordinates to GT space."""
    path = pred_dir / f"cam_{source_id}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"frame_no_cam": "frame"})
    # Scale from pipeline space (1920×1080) to GT space (640×360)
    df["left"]   = df["left"]   * SCALE_X
    df["top"]    = df["top"]    * SCALE_Y
    df["width"]  = df["width"]  * SCALE_X
    df["height"] = df["height"] * SCALE_Y
    return df


def _filter_boxes(df: pd.DataFrame, min_h: float, min_w: float,
                  min_vis: float) -> pd.DataFrame:
    w = df["width"].values
    h = df["height"].values
    l = df["left"].values
    t = df["top"].values

    size_ok = (w >= min_w) & (h >= min_h)

    ix1 = np.clip(l,     0, GT_W)
    iy1 = np.clip(t,     0, GT_H)
    ix2 = np.clip(l + w, 0, GT_W)
    iy2 = np.clip(t + h, 0, GT_H)
    vis  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area = np.maximum(1, w * h)
    vis_ok = (vis / area) >= min_vis

    return df[size_ok & vis_ok].reset_index(drop=True)


def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    def to_xyxy(b):
        return np.stack([b[:, 0], b[:, 1],
                         b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)

    ga = to_xyxy(gt_boxes)
    pa = to_xyxy(pred_boxes)
    iou = np.zeros((len(ga), len(pa)))
    for i, g in enumerate(ga):
        ix1 = np.maximum(g[0], pa[:, 0])
        iy1 = np.maximum(g[1], pa[:, 1])
        ix2 = np.minimum(g[2], pa[:, 2])
        iy2 = np.minimum(g[3], pa[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_g = (g[2] - g[0]) * (g[3] - g[1])
        area_p = (pa[:, 2] - pa[:, 0]) * (pa[:, 3] - pa[:, 1])
        denom  = area_g + area_p - inter
        iou[i] = np.where(denom > 0, inter / denom, 0.0)
    return iou


def _eval_camera_motmetrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float,
) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(
        set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())
    )
    for frame in all_frames:
        g = gt_df[gt_df["frame"] == frame]
        p = pred_df[pred_df["frame"] == frame]

        gt_ids  = g["person_id"].tolist()
        gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)

        pred_ids  = p["global_id"].tolist()
        pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        dists = 1.0 - _iou_matrix(gt_boxes, pred_boxes) if (
            len(gt_ids) > 0 and len(pred_ids) > 0
        ) else np.empty((len(gt_ids), len(pred_ids)))

        dists[dists > 1 - iou_threshold] = np.nan
        acc.update(gt_ids, pred_ids, dists)
    return acc


def _eval_global_idf1(
    all_gt: dict[int, pd.DataFrame],
    all_pred: dict[int, pd.DataFrame],
    iou_threshold: float,
) -> dict:
    """Cross-camera IDF1 using linear assignment of person_id → global_id."""
    from scipy.optimize import linear_sum_assignment

    hits: dict[int, dict[int, int]] = {}
    gt_det_count:   dict[int, int] = {}
    pred_det_count: dict[int, int] = {}

    for cam_id in all_gt:
        gt_df   = all_gt[cam_id]
        pred_df = all_pred.get(cam_id)
        if pred_df is None:
            continue

        all_frames = sorted(
            set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())
        )
        for frame in all_frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]

            gt_pids   = g["person_id"].tolist()
            gt_boxes  = g[["left", "top", "width", "height"]].values.astype(float)
            pred_gids = [
                int(gid) if int(gid) != -1
                else -(cam_id * 10_000_000 + int(frame) * 10_000 + int(tid))
                for gid, tid in zip(p["global_id"].tolist(),
                                    p["local_track_id"].tolist())
            ]
            pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

            for pid in gt_pids:
                gt_det_count[pid] = gt_det_count.get(pid, 0) + 1
            for gid in pred_gids:
                pred_det_count[gid] = pred_det_count.get(gid, 0) + 1

            if not gt_pids or not pred_gids:
                continue

            iou = _iou_matrix(gt_boxes, pred_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou)
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] >= iou_threshold:
                    pid = int(gt_pids[r])
                    gid = int(pred_gids[c])
                    hits.setdefault(pid, {})
                    hits[pid][gid] = hits[pid].get(gid, 0) + 1

    all_pids = sorted(gt_det_count)
    all_gids = sorted(pred_det_count)
    if not all_pids or not all_gids:
        return dict(idf1=0.0, idtp=0,
                    idfp=sum(pred_det_count.values()),
                    idfn=sum(gt_det_count.values()),
                    num_gt_ids=len(all_pids), num_pred_ids=len(all_gids))

    pid_idx = {p: i for i, p in enumerate(all_pids)}
    gid_idx = {g: i for i, g in enumerate(all_gids)}
    cost = np.zeros((len(all_pids), len(all_gids)), dtype=np.int64)
    for pid, gid_map in hits.items():
        for gid, cnt in gid_map.items():
            if pid in pid_idx and gid in gid_idx:
                cost[pid_idx[pid], gid_idx[gid]] = cnt

    row_ind, col_ind = linear_sum_assignment(-cost)
    idtp = int(cost[row_ind, col_ind].sum())
    idfn = sum(gt_det_count.values()) - idtp
    idfp = sum(pred_det_count.values()) - idtp
    idf1 = (2 * idtp) / max(1, 2 * idtp + idfp + idfn)

    return dict(idf1=idf1, idtp=idtp, idfp=idfp, idfn=idfn,
                num_gt_ids=len(all_pids), num_pred_ids=len(all_gids))


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def _eval_scene(
    scene: str,
    short_root: Path,
    pred_dir: Path,
    iou_thr: float,
    min_h: float,
    min_w: float,
    min_vis: float,
    cam_ids_override: list[int] | None,
) -> dict:
    try:
        ds = MMPTrackingShortDataset(str(short_root), scene)
    except FileNotFoundError as e:
        print(f"[{scene}] ERROR: {e}")
        return {}

    # cam_ids thực của scene (1-based, e.g. [1,2,3,4])
    real_cam_ids = cam_ids_override if cam_ids_override else ds.get_cam_ids()
    # source_id (0-based) → real_cam_id
    source_to_cam = {i: c for i, c in enumerate(real_cam_ids)}

    all_gt:       dict[int, pd.DataFrame] = {}
    all_pred:     dict[int, pd.DataFrame] = {}
    per_cam_accs: dict[int, mm.MOTAccumulator] = {}

    print(f"\n[{scene}] real_cam_ids={real_cam_ids}  "
          f"source_id mapping: {source_to_cam}")

    for source_id, cam_id in source_to_cam.items():
        try:
            gt_df = ds.load_gt(cam_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [cam{cam_id}] GT not found: {e} — skipping")
            continue

        try:
            pred_df = _load_pred(pred_dir, source_id)
        except FileNotFoundError as e:
            print(f"  [cam{cam_id}] {e} — skipping")
            continue

        if min_h > 0 or min_w > 0 or min_vis > 0:
            gt_raw, pred_raw = len(gt_df), len(pred_df)
            gt_df   = _filter_boxes(gt_df,   min_h, min_w, min_vis)
            pred_df = _filter_boxes(pred_df, min_h, min_w, min_vis)
            print(f"  [cam{cam_id}] GT {gt_raw}→{len(gt_df)}  "
                  f"Pred {pred_raw}→{len(pred_df)}")
        else:
            print(f"  [cam{cam_id}] GT={len(gt_df)}  Pred={len(pred_df)}")

        all_gt[cam_id]   = gt_df
        all_pred[cam_id] = pred_df
        per_cam_accs[cam_id] = _eval_camera_motmetrics(gt_df, pred_df, iou_thr)

    return {
        "per_cam_accs": per_cam_accs,
        "all_gt":       all_gt,
        "all_pred":     all_pred,
        "iou_thr":      iou_thr,
    }


def _print_scene_summary(scene: str, result: dict) -> None:
    if not result or not result["per_cam_accs"]:
        print(f"[{scene}] No cameras evaluated.")
        return

    per_cam_accs = result["per_cam_accs"]
    all_gt       = result["all_gt"]
    all_pred     = result["all_pred"]
    iou_thr      = result["iou_thr"]

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
        g = _eval_global_idf1(all_gt, all_pred, iou_thr)
        print(f"  Global IDF1: {g['idf1']:.4f}  "
              f"(IDTP={g['idtp']}  IDFP={g['idfp']}  IDFN={g['idfn']}  "
              f"GT IDs={g['num_gt_ids']}  Pred IDs={g['num_pred_ids']})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval MOTA/IDF1 on MMPTracking_short predictions")
    p.add_argument("--short-root", default="dataset/MMPTracking_short")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene",     default=None,
                      help="Single scene name, e.g. lobby_0")
    mode.add_argument("--pred-root", default=None,
                      help="Root dir containing one sub-dir per scene")

    p.add_argument("--pred-dir", default=None,
                   help="Prediction dir for --scene mode")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Real camera IDs to evaluate (default: all)")
    p.add_argument("--iou-threshold",  type=float, default=0.5)
    p.add_argument("--min-height",     type=float, default=_DEFAULT_MIN_H)
    p.add_argument("--min-width",      type=float, default=_DEFAULT_MIN_W)
    p.add_argument("--min-visibility", type=float, default=_DEFAULT_MIN_VIS)
    p.add_argument("--no-filter",      action="store_true",
                   help="Disable difficulty filter")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    short_root = Path(args.short_root)

    min_h   = 0.0 if args.no_filter else args.min_height
    min_w   = 0.0 if args.no_filter else args.min_width
    min_vis = 0.0 if args.no_filter else args.min_visibility

    print(f"[eval] short-root    : {short_root}")
    print(f"[eval] IoU threshold : {args.iou_threshold}")
    print(f"[eval] Pred space    : {PRED_W}×{PRED_H} → scale {SCALE_X:.4f}×{SCALE_Y:.4f} → GT {GT_W}×{GT_H}")
    if min_h > 0 or min_w > 0 or min_vis > 0:
        print(f"[eval] Filter        : min_h={min_h} min_w={min_w} min_vis={min_vis:.0%}")
    else:
        print("[eval] Filter        : disabled")

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

    # ── All scenes ────────────────────────────────────────────────────────────
    pred_root = Path(args.pred_root)
    scenes = sorted(
        d.name for d in pred_root.iterdir()
        if d.is_dir() and (short_root / d.name).exists()
    )
    if not scenes:
        sys.exit(f"[eval] No matching scenes under {pred_root}")

    print(f"[eval] {len(scenes)} scenes: {scenes}\n")

    all_accs_flat:  list[mm.MOTAccumulator] = []
    all_names_flat: list[str] = []
    grand_gt:   dict[str, pd.DataFrame] = {}
    grand_pred: dict[str, pd.DataFrame] = {}

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
            for cam_id, df in result["all_gt"].items():
                grand_gt[f"{scene}_cam{cam_id}"]   = df
            for cam_id, df in result["all_pred"].items():
                grand_pred[f"{scene}_cam{cam_id}"] = df

    if not all_accs_flat:
        sys.exit("[eval] No cameras evaluated.")

    mh = mm.metrics.create()
    grand = mh.compute_many(
        all_accs_flat,
        metrics=["mota", "motp", "idf1", "num_switches",
                 "num_misses", "num_false_positives", "precision", "recall"],
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
          f"(GT IDs={g['num_gt_ids']}  Pred IDs={g['num_pred_ids']})")


if __name__ == "__main__":
    main()
