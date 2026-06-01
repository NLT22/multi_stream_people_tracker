"""
Offline evaluation script for MTA dataset.

Usage:
    python -m src.eval.metrics \\
        --gt-dir  dataset/mta/MTA_ext_short/test \\
        --pred-dir output/eval/mta_test \\
        [--cameras 0 1 2 3 4 5] \\
        [--iou-threshold 0.5]

Metrics computed:
    Per-camera : MOTA, MOTP, IDF1, IDS  (motmetrics)
    Global     : IDF1 using global_id across cameras (motmetrics)
    Per-camera : HOTA  (trackeval)

Required packages:
    pip install motmetrics pandas trackeval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks — fail early with a helpful message
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    sys.exit("[eval] pandas not found. Install: pip install pandas")

try:
    import numpy as np
except ImportError:
    sys.exit("[eval] numpy not found. Install: pip install numpy")

try:
    import motmetrics as mm
except ImportError:
    sys.exit("[eval] motmetrics not found. Install: pip install motmetrics")

try:
    import trackeval
    _TRACKEVAL_AVAILABLE = True
except ImportError:
    _TRACKEVAL_AVAILABLE = False
    print("[eval] WARNING: trackeval not found — HOTA will be skipped. "
          "Install: pip install trackeval")

from src.dataset.mta import MtaDataset


# ---------------------------------------------------------------------------
# GT / pred difficulty filter
# ---------------------------------------------------------------------------

# Defaults derived from model capability:
#   YOLOv11n input = 640px, source = 1920px → scale = 1/3
#   min detectable object at 640px ≈ 20px → 60px in 1920px source
_DEFAULT_MIN_HEIGHT = 60   # px in source resolution (1920×1080)
_DEFAULT_MIN_WIDTH  = 20   # px in source resolution
_DEFAULT_MIN_VIS    = 0.3  # fraction of box that must be inside frame


def _filter_boxes(
    df: pd.DataFrame,
    min_height: float,
    min_width: float,
    min_visibility: float,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> pd.DataFrame:
    """Remove boxes that are too small or mostly outside the frame.

    Works for both GT DataFrames (columns: left, top, width, height)
    and pred DataFrames (same columns).

    Filters applied in order:
      1. min_width  / min_height — removes tiny, un-detectable boxes
      2. min_visibility — fraction of the box area that lies inside the
         frame.  Removes persons who are mostly outside the camera view
         (simulation artifact: GT assigned even when only 1-2px visible).
    """
    w = df["width"].values
    h = df["height"].values
    l = df["left"].values
    t = df["top"].values

    size_ok = (w >= min_width) & (h >= min_height)

    # Visible area = intersection of box with [0,W]×[0,H]
    ix1 = np.clip(l,     0, frame_w)
    iy1 = np.clip(t,     0, frame_h)
    ix2 = np.clip(l + w, 0, frame_w)
    iy2 = np.clip(t + h, 0, frame_h)
    vis_area  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    total_area = np.maximum(1, w * h)
    vis_ok = (vis_area / total_area) >= min_visibility

    return df[size_ok & vis_ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pred(pred_dir: Path, cam_id: int) -> pd.DataFrame:
    path = pred_dir / f"cam_{cam_id}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"frame_no_cam": "frame"})
    return df


def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between GT and pred boxes (both Nx4: left,top,w,h)."""
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
        denom = area_g + area_p - inter
        iou[i] = np.where(denom > 0, inter / denom, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Per-camera motmetrics evaluation
# ---------------------------------------------------------------------------

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

        gt_ids = g["person_id"].tolist()
        pred_ids = p["local_track_id"].tolist()

        gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)
        pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        iou = _iou_matrix(gt_boxes, pred_boxes)
        dist = np.where(iou >= iou_threshold, 1.0 - iou, np.nan)
        acc.update(gt_ids, pred_ids, dist)

    return acc


# ---------------------------------------------------------------------------
# Global cross-camera IDF1 using global_id
# ---------------------------------------------------------------------------

def _eval_global_idf1(
    all_gt: dict[int, pd.DataFrame],
    all_pred: dict[int, pd.DataFrame],
    cam_ids: list[int],
    iou_threshold: float,
) -> dict:
    """Compute cross-camera IDF1 at trajectory level to avoid OOM.

    Instead of building a per-frame accumulator across all cameras (which
    creates an N_gt_boxes × N_pred_boxes matrix and OOMs on MTA), we:

    1. For each (cam, frame), match GT boxes → pred boxes by IoU.
    2. Accumulate, per GT person_id, which global_ids were matched to it
       (weighted by number of matched frames = TP hits).
    3. Solve a linear assignment: each GT person_id → best global_id.
    4. Compute IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN).

    Returns a dict with keys: idf1, idtp, idfp, idfn,
                               num_gt_ids, num_pred_ids.
    """
    from scipy.optimize import linear_sum_assignment

    # hits[person_id][global_id] = number of IoU-matched frames
    hits: dict[int, dict[int, int]] = {}
    gt_det_count:  dict[int, int] = {}   # total GT detections per person_id
    pred_det_count: dict[int, int] = {}  # total pred detections per global_id

    for cam_id in cam_ids:
        gt_df   = all_gt.get(cam_id)
        pred_df = all_pred.get(cam_id)
        if gt_df is None or pred_df is None:
            continue

        all_frames = sorted(
            set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())
        )
        for frame in all_frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]

            gt_pids  = g["person_id"].tolist()
            gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)

            # Replace unassigned global_id (-1) with a unique negative sentinel
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

            if len(gt_pids) == 0 or len(pred_gids) == 0:
                continue

            iou = _iou_matrix(gt_boxes, pred_boxes)
            # Greedy row→col assignment at IoU threshold
            row_ind, col_ind = linear_sum_assignment(-iou)
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] >= iou_threshold:
                    pid = int(gt_pids[r])
                    gid = int(pred_gids[c])
                    hits.setdefault(pid, {})
                    hits[pid][gid] = hits[pid].get(gid, 0) + 1

    # --- Linear assignment: person_id rows, global_id cols -----------------
    all_pids = sorted(gt_det_count.keys())
    all_gids = sorted(pred_det_count.keys())
    pid_idx  = {p: i for i, p in enumerate(all_pids)}
    gid_idx  = {g: i for i, g in enumerate(all_gids)}

    if not all_pids or not all_gids:
        return dict(idf1=0.0, idtp=0, idfp=sum(pred_det_count.values()),
                    idfn=sum(gt_det_count.values()),
                    num_gt_ids=len(all_pids), num_pred_ids=len(all_gids))

    cost = np.zeros((len(all_pids), len(all_gids)), dtype=np.int64)
    for pid, gid_map in hits.items():
        for gid, cnt in gid_map.items():
            if pid in pid_idx and gid in gid_idx:
                cost[pid_idx[pid], gid_idx[gid]] = cnt

    row_ind, col_ind = linear_sum_assignment(-cost)

    idtp = int(cost[row_ind, col_ind].sum())
    idfn = sum(gt_det_count.values())   - idtp
    idfp = sum(pred_det_count.values()) - idtp
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

    return dict(idf1=idf1, idtp=idtp, idfp=idfp, idfn=idfn,
                num_gt_ids=len(all_pids), num_pred_ids=len(all_gids))


# ---------------------------------------------------------------------------
# HOTA via trackeval
# ---------------------------------------------------------------------------

def _eval_hota_camera(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cam_id: int,
) -> dict | None:
    if not _TRACKEVAL_AVAILABLE:
        return None

    # TrackEval expects dicts keyed by frame with arrays:
    #   gt_ids, tracker_ids, gt_dets, tracker_dets
    # We build these from the dataframes.
    from trackeval.metrics import HOTA as HOTAMetric

    all_frames = sorted(
        set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())
    )

    # Remap string IDs to integers for TrackEval
    gt_id_map: dict[int, int] = {}
    pred_id_map: dict[int, int] = {}

    def _remap(val, id_map):
        if val not in id_map:
            id_map[val] = len(id_map)
        return id_map[val]

    data: dict = {
        "num_gt_ids": 0,
        "num_tracker_ids": 0,
        "num_gt_dets": 0,
        "num_tracker_dets": 0,
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }

    for frame in all_frames:
        g = gt_df[gt_df["frame"] == frame]
        p = pred_df[pred_df["frame"] == frame]

        gt_ids_raw = g["person_id"].tolist()
        pred_ids_raw = p["local_track_id"].tolist()

        gt_ids_int = [_remap(x, gt_id_map) for x in gt_ids_raw]
        pred_ids_int = [_remap(x, pred_id_map) for x in pred_ids_raw]

        gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)
        pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

        sim = _iou_matrix(gt_boxes, pred_boxes)

        data["gt_ids"].append(np.array(gt_ids_int, dtype=int))
        data["tracker_ids"].append(np.array(pred_ids_int, dtype=int))
        data["similarity_scores"].append(sim)
        data["num_gt_dets"] += len(gt_ids_int)
        data["num_tracker_dets"] += len(pred_ids_int)

    data["num_gt_ids"] = len(gt_id_map)
    data["num_tracker_ids"] = len(pred_id_map)

    hota = HOTAMetric()
    try:
        result = hota.eval_sequence(data)
        # HOTA returns arrays per alpha threshold; take mean over alphas.
        return {k: float(np.mean(v)) for k, v in result.items()
                if isinstance(v, np.ndarray)}
    except Exception as exc:
        print(f"  [eval] HOTA failed for cam_{cam_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline evaluation: MOTA/IDF1/HOTA for MTA predictions")
    p.add_argument("--gt-dir", required=True,
                   help="MTA split folder (e.g. dataset/mta/MTA_ext_short/test)")
    p.add_argument("--pred-dir", required=True,
                   help="Folder containing cam_<N>_predictions.csv files")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Which camera IDs to evaluate. Default: all found in pred-dir")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for GT-prediction matching (default: 0.5)")
    p.add_argument("--min-height", type=float, default=_DEFAULT_MIN_HEIGHT,
                   help=f"Minimum box height in source pixels to include in eval "
                        f"(default: {_DEFAULT_MIN_HEIGHT}). "
                        f"Set 0 to disable.")
    p.add_argument("--min-width", type=float, default=_DEFAULT_MIN_WIDTH,
                   help=f"Minimum box width in source pixels (default: {_DEFAULT_MIN_WIDTH}). "
                        f"Set 0 to disable.")
    p.add_argument("--min-visibility", type=float, default=_DEFAULT_MIN_VIS,
                   help=f"Minimum fraction of box area inside frame (default: {_DEFAULT_MIN_VIS}). "
                        f"Set 0 to disable.")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable all difficulty filters (evaluate on raw GT).")
    p.add_argument("--split", default="test",
                   help="MTA split name (default: test). Used only when --gt-dir "
                        "points to the dataset root rather than the split folder.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    iou_thr = args.iou_threshold

    # Load MTA GT — try args.gt_dir directly as split folder, then with --split
    gt_dir = Path(args.gt_dir)
    if not (gt_dir / "cam_0").exists():
        gt_dir = gt_dir / args.split
    try:
        mta = MtaDataset(str(gt_dir.parent), split=gt_dir.name)
    except FileNotFoundError as e:
        sys.exit(f"[eval] {e}")

    # Determine which cameras to evaluate
    if args.cameras:
        cam_ids = args.cameras
    else:
        cam_ids = [
            int(p.name.split("_")[1])
            for p in sorted(pred_dir.glob("cam_*_predictions.csv"))
        ]
        if not cam_ids:
            sys.exit(f"[eval] No cam_*_predictions.csv found in {pred_dir}")

    # Difficulty filter settings
    if args.no_filter:
        min_h = min_w = min_vis = 0.0
    else:
        min_h   = args.min_height
        min_w   = args.min_width
        min_vis = args.min_visibility

    print(f"[eval] GT:   {gt_dir}")
    print(f"[eval] Pred: {pred_dir}")
    print(f"[eval] Cameras: {cam_ids}  IoU threshold: {iou_thr}")
    if min_h > 0 or min_w > 0 or min_vis > 0:
        print(f"[eval] Difficulty filter: min_height={min_h}px  "
              f"min_width={min_w}px  min_visibility={min_vis:.0%}  "
              f"(--no-filter to disable)")
    else:
        print("[eval] Difficulty filter: disabled")
    print()

    all_gt: dict[int, pd.DataFrame] = {}
    all_pred: dict[int, pd.DataFrame] = {}
    per_cam_accs: dict[int, mm.MOTAccumulator] = {}
    hota_results: dict[int, dict | None] = {}

    for cam_id in cam_ids:
        try:
            gt_df = mta.load_gt(cam_id)
        except FileNotFoundError as e:
            print(f"  [cam_{cam_id}] GT not found: {e} — skipping")
            continue
        try:
            pred_df = _load_pred(pred_dir, cam_id)
        except FileNotFoundError as e:
            print(f"  [cam_{cam_id}] Predictions not found: {e} — skipping")
            continue

        # Apply difficulty filter to both GT and predictions
        if min_h > 0 or min_w > 0 or min_vis > 0:
            gt_raw, pred_raw = len(gt_df), len(pred_df)
            gt_df   = _filter_boxes(gt_df,   min_h, min_w, min_vis)
            pred_df = _filter_boxes(pred_df, min_h, min_w, min_vis)
            print(f"  [cam_{cam_id}] GT {gt_raw}→{len(gt_df)}  "
                  f"Pred {pred_raw}→{len(pred_df)}  "
                  f"(after difficulty filter)")
        else:
            print(f"  [cam_{cam_id}] GT detections={len(gt_df)}  "
                  f"Pred detections={len(pred_df)}")

        # Store filtered DFs — used by global IDF1 below
        all_gt[cam_id]   = gt_df
        all_pred[cam_id] = pred_df

        acc = _eval_camera_motmetrics(gt_df, pred_df, iou_thr)
        per_cam_accs[cam_id] = acc

        if _TRACKEVAL_AVAILABLE:
            hota_results[cam_id] = _eval_hota_camera(gt_df, pred_df, cam_id)

    if not per_cam_accs:
        sys.exit("[eval] No cameras could be evaluated.")

    # ------------------------------------------------------------------
    # motmetrics summary
    # ------------------------------------------------------------------
    mh = mm.metrics.create()
    metric_names = ["num_frames", "mota", "motp", "idf1", "num_switches",
                    "num_fragmentations", "num_misses", "num_false_positives",
                    "precision", "recall"]

    print()
    print("=" * 72)
    print(" Per-camera tracking metrics (motmetrics)")
    print("=" * 72)

    rows = []
    for cam_id in cam_ids:
        acc = per_cam_accs.get(cam_id)
        if acc is None:
            continue
        summary = mh.compute(acc, metrics=metric_names, name=f"cam_{cam_id}")
        rows.append(summary)

    if rows:
        combined = pd.concat(rows)
        print(combined.to_string(float_format=lambda x: f"{x:.4f}"))

    # Global IDF1 across all cameras (trajectory-level, avoids OOM)
    print()
    print("=" * 72)
    print(" Global cross-camera IDF1 (global_id vs person_id)")
    print("=" * 72)
    g = _eval_global_idf1(all_gt, all_pred, cam_ids, iou_thr)
    print(f"  IDF1         : {g['idf1']:.4f}")
    print(f"  IDTP         : {g['idtp']}")
    print(f"  IDFP         : {g['idfp']}")
    print(f"  IDFN         : {g['idfn']}")
    print(f"  GT IDs       : {g['num_gt_ids']}")
    print(f"  Pred IDs     : {g['num_pred_ids']}")

    # ------------------------------------------------------------------
    # HOTA summary
    # ------------------------------------------------------------------
    if _TRACKEVAL_AVAILABLE and hota_results:
        print()
        print("=" * 72)
        print(" Per-camera HOTA (trackeval)")
        print("=" * 72)
        hota_rows = []
        for cam_id in cam_ids:
            res = hota_results.get(cam_id)
            if res is None:
                continue
            row = {"camera": f"cam_{cam_id}"}
            for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr"):
                row[k] = round(res.get(k, float("nan")) * 100, 2)
            hota_rows.append(row)
        if hota_rows:
            print(pd.DataFrame(hota_rows).set_index("camera").to_string())
    elif not _TRACKEVAL_AVAILABLE:
        print()
        print("[eval] HOTA skipped (trackeval not installed).")

    print()
    print("[eval] Done.")


if __name__ == "__main__":
    main()
