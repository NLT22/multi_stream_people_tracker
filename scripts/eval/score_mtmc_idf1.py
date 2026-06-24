#!/usr/bin/env python3
"""Global IDF1 scorer for MTMC_Tracking_2026 pipeline exports.

Reads the pipeline's cam_*_predictions.csv and the warehouse ground_truth.json,
then computes cross-camera Global IDF1 using the same algorithm as mmp_metrics.

Usage (single warehouse):
    python scripts/eval/score_mtmc_idf1.py \
        --export-dir output/eval/mtmc_w022 \
        --gt-json dataset/MTMC_Tracking_2026/val/Warehouse_022/ground_truth.json \
        --cam-ids 0 1 2 3

Usage (multi-warehouse, separate export dirs):
    python scripts/eval/score_mtmc_idf1.py \
        --export-dir output/eval/mtmc_w020 \
        --gt-json dataset/MTMC_Tracking_2026/val/Warehouse_020/ground_truth.json

The script auto-discovers cam IDs from the export-dir prediction CSVs if --cam-ids is omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# GT loading from ground_truth.json
# ---------------------------------------------------------------------------

def load_gt(gt_json: Path, cam_ids: list[int] | None = None) -> dict[int, pd.DataFrame]:
    """Stream ground_truth.json -> {cam_id: DataFrame(frame, person_id, left, top, width, height)}.

    Camera names in the JSON are like "Camera_0000"; cam_id is the trailing int.
    Bboxes are stored as [x1, y1, x2, y2] (visible portion).
    """
    rows: dict[int, list] = {}
    with open(gt_json, "r") as f:
        gt = json.load(f)

    for frame_str, objs in gt.items():
        fnum = int(frame_str)
        for obj in objs:
            if obj.get("object type") != "Person":
                continue
            pid = obj.get("object id")
            bboxes = obj.get("2d bounding box visible") or {}
            for cam_name, b in bboxes.items():
                cam_id = int(cam_name.split("_")[-1])
                if cam_ids is not None and cam_id not in cam_ids:
                    continue
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                rows.setdefault(cam_id, []).append(
                    (fnum, pid, x1, y1, w, h))

    result = {}
    for cam_id, r in rows.items():
        df = pd.DataFrame(r, columns=["frame", "person_id", "left", "top", "width", "height"])
        result[cam_id] = df
    return result


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_preds(export_dir: Path, cam_ids: list[int] | None = None,
               cam_offset: int = 0) -> dict[int, pd.DataFrame]:
    """Load predictions; cam_offset shifts pipeline cam IDs to GT cam IDs (e.g. -16 for W022)."""
    result = {}
    for p in sorted(export_dir.glob("cam_*_predictions.csv")):
        cid = int(p.stem.split("_")[1])
        if cam_ids is not None and cid not in cam_ids:
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={"frame_no_cam": "frame"})
        df = df[df["global_id"] >= 0].reset_index(drop=True)
        if not df.empty:
            result[cid + cam_offset] = df
    return result


# ---------------------------------------------------------------------------
# IOU helpers (same as mmp_metrics/core.py)
# ---------------------------------------------------------------------------

def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    def to_xyxy(b):
        return np.stack([b[:, 0], b[:, 1],
                         b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)
    ga = to_xyxy(gt_boxes)
    pa = to_xyxy(pred_boxes)
    iou = np.zeros((len(ga), len(pa)))
    for i, g in enumerate(ga):
        xi1 = np.maximum(g[0], pa[:, 0]); yi1 = np.maximum(g[1], pa[:, 1])
        xi2 = np.minimum(g[2], pa[:, 2]); yi2 = np.minimum(g[3], pa[:, 3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        ag = (g[2] - g[0]) * (g[3] - g[1])
        ap = (pa[:, 2] - pa[:, 0]) * (pa[:, 3] - pa[:, 1])
        union = ag + ap - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Global IDF1 (same algorithm as mmp_metrics/core.py _eval_global_idf1)
# ---------------------------------------------------------------------------

def global_idf1(all_gt: dict[int, pd.DataFrame],
                all_pred: dict[int, pd.DataFrame],
                iou_threshold: float = 0.5) -> dict:
    hits: dict = {}
    gt_det_count: dict = {}
    pred_det_count: dict = {}

    # Only score cameras present in both GT and predictions — cameras in GT but
    # not in predictions were not evaluated (don't count as missed).
    evaluated_cams = set(all_gt.keys()) & set(all_pred.keys())

    for cam_id, gt_df in all_gt.items():
        if cam_id not in evaluated_cams:
            continue
        pred_df = all_pred.get(cam_id)

        all_frames = sorted(
            set(gt_df["frame"].unique()) | set(pred_df["frame"].unique()))
        for frame in all_frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]

            gt_pids = g["person_id"].tolist()
            gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)
            pred_gids = p["global_id"].tolist()
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
                    pid = gt_pids[r]
                    gid = int(pred_gids[c])
                    hits.setdefault(pid, {})
                    hits[pid][gid] = hits[pid].get(gid, 0) + 1

    all_pids = sorted(gt_det_count.keys(), key=str)
    all_gids = sorted(pred_det_count.keys())

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
# GT / pred coordinate space reconciliation
# ---------------------------------------------------------------------------

def _infer_scale(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[float, float]:
    """GT is 1920x1080 source space; pred may be in pipeline (mux) space.

    Detect mismatch by comparing max bbox coordinate; return (sx, sy) to
    scale pred -> GT space. If both are 1920x1080 returns (1, 1).
    """
    gt_max_x = (gt_df["left"] + gt_df["width"]).max() if not gt_df.empty else 1920
    pred_max_x = (pred_df["left"] + pred_df["width"]).max() if not pred_df.empty else 1920
    if gt_max_x > 0 and pred_max_x > 0:
        sx = gt_max_x / pred_max_x
    else:
        sx = 1.0
    gt_max_y = (gt_df["top"] + gt_df["height"]).max() if not gt_df.empty else 1080
    pred_max_y = (pred_df["top"] + pred_df["height"]).max() if not pred_df.empty else 1080
    if gt_max_y > 0 and pred_max_y > 0:
        sy = gt_max_y / pred_max_y
    else:
        sy = 1.0
    return sx, sy


def rescale_preds(all_pred: dict, all_gt: dict) -> dict:
    """Scale prediction bboxes to GT coordinate space if needed."""
    out = {}
    for cam_id, pred_df in all_pred.items():
        gt_df = all_gt.get(cam_id)
        if gt_df is None or pred_df.empty:
            out[cam_id] = pred_df
            continue
        sx, sy = _infer_scale(gt_df, pred_df)
        if abs(sx - 1.0) > 0.05 or abs(sy - 1.0) > 0.05:
            print(f"  [scale] cam {cam_id}: pred×({sx:.3f},{sy:.3f}) → GT space")
            pred_df = pred_df.copy()
            pred_df["left"]   *= sx
            pred_df["top"]    *= sy
            pred_df["width"]  *= sx
            pred_df["height"] *= sy
        out[cam_id] = pred_df
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path,
                    help="Directory with cam_*_predictions.csv from the pipeline")
    ap.add_argument("--gt-json", required=True, type=Path,
                    help="ground_truth.json for this warehouse")
    ap.add_argument("--cam-ids", nargs="*", type=int, default=None,
                    help="Subset of camera IDs to evaluate (default: all in export-dir)")
    ap.add_argument("--iou-thr", type=float, default=0.5,
                    help="IoU threshold for GT-pred matching (default: 0.5)")
    ap.add_argument("--pred-cam-offset", type=int, default=0,
                    help="Add this value to pipeline cam IDs to map to GT cam IDs "
                         "(e.g. -16 when W022 cams appear as 16-19 in a 20-cam run)")
    args = ap.parse_args()

    # auto-discover cam ids from prediction CSVs if not specified
    if args.cam_ids is None:
        args.cam_ids = sorted(
            int(p.stem.split("_")[1])
            for p in args.export_dir.glob("cam_*_predictions.csv"))
    print(f"[score_mtmc] export_dir={args.export_dir}  cam_ids={args.cam_ids}")
    print(f"[score_mtmc] gt_json={args.gt_json}  iou_thr={args.iou_thr}")

    print("[score_mtmc] loading GT ...")
    # Load all GT cameras; pred cam_ids + offset handles the alignment
    all_gt = load_gt(args.gt_json, cam_ids=None)
    print(f"[score_mtmc] GT cameras: {sorted(all_gt.keys())}")
    for cid, df in all_gt.items():
        print(f"  cam {cid}: {len(df)} GT boxes, {df['person_id'].nunique()} unique pids, "
              f"{df['frame'].nunique()} frames")

    print("[score_mtmc] loading predictions ...")
    all_pred = load_preds(args.export_dir, cam_ids=args.cam_ids,
                          cam_offset=args.pred_cam_offset)
    print(f"[score_mtmc] Pred cameras: {sorted(all_pred.keys())}")
    for cid, df in all_pred.items():
        print(f"  cam {cid}: {len(df)} pred boxes, {df['global_id'].nunique()} unique gids, "
              f"{df['frame'].nunique()} frames")

    all_pred = rescale_preds(all_pred, all_gt)

    print("[score_mtmc] computing Global IDF1 ...")
    r = global_idf1(all_gt, all_pred, iou_threshold=args.iou_thr)
    print(f"\nGlobal IDF1 : {r['idf1']:.4f}")
    print(f"  IDTP={r['idtp']}  IDFP={r['idfp']}  IDFN={r['idfn']}")
    print(f"  GT IDs={r['num_gt_ids']}  Pred IDs={r['num_pred_ids']}")


if __name__ == "__main__":
    main()
