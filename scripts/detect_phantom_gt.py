"""
detect_phantom_gt.py — Tìm person_id trong GT không bao giờ được detect.

Mục đích: retail scenes chứa GT annotation cho người đứng sau tủ kệ mà
detector hoàn toàn không thể thấy. Những person_id này tạo ra FN cố định
kéo Global IDF1 xuống ~0.4. Script này so sánh GT vs predictions để tìm
person_id có per-camera recall dưới ngưỡng (mặc định 5%), rồi ghi ra file
để dùng với --exclude-person-ids trong metrics_mmp.py.

Usage:
    python scripts/detect_phantom_gt.py \\
        --short-root dataset/MMPTracking_short \\
        --scene      retail_0 \\
        --pred-dir   output/eval/mmp_retail_0 \\
        --out        output/eval/mmp_retail_0/phantom_ids.txt

    # Sau đó eval lại với GT đã lọc:
    python -m src.eval.metrics_mmp \\
        --short-root dataset/MMPTracking_short \\
        --scene      retail_0 \\
        --pred-dir   output/eval/mmp_retail_0 \\
        --exclude-person-ids output/eval/mmp_retail_0/phantom_ids.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/detect_phantom_gt.py` (not just `python -m`)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# GT resolution
GT_W, GT_H = 640, 360
PRED_W, PRED_H = 1280, 720
MUX_W, MUX_H = 1920, 1080


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


def _infer_pred_space(pred_dir: Path, source_ids: list[int]) -> tuple[float, float]:
    max_right = max_bottom = 0.0
    for sid in source_ids:
        path = pred_dir / f"cam_{sid}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "global_id" in df.columns:
            df = df[df["global_id"] >= 0]
        if df.empty:
            continue
        max_right  = max(max_right,  float((df["left"] + df["width"]).max()))
        max_bottom = max(max_bottom, float((df["top"]  + df["height"]).max()))

    if max_right <= GT_W * 1.25 and max_bottom <= GT_H * 1.25:
        return float(GT_W), float(GT_H)
    if max_right <= PRED_W * 1.25 and max_bottom <= PRED_H * 1.25:
        return float(PRED_W), float(PRED_H)
    return float(MUX_W), float(MUX_H)


def detect_phantom_ids(
    short_root: Path,
    scene: str,
    pred_dir: Path,
    recall_threshold: float,
    iou_threshold: float,
    min_height: float,
    min_width: float,
) -> dict[int, float]:
    """Return {person_id: recall} for all person_ids with recall < recall_threshold."""
    try:
        from src.dataset.mmp_tracking import MMPTrackingShortDataset
    except ImportError:
        sys.exit("[phantom] Cannot import src.dataset.mmp_tracking. "
                 "Run from project root with venv active.")

    ds = MMPTrackingShortDataset(str(short_root), scene)
    cam_ids = ds.get_cam_ids()
    source_ids = list(range(len(cam_ids)))

    pred_w, pred_h = _infer_pred_space(pred_dir, source_ids)
    scale_x = GT_W / pred_w
    scale_y = GT_H / pred_h
    print(f"[phantom] pred-space={pred_w:g}×{pred_h:g}  "
          f"scale=×{scale_x:.4f}/×{scale_y:.4f}")

    # Per-person_id: how many GT frames exist and how many were matched
    gt_frames:   dict[int, int] = {}
    hit_frames:  dict[int, int] = {}

    for source_id, cam_id in enumerate(cam_ids):
        try:
            gt_df = ds.load_gt(cam_id)
        except (FileNotFoundError, ValueError):
            continue

        pred_path = pred_dir / f"cam_{source_id}_predictions.csv"
        if not pred_path.exists():
            print(f"  [cam_{cam_id}] no prediction file — treating all GT as FN")
            for pid in gt_df["person_id"].unique():
                n = int((gt_df["person_id"] == pid).sum())
                gt_frames[pid]  = gt_frames.get(pid, 0)  + n
                hit_frames[pid] = hit_frames.get(pid, 0)
            continue

        pred_df = pd.read_csv(pred_path)
        pred_df = pred_df.rename(columns={"frame_no_cam": "frame"})
        if "global_id" in pred_df.columns:
            pred_df = pred_df[pred_df["global_id"] >= 0]
        pred_df["left"]   = pred_df["left"]   * scale_x
        pred_df["top"]    = pred_df["top"]    * scale_y
        pred_df["width"]  = pred_df["width"]  * scale_x
        pred_df["height"] = pred_df["height"] * scale_y

        # Apply size filter (same defaults as metrics_mmp.py)
        if min_height > 0 or min_width > 0:
            gt_df = gt_df[
                (gt_df["width"] >= min_width) & (gt_df["height"] >= min_height)
            ].reset_index(drop=True)

        all_frames = sorted(set(gt_df["frame"].unique()) | set(pred_df["frame"].unique()))

        for frame in all_frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]

            gt_pids   = g["person_id"].tolist()
            gt_boxes  = g[["left", "top", "width", "height"]].values.astype(float)
            pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

            for pid in gt_pids:
                gt_frames[pid] = gt_frames.get(pid, 0) + 1

            if len(gt_pids) == 0 or len(pred_boxes) == 0:
                continue

            iou = _iou_matrix(gt_boxes, pred_boxes)
            for i, pid in enumerate(gt_pids):
                if iou[i].max() >= iou_threshold:
                    hit_frames[pid] = hit_frames.get(pid, 0) + 1

    # Compute recall per person_id
    low_recall: dict[int, float] = {}
    for pid, total in gt_frames.items():
        hits = hit_frames.get(pid, 0)
        recall = hits / total if total > 0 else 0.0
        if recall < recall_threshold:
            low_recall[pid] = recall

    return low_recall


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find GT person_ids that the detector can never see "
                    "(phantom annotations from permanently occluded persons).")
    p.add_argument("--short-root", default="dataset/MMPTracking_short",
                   help="MMPTracking_short root directory")
    p.add_argument("--scene", required=True,
                   help="Scene name, e.g. retail_0")
    p.add_argument("--pred-dir", required=True,
                   help="Prediction directory (must contain cam_*_predictions.csv)")
    p.add_argument("--out", default=None,
                   help="Output file path (default: <pred-dir>/phantom_ids.txt)")
    p.add_argument("--recall-threshold", type=float, default=0.05,
                   help="Person_ids with per-camera recall below this are considered "
                        "phantom (default: 0.05 = detected in <5%% of their GT frames)")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="IoU threshold for GT-prediction match (default: 0.3, "
                        "lower than eval to catch partially matched detections)")
    p.add_argument("--min-height", type=float, default=20.0,
                   help="Min GT box height in pixels (default: 20)")
    p.add_argument("--min-width", type=float, default=8.0,
                   help="Min GT box width in pixels (default: 8)")
    args = p.parse_args()

    short_root = Path(args.short_root)
    pred_dir   = Path(args.pred_dir)
    out_path   = Path(args.out) if args.out else pred_dir / "phantom_ids.txt"

    print(f"[phantom] scene={args.scene}  pred-dir={pred_dir}")
    print(f"[phantom] recall_threshold={args.recall_threshold}  "
          f"iou_threshold={args.iou_threshold}")

    low_recall = detect_phantom_ids(
        short_root, args.scene, pred_dir,
        recall_threshold=args.recall_threshold,
        iou_threshold=args.iou_threshold,
        min_height=args.min_height,
        min_width=args.min_width,
    )

    sorted_ids = sorted(low_recall.items(), key=lambda x: x[1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Phantom person_ids for scene={args.scene}\n")
        f.write(f"# recall_threshold={args.recall_threshold}  "
                f"iou_threshold={args.iou_threshold}\n")
        f.write(f"# Generated from: {pred_dir}\n")
        f.write(f"# person_id  recall\n")
        for pid, recall in sorted_ids:
            f.write(f"{pid}  # recall={recall:.4f}\n")

    if not low_recall:
        print("[phantom] No phantom person_ids found — all GT persons were detected "
              "at least once.")
        print(f"[phantom] Written empty exclusion list → {out_path}")
        return
    print(f"\n[phantom] Found {len(sorted_ids)} phantom person_ids "
          f"(recall < {args.recall_threshold:.0%}):\n")
    print(f"  {'person_id':>12}  {'recall':>8}  {'note'}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*30}")
    for pid, recall in sorted_ids:
        note = "never detected" if recall == 0.0 else f"seen {recall:.1%} of frames"
        print(f"  {pid:>12}  {recall:>8.3f}  {note}")

    print(f"\n[phantom] Written {len(sorted_ids)} IDs → {out_path}")
    print(f"\n[phantom] To re-eval with clean GT:")
    print(f"  python -m src.eval.metrics_mmp \\")
    print(f"      --short-root {args.short_root} \\")
    print(f"      --scene {args.scene} \\")
    print(f"      --pred-dir {pred_dir} \\")
    print(f"      --exclude-person-ids {out_path}")


if __name__ == "__main__":
    main()
