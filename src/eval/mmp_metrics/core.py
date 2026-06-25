"""MMPTracking_short metric engine — MOTA / IDF1 / Global-IDF1 (no CLI).

Pure evaluation functions used by mmp_metrics/cli.py and the backward-compat
entry src/eval/metrics_mmp.py. Importable for unit testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
except ImportError:
    sys.exit("[eval] motmetrics not found.  pip install motmetrics")

try:
    import trackeval  # noqa: F401
    _TRACKEVAL_AVAILABLE = True
except ImportError:
    _TRACKEVAL_AVAILABLE = False

from src.dataset.mmp_tracking import MMPTrackingShortDataset

# GT resolution
GT_W, GT_H = 640, 360

# Default pipeline export space.
#
# In the normal two-probe path the exporter runs post-tiler, after
# nvmultistreamtiler has scaled each source into a tile.  The gallery subtracts
# the tile offset before writing CSV, so prediction boxes are tile-local
# 1280x720 by default, while MMPTracking_short GT is source-frame 640x360.
#
# If you export with different tile dimensions, pass --pred-width/--pred-height
# or leave them unset to auto-infer the most likely space from the CSV extents.
PRED_W, PRED_H = 1280, 720
MUX_W, MUX_H = 1920, 1080
SCALE_X = GT_W / PRED_W
SCALE_Y = GT_H / PRED_H

# Default difficulty filter (GT space, 640×360)
_DEFAULT_MIN_HEIGHT     = 20.0
_DEFAULT_MIN_WIDTH      = 8.0
_DEFAULT_MIN_VIS        = 0.30



def _load_pred(
    pred_dir: Path,
    source_id: int,
    scale_x: float,
    scale_y: float,
) -> pd.DataFrame:
    """Load cam_<source_id>_predictions.csv và scale tọa độ về GT space."""
    path = pred_dir / f"cam_{source_id}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"frame_no_cam": "frame"})
    # global_id=-1 means unassigned — exclude from evaluation
    df = df[df["global_id"] >= 0].reset_index(drop=True)
    df["left"]   = df["left"]   * scale_x
    df["top"]    = df["top"]    * scale_y
    df["width"]  = df["width"]  * scale_x
    df["height"] = df["height"] * scale_y
    return df


def _infer_pred_space(pred_dir: Path, source_ids: list[int]) -> tuple[float, float]:
    """Infer source/tiler/mux prediction space from bbox extents."""
    max_right = 0.0
    max_bottom = 0.0
    for source_id in source_ids:
        path = pred_dir / f"cam_{source_id}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "global_id" in df.columns:
            df = df[df["global_id"] >= 0]
        if df.empty:
            continue
        max_right = max(max_right, float((df["left"] + df["width"]).max()))
        max_bottom = max(max_bottom, float((df["top"] + df["height"]).max()))

    if max_right <= GT_W * 1.25 and max_bottom <= GT_H * 1.25:
        return float(GT_W), float(GT_H)
    if max_right <= PRED_W * 1.25 and max_bottom <= PRED_H * 1.25:
        return float(PRED_W), float(PRED_H)
    return float(MUX_W), float(MUX_H)


def _load_exclude_ids(path: str | None) -> set:
    """Load a set of person_ids to exclude from GT evaluation.

    File format: one integer per line (comments with # are ignored).
    Used to remove phantom annotations (persons permanently occluded by
    shelves/walls that the detector can never see).
    """
    if path is None:
        return set()
    excluded: set = set()
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                excluded.add(int(line))
    return excluded


def _filter_boxes(
    df: pd.DataFrame,
    min_height: float,
    min_width: float,
    min_visibility: float,
) -> pd.DataFrame:
    w = df["width"].values
    h = df["height"].values
    l = df["left"].values
    t = df["top"].values

    size_ok = (w >= min_width) & (h >= min_height)

    ix1 = np.clip(l,     0, GT_W)
    iy1 = np.clip(t,     0, GT_H)
    ix2 = np.clip(l + w, 0, GT_W)
    iy2 = np.clip(t + h, 0, GT_H)
    vis_area  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    total_area = np.maximum(1, w * h)
    vis_ok = (vis_area / total_area) >= min_visibility

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

        gt_ids    = g["person_id"].tolist()
        gt_boxes  = g[["left", "top", "width", "height"]].values.astype(float)
        pred_ids  = p["global_id"].tolist()
        pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        if len(gt_ids) > 0 and len(pred_ids) > 0:
            dists = 1.0 - _iou_matrix(gt_boxes, pred_boxes)
            dists[dists > 1 - iou_threshold] = np.nan
        else:
            dists = np.empty((len(gt_ids), len(pred_ids)))

        acc.update(gt_ids, pred_ids, dists)
    return acc


def _eval_global_idf1(
    all_gt: dict,
    all_pred: dict,
    iou_threshold: float,
) -> dict:
    """Cross-camera IDF1 bằng linear assignment person_id → global_id.

    all_gt / all_pred: key có thể là int (cam_id) hoặc str (scene_cam).
    """
    from scipy.optimize import linear_sum_assignment

    hits: dict = {}
    gt_det_count:   dict = {}
    pred_det_count: dict = {}

    for key in all_gt:
        gt_df   = all_gt[key]
        pred_df = all_pred.get(key)
        if pred_df is None:
            continue

        all_frames = sorted(
            set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())
        )
        for frame in all_frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]

            gt_pids    = g["person_id"].tolist()
            gt_boxes   = g[["left", "top", "width", "height"]].values.astype(float)

            # global_id=-1 already filtered in _load_pred
            pred_gids = [int(g) for g in p["global_id"].tolist()]

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
                    gid = pred_gids[c]
                    hits.setdefault(pid, {})
                    hits[pid][gid] = hits[pid].get(gid, 0) + 1

    all_pids = sorted(gt_det_count.keys(), key=str)
    all_gids = sorted(pred_det_count.keys(), key=str)

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
# HOTA via trackeval
# ---------------------------------------------------------------------------

def _eval_hota_camera(gt_df: pd.DataFrame, pred_df: pd.DataFrame,
                      pred_id_col: str = "global_id") -> dict | None:
    """Per-camera HOTA (and its DetA/AssA sub-metrics) via trackeval.
    Returns means over the alpha thresholds, or None if trackeval is missing."""
    if not _TRACKEVAL_AVAILABLE:
        return None
    from trackeval.metrics import HOTA as HOTAMetric

    all_frames = sorted(set(gt_df["frame"].unique()) | set(pred_df["frame"].unique()))
    gt_id_map: dict = {}
    pred_id_map: dict = {}

    def _remap(val, id_map):
        if val not in id_map:
            id_map[val] = len(id_map)
        return id_map[val]

    data: dict = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [],
                  "num_gt_dets": 0, "num_tracker_dets": 0}
    for frame in all_frames:
        g = gt_df[gt_df["frame"] == frame]
        p = pred_df[pred_df["frame"] == frame]
        gt_ids_int = [_remap(x, gt_id_map) for x in g["person_id"].tolist()]
        pred_ids_int = [_remap(x, pred_id_map) for x in p[pred_id_col].tolist()]
        gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)
        pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)
        data["gt_ids"].append(np.array(gt_ids_int, dtype=int))
        data["tracker_ids"].append(np.array(pred_ids_int, dtype=int))
        data["similarity_scores"].append(_iou_matrix(gt_boxes, pred_boxes))
        data["num_gt_dets"] += len(gt_ids_int)
        data["num_tracker_dets"] += len(pred_ids_int)
    data["num_gt_ids"] = len(gt_id_map)
    data["num_tracker_ids"] = len(pred_id_map)

    try:
        result = HOTAMetric().eval_sequence(data)
        return {k: float(np.mean(v)) for k, v in result.items()
                if isinstance(v, np.ndarray)}
    except Exception as exc:
        print(f"  [eval] HOTA failed: {exc}")
        return None


def compute_per_camera_metrics(all_gt: dict, all_pred: dict,
                               iou_threshold: float = 0.5,
                               pred_id_col: str = "global_id",
                               with_hota: bool = True) -> list[dict]:
    """Per-camera MOTA/MOTP/IDF1/IDS/Frag (+ optional HOTA/DetA/AssA).

    all_gt / all_pred: {cam_id: dataframe}. Pred frames use `pred_id_col` as the
    tracker id; default 'global_id' so metrics reflect the system's end-to-end
    identity output (same id space as Global IDF1)."""
    mh = mm.metrics.create()
    mot_names = ["num_frames", "mota", "motp", "idf1", "num_switches",
                 "num_fragmentations", "num_misses", "num_false_positives",
                 "precision", "recall"]
    rows: list[dict] = []
    for cam_id in sorted(all_gt):
        gt_df = all_gt[cam_id]
        pred_df = all_pred.get(cam_id)
        if pred_df is None:
            continue
        # motmetrics accumulator uses pred_id_col as the hypothesis id
        acc = mm.MOTAccumulator(auto_id=True)
        frames = sorted(set(gt_df["frame"].unique()) | set(pred_df["frame"].unique()))
        for frame in frames:
            g = gt_df[gt_df["frame"] == frame]
            p = pred_df[pred_df["frame"] == frame]
            gt_ids = g["person_id"].tolist()
            pred_ids = p[pred_id_col].tolist()
            gt_boxes = g[["left", "top", "width", "height"]].values.astype(float)
            pred_boxes = p[["left", "top", "width", "height"]].values.astype(float)
            if len(gt_ids) and len(pred_ids):
                dists = 1.0 - _iou_matrix(gt_boxes, pred_boxes)
                dists[dists > 1 - iou_threshold] = np.nan
            else:
                dists = np.empty((len(gt_ids), len(pred_ids)))
            acc.update(gt_ids, pred_ids, dists)
        summ = mh.compute(acc, metrics=mot_names, name=f"cam_{cam_id}")
        row = {"camera": int(cam_id)}
        for m in mot_names:
            row[m] = float(summ[m].iloc[0])
        if with_hota:
            h = _eval_hota_camera(gt_df, pred_df, pred_id_col=pred_id_col)
            if h is not None:
                for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr"):
                    if k in h:
                        row[k.lower()] = h[k]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def _eval_scene(
    scene: str,
    short_root: Path,
    pred_dir: Path,
    iou_threshold: float,
    min_height: float,
    min_width: float,
    min_visibility: float,
    cameras: list[int] | None,
    pred_width: float | None,
    pred_height: float | None,
    exclude_person_ids: set | None = None,
    gt_suffix: str = "",
) -> dict:
    try:
        ds = MMPTrackingShortDataset(str(short_root), scene, gt_suffix=gt_suffix)
    except FileNotFoundError as e:
        print(f"[{scene}] ERROR: {e}")
        return {}

    # Thứ tự get_cam_ids() khớp với thứ tự source_id trong pipeline
    real_cam_ids = cameras if cameras else ds.get_cam_ids()
    source_to_cam = {i: c for i, c in enumerate(real_cam_ids)}
    if pred_width is None or pred_height is None:
        pred_width, pred_height = _infer_pred_space(
            pred_dir, list(source_to_cam.keys()))
    scale_x = GT_W / pred_width
    scale_y = GT_H / pred_height

    all_gt:       dict[int, pd.DataFrame] = {}
    all_pred:     dict[int, pd.DataFrame] = {}
    per_cam_accs: dict[int, mm.MOTAccumulator] = {}

    print(f"\n[{scene}] cameras={real_cam_ids}  "
          f"(source_id 0→cam{real_cam_ids[0]} ... "
          f"{len(real_cam_ids)-1}→cam{real_cam_ids[-1]})")
    print(f"  pred-space={pred_width:g}×{pred_height:g}  "
          f"scale=×{scale_x:.4f}/×{scale_y:.4f}")

    for source_id, cam_id in source_to_cam.items():
        try:
            gt_df = ds.load_gt(cam_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [cam_{cam_id}] GT not found: {e} — skipping")
            continue

        try:
            pred_df = _load_pred(pred_dir, source_id, scale_x, scale_y)
        except FileNotFoundError as e:
            print(f"  [cam_{cam_id}] {e} — skipping")
            continue

        if exclude_person_ids:
            before = len(gt_df)
            gt_df = gt_df[~gt_df["person_id"].isin(exclude_person_ids)].reset_index(drop=True)
            removed = before - len(gt_df)
            if removed:
                print(f"  [cam_{cam_id}] Excluded {removed} GT rows "
                      f"({len(exclude_person_ids)} phantom person_ids)")

        if min_height > 0 or min_width > 0 or min_visibility > 0:
            gt_raw, pred_raw = len(gt_df), len(pred_df)
            gt_df   = _filter_boxes(gt_df,   min_height, min_width, min_visibility)
            pred_df = _filter_boxes(pred_df, min_height, min_width, min_visibility)
            print(f"  [cam_{cam_id}] GT {gt_raw}→{len(gt_df)}  "
                  f"Pred {pred_raw}→{len(pred_df)}")
        else:
            print(f"  [cam_{cam_id}] GT={len(gt_df)}  Pred={len(pred_df)}")

        all_gt[cam_id]   = gt_df
        all_pred[cam_id] = pred_df
        per_cam_accs[cam_id] = _eval_camera_motmetrics(gt_df, pred_df, iou_threshold)

    return {
        "per_cam_accs": per_cam_accs,
        "all_gt":       all_gt,
        "all_pred":     all_pred,
        "iou_threshold": iou_threshold,
    }


def _print_scene_summary(scene: str, result: dict) -> None:
    if not result or not result["per_cam_accs"]:
        print(f"[{scene}] No cameras evaluated.")
        return

    per_cam_accs = result["per_cam_accs"]
    all_gt       = result["all_gt"]
    all_pred     = result["all_pred"]
    iou_threshold = result["iou_threshold"]

    mh = mm.metrics.create()
    metric_names = ["num_frames", "mota", "motp", "idf1",
                    "num_switches", "num_fragmentations",
                    "num_misses", "num_false_positives",
                    "precision", "recall"]
    summary = mh.compute_many(
        list(per_cam_accs.values()),
        metrics=metric_names,
        names=[f"cam_{c}" for c in per_cam_accs],
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
        g = _eval_global_idf1(all_gt, all_pred, iou_threshold)
        print(f"  Global IDF1: {g['idf1']:.4f}  "
              f"(IDTP={g['idtp']}  IDFP={g['idfp']}  IDFN={g['idfn']}  "
              f"GT IDs={g['num_gt_ids']}  Pred IDs={g['num_pred_ids']})")


def _eval_scene_job(params: tuple) -> tuple[str, dict]:
    """Process-pool wrapper for scene-level parallel evaluation."""
    (
        scene,
        short_root,
        pred_dir,
        iou_threshold,
        min_height,
        min_width,
        min_visibility,
        cameras,
        pred_width,
        pred_height,
        exclude_person_ids,
        gt_suffix,
    ) = params
    result = _eval_scene(
        scene,
        Path(short_root),
        Path(pred_dir),
        iou_threshold,
        min_height,
        min_width,
        min_visibility,
        cameras,
        pred_width,
        pred_height,
        exclude_person_ids=exclude_person_ids,
        gt_suffix=gt_suffix,
    )
    return scene, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

