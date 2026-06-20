"""
Detection-only evaluation on MMPTracking_short.

Runs two YOLO models (COCO baseline vs fine-tuned MMP) directly via
ONNXRuntime — no DeepStream, no tracker — and reports per-scene
Precision / Recall / F1 at IoU=0.5 (PASCAL VOC style).

Usage:
    python -m src.eval.detect_eval_mmp \\
        --short-root dataset/MMPTracking_short \\
        --scene      lobby_0 \\
        [--max-frames 300]

    # all scenes
    python -m src.eval.detect_eval_mmp \\
        --short-root dataset/MMPTracking_short
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    from ultralytics import YOLO as _UltralyticsYOLO
except ImportError:
    sys.exit("[detect_eval] ultralytics not installed. pip install ultralytics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_THRESH  = 0.25
IOU_MATCH    = 0.50         # IoU threshold for TP/FP/FN counting
PERSON_CLASS = 0            # COCO person class index (for baseline model)

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class YoloModel:
    """Ultralytics YOLO wrapper — supports .pt and .onnx, runs on GPU."""

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self._model  = _UltralyticsYOLO(model_path)
        self._device = device
        # warm-up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self._device,
                            conf=CONF_THRESH, verbose=False)

    def detect_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Batch inference. Returns list of Nx5 arrays [x1,y1,x2,y2,conf],
        one per input frame.
        """
        results = self._model.predict(
            frames,
            device=self._device,
            conf=CONF_THRESH,
            verbose=False,
            stream=False,
        )
        out = []
        is_coco = len(self._model.names) > 1
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                out.append(np.zeros((0, 5), dtype=np.float32))
                continue
            cls  = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            if is_coco:
                mask = cls == PERSON_CLASS
                xyxy, conf = xyxy[mask], conf[mask]
            if len(conf) == 0:
                out.append(np.zeros((0, 5), dtype=np.float32))
            else:
                out.append(np.concatenate([xyxy, conf[:, None]], axis=1
                                          ).astype(np.float32))
        return out


def _iou_pair(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


# ---------------------------------------------------------------------------
# Per-frame matching: greedy IoU matching (TP / FP / FN)
# ---------------------------------------------------------------------------

def match_frame(preds: np.ndarray, gt_xywh: np.ndarray,
                iou_thresh: float = IOU_MATCH):
    """
    preds: Nx4 (xyxy in frame coords)
    gt_xywh: Mx4 (left,top,w,h in frame coords, from GT CSV)
    Returns (tp, fp, fn).
    """
    if len(gt_xywh) == 0:
        return 0, len(preds), 0
    if len(preds) == 0:
        return 0, 0, len(gt_xywh)

    # GT → xyxy
    gt_xyxy = np.stack([
        gt_xywh[:, 0],
        gt_xywh[:, 1],
        gt_xywh[:, 0] + gt_xywh[:, 2],
        gt_xywh[:, 1] + gt_xywh[:, 3],
    ], axis=1)

    matched_gt  = set()
    matched_det = set()
    for di, d in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt_xyxy):
            if gi in matched_gt:
                continue
            iou = _iou_pair(d, g)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh:
            matched_gt.add(best_gi)
            matched_det.add(di)

    tp = len(matched_det)
    fp = len(preds) - tp
    fn = len(gt_xywh) - len(matched_gt)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Scene evaluation
# ---------------------------------------------------------------------------

def _load_gt(gt_csv: Path) -> dict[int, np.ndarray]:
    """Returns {frame_idx: np.ndarray shape (N,4) ltwh}."""
    import pandas as pd
    df = pd.read_csv(gt_csv)
    out: dict[int, list] = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        out.setdefault(f, []).append([row["left"], row["top"],
                                      row["width"], row["height"]])
    return {f: np.array(v, dtype=np.float32) for f, v in out.items()}


_DECODE_SENTINEL = None   # signals decode thread is done

INFER_BATCH_SIZE = 8      # frames per GPU batch
DECODE_QUEUE_MAX = 64     # max buffered frames per camera


def eval_scene(
    scene_dir: Path,
    model: YoloModel,
    max_frames: Optional[int] = None,
    pbar_ext: Optional[object] = None,
) -> dict:
    """
    Evaluate one scene using a producer-consumer pipeline:
      - One decode thread per camera feeds frames into a shared queue.
      - Main thread batches frames from the queue and runs GPU inference.
    """
    cam_files = sorted(scene_dir.glob("cam*.mp4"))
    gt_files  = {int(p.stem.replace("gt_cam", "")): p
                 for p in scene_dir.glob("gt_cam*.csv")}
    if not cam_files:
        return {}

    # Load all GT upfront (fast, CSV)
    gt_by_cam: dict[int, dict[int, np.ndarray]] = {}
    for cam_path in cam_files:
        cid = int(cam_path.stem.replace("cam", ""))
        gt_path = gt_files.get(cid)
        if gt_path:
            gt_by_cam[cid] = _load_gt(gt_path)

    # Count total frames for progress bar
    # use external pbar if provided, else no per-scene bar (avoids output clash)
    pbar = pbar_ext

    n_cams = len(cam_files)

    # Shared queue: items are (cam_id, frame_idx, bgr_frame)
    # Sentinel per camera: (cam_id, -1, None)
    frame_q: queue.Queue = queue.Queue(maxsize=DECODE_QUEUE_MAX * n_cams)

    def _decode_worker(cam_path: Path) -> None:
        cid = int(cam_path.stem.replace("cam", ""))
        if cid not in gt_by_cam:
            frame_q.put((cid, -1, None))
            return
        cap = cv2.VideoCapture(str(cam_path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            frame_q.put((cid, idx, frame))
            idx += 1
        cap.release()
        frame_q.put((cid, -1, None))   # sentinel

    # Start decode threads
    decode_threads = [
        threading.Thread(target=_decode_worker, args=(p,), daemon=True)
        for p in cam_files
    ]
    for t in decode_threads:
        t.start()

    # Main thread: collect batches and run GPU inference
    total_tp = total_fp = total_fn = 0
    total_frames = 0
    t_infer = 0.0
    done_cams = 0

    batch_items: list[tuple[int, int, np.ndarray]] = []  # (cam_id, frame_idx, frame)

    def _flush_batch(items):
        nonlocal total_tp, total_fp, total_fn, total_frames, t_infer
        if not items:
            return
        frames = [it[2] for it in items]
        t0 = time.perf_counter()
        dets_batch = model.detect_batch(frames)
        t_infer += time.perf_counter() - t0
        for (cid, fidx, _), dets in zip(items, dets_batch):
            gt_boxes = gt_by_cam[cid].get(fidx, np.zeros((0, 4)))
            tp, fp, fn = match_frame(dets[:, :4], gt_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_frames += 1
        if pbar:
            pbar.update(len(items))

    while done_cams < len(cam_files):
        try:
            item = frame_q.get(timeout=5.0)
        except queue.Empty:
            continue
        cam_id, frame_idx, frame = item
        if frame is None:
            # sentinel — flush whatever is pending then mark cam done
            _flush_batch(batch_items)
            batch_items = []
            done_cams += 1
            continue
        batch_items.append((cam_id, frame_idx, frame))
        if len(batch_items) >= INFER_BATCH_SIZE:
            _flush_batch(batch_items)
            batch_items = []

    # flush remainder
    _flush_batch(batch_items)

    for t in decode_threads:
        t.join()

    prec = total_tp / (total_tp + total_fp + 1e-9)
    rec  = total_tp / (total_tp + total_fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": prec, "recall": rec, "f1": f1,
        "frames": total_frames,
        "ms_per_frame": (t_infer / total_frames * 1000) if total_frames else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--short-root", default="dataset/MMPTracking_short")
    p.add_argument("--scene", default=None, help="single scene name; omit for all")
    p.add_argument("--max-frames", type=int, default=None,
                   help="limit frames per camera (faster debugging)")
    p.add_argument("--model-baseline",
                   default="models/yolov11/yolo11n.onnx")
    p.add_argument("--model-mmp",
                   default="models/yolov11/yolo11n_mmp.onnx")
    p.add_argument("--iou-thresh", type=float, default=IOU_MATCH)
    p.add_argument("--conf-thresh", type=float, default=CONF_THRESH)
    return p.parse_args()


def _fmt(r: dict) -> str:
    if not r:
        return "  (no data)"
    return (
        f"  Precision={r['precision']:.3f}  Recall={r['recall']:.3f}"
        f"  F1={r['f1']:.3f}"
        f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}"
        f"  ({r['ms_per_frame']:.1f} ms/frame, {r['frames']} frames)"
    )


def main():
    args = _parse_args()
    root = Path(args.short_root)

    global CONF_THRESH, IOU_MATCH
    CONF_THRESH = args.conf_thresh
    IOU_MATCH   = args.iou_thresh

    scenes = [root / args.scene] if args.scene else sorted(root.glob("*_*"))
    scenes = [s for s in scenes if s.is_dir()]
    if not scenes:
        sys.exit(f"[detect_eval] No scenes found in {root}")

    print(f"Loading baseline : {args.model_baseline}")
    m_base = YoloModel(args.model_baseline)
    print(f"Loading MMP model: {args.model_mmp}")
    m_mmp  = YoloModel(args.model_mmp)
    print()

    # count total frames for single global progress bar
    def _count_frames(scene_dir):
        total = 0
        for p in sorted(scene_dir.glob("cam*.mp4")):
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total += min(n, args.max_frames) if args.max_frames else n
        return total

    total_frames_all = sum(_count_frames(s) for s in scenes)
    # x2 because baseline + mmp both process every frame
    pbar = _tqdm(total=total_frames_all * 2, unit="fr",
                 desc="evaluating", dynamic_ncols=True) if _tqdm else None

    def _log(msg: str) -> None:
        if pbar:
            pbar.write(msg)
        else:
            print(msg)

    header = f"{'Scene':<25}  {'Model':<12}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'ms/fr':>7}"
    _log(header)
    _log("-" * len(header))

    agg: dict[str, dict[str, int]] = {"baseline": {}, "mmp": {}}

    def _run(scene_dir, tag, model):
        r = eval_scene(scene_dir, model,
                       max_frames=args.max_frames, pbar_ext=pbar)
        return scene_dir, tag, r

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for scene_dir in scenes:
            futures[pool.submit(_run, scene_dir, "baseline", m_base)] = (scene_dir.name, "baseline")
            futures[pool.submit(_run, scene_dir, "mmp",      m_mmp )] = (scene_dir.name, "mmp")

        pending: dict[str, dict] = {}
        for fut in as_completed(futures):
            scene_dir, tag, r = fut.result()
            pending.setdefault(scene_dir.name, {})[tag] = (scene_dir, r)
            if len(pending[scene_dir.name]) == 2:
                for t in ("baseline", "mmp"):
                    sd, res = pending[scene_dir.name][t]
                    if not res:
                        continue
                    for k in ("tp", "fp", "fn"):
                        agg[t][k] = agg[t].get(k, 0) + res[k]
                    _log(
                        f"{sd.name:<25}  {t:<12}  "
                        f"{res['precision']:>6.3f}  {res['recall']:>6.3f}  {res['f1']:>6.3f}  "
                        f"{res['tp']:>6}  {res['fp']:>6}  {res['fn']:>6}  "
                        f"{res['ms_per_frame']:>7.1f}"
                    )

    if pbar:
        pbar.close()

    if len(scenes) > 1:
        _log("-" * len(header))
        for tag in ("baseline", "mmp"):
            d = agg[tag]
            if not d:
                continue
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            _log(
                f"{'TOTAL':<25}  {tag:<12}  "
                f"{prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  "
                f"{tp:>6}  {fp:>6}  {fn:>6}"
            )


if __name__ == "__main__":
    main()
