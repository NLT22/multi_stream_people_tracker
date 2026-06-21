"""Evaluate a ReID ONNX on MMPTracking crop manifests.

Metrics are retrieval-style:

- cross-camera top1: nearest gallery crop from a different camera has same pid
- cross-camera mAP: AP over gallery crops from different cameras

This evaluates embedding quality only. It is not the end-to-end MTMC IDF1 score.
Manifests may use either the exact-cache `cam` column or the older crop-cache
`cam_id` column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


INPUT_H = 256
INPUT_W = 128
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preload_onnxruntime_gpu() -> None:
    """Let ORT find CUDA/cuDNN wheels installed inside the venv."""
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True, msvc=False)
    except Exception as exc:
        print(f"[reid-eval] warning: onnxruntime GPU preload failed: {exc}")


def _read_manifest(
    root: Path,
    split: str,
    max_crops: int | None,
    max_crops_per_scene: int | None,
    max_crops_per_scene_camera: int | None,
) -> list[dict[str, str]]:
    manifest = root / split / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as fh:
        scene_counts: dict[str, int] = {}
        scene_camera_counts: dict[tuple[str, str], int] = {}
        for row in csv.DictReader(fh):
            cam = row.get("cam", row.get("cam_id"))
            if cam is None:
                raise KeyError(f"manifest row missing cam/cam_id: {row}")
            if max_crops_per_scene_camera is not None:
                scene_cam = (row["scene"], cam)
                count = scene_camera_counts.get(scene_cam, 0)
                if count >= max_crops_per_scene_camera:
                    continue
                scene_camera_counts[scene_cam] = count + 1
            if max_crops_per_scene is not None:
                scene = row["scene"]
                count = scene_counts.get(scene, 0)
                if count >= max_crops_per_scene:
                    continue
                scene_counts[scene] = count + 1
            rows.append(row)
            if max_crops is not None and len(rows) >= max_crops:
                break
    if not rows:
        raise SystemExit(f"manifest contains no crops: {manifest}")
    return rows


def _preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return (arr - MEAN) / STD


def _embed(
    rows: list[dict[str, str]],
    root: Path,
    weights: Path,
    batch_size: int,
) -> np.ndarray:
    _preload_onnxruntime_gpu()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    providers = [p for p in providers if p in available]
    session = ort.InferenceSession(str(weights), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[reid-eval] providers={session.get_providers()}")
    print(f"[reid-eval] input={input_name} output={output_name}")

    feats: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = np.stack([_preprocess(root / row["rel_path"]) for row in chunk]).astype(np.float32)
        out = session.run([output_name], {input_name: batch})[0].astype(np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        feats.append(out / np.maximum(norm, 1e-12))
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(f"[reid-eval] embedded {min(start + len(chunk), len(rows))}/{len(rows)}")
    return np.concatenate(feats, axis=0)


def _average_precision(matches: np.ndarray) -> float:
    positives = int(matches.sum())
    if positives == 0:
        return float("nan")
    ranks = np.flatnonzero(matches) + 1
    precision_at_hits = np.arange(1, positives + 1, dtype=np.float32) / ranks
    return float(precision_at_hits.mean())


def _eval_retrieval(
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    max_queries: int | None,
) -> dict[str, float | int]:
    pids = np.asarray([int(r["pid"]) for r in rows], dtype=np.int64)
    cams = np.asarray([int(r.get("cam", r.get("cam_id"))) for r in rows], dtype=np.int64)
    scenes = np.asarray([r["scene"] for r in rows], dtype=object)

    # Only query crops that have at least one same-pid crop in another camera.
    query_idxs = []
    for idx, pid in enumerate(pids):
        valid_pos = (pids == pid) & (cams != cams[idx])
        if valid_pos.any():
            query_idxs.append(idx)
    if max_queries is not None:
        query_idxs = query_idxs[:max_queries]
    if not query_idxs:
        raise SystemExit("no cross-camera ReID queries found")

    top1_hits = 0
    aps: list[float] = []
    scene_top1: dict[str, list[int]] = {}
    scene_ap: dict[str, list[float]] = {}

    for count, qi in enumerate(query_idxs, start=1):
        valid_gallery = cams != cams[qi]
        scores = embeddings[valid_gallery] @ embeddings[qi]
        gallery_pids = pids[valid_gallery]
        order = np.argsort(-scores)
        matches = gallery_pids[order] == pids[qi]

        hit = int(bool(matches[0]))
        ap = _average_precision(matches)
        top1_hits += hit
        aps.append(ap)

        scene = str(scenes[qi])
        scene_top1.setdefault(scene, []).append(hit)
        if not np.isnan(ap):
            scene_ap.setdefault(scene, []).append(ap)

        if count == 1 or count % 1000 == 0:
            print(f"[reid-eval] scored {count}/{len(query_idxs)} queries")

    metrics: dict[str, float | int] = {
        "crops": len(rows),
        "queries": len(query_idxs),
        "cross_camera_top1": top1_hits / len(query_idxs),
        "cross_camera_mAP": float(np.nanmean(aps)),
    }
    for scene in sorted(scene_top1):
        metrics[f"scene/{scene}/top1"] = float(np.mean(scene_top1[scene]))
        metrics[f"scene/{scene}/mAP"] = float(np.mean(scene_ap.get(scene, [float("nan")])))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", default="dataset/mmp_exact_reid")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--weights", default="models/reid/swin_tiny_mmp_reid_all.onnx")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--max-crops-per-scene", type=int, default=None)
    parser.add_argument("--max-crops-per-scene-camera", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.crop_root).resolve()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    rows = _read_manifest(
        root,
        args.split,
        args.max_crops,
        args.max_crops_per_scene,
        args.max_crops_per_scene_camera,
    )
    print(f"[reid-eval] crops={len(rows)} split={args.split} root={root}")
    embeddings = _embed(rows, root, weights, args.batch)
    metrics = _eval_retrieval(rows, embeddings, args.max_queries)

    print("[reid-eval] metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
