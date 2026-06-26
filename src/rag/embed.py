"""Embed a person crop with the DEPLOYED Swin-Tiny ReID ONNX (RAG Route B).

Uses onnxruntime + the exact SGIE preprocessing from
`configs/models/nvinfer_reid_swin_sgie_all.yml` (RGB, 128x256, ImageNet-style
normalisation: scale*(pixel - offset)), so a query crop is embedded in the same
space as the gallery embeddings persisted by ingest.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ONNX = REPO / "models/reid/swin_tiny_mmp_reid_all.onnx"
SCALE = 0.01735207
OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB means
IN_W, IN_H = 128, 256


@lru_cache(maxsize=1)
def _session(onnx: str):
    import onnxruntime as ort
    return ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])


def embed_crop(image_path: str | Path, onnx: str | Path = ONNX) -> np.ndarray:
    """Return an L2-normalised 256-d embedding for a person-crop image."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read crop image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = SCALE * (img - OFFSETS)              # HWC
    x = np.transpose(x, (2, 0, 1))[None]     # 1,C,H,W
    sess = _session(str(onnx))
    out = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    emb = np.asarray(out, dtype=np.float32).ravel()
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb
