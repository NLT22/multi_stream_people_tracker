#!/usr/bin/env python3
"""Cross-camera retrieval eval on the MTMC ReID val crops — compares ReID ONNX models.

For each model: extract embeddings on a balanced sample of the val crop cache, then for
every query crop rank all crops from OTHER cameras (same warehouse) by cosine; report
cross-camera top-1 and mAP. Same preprocessing (resize 256x128 + ImageNet norm) for all.

  python scripts/eval/eval_reid_mtmc.py --cache-root dataset/mtmc_reid_cache \
      --models swin=output/reid_mtmc_swin/swin_tiny_mmp_reid.onnx \
               osnet=output/reid_mtmc_osnet/osnet_x1_0_mtmc_reid.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load(path, H=256, W=128):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return rgb.transpose(2, 0, 1)


def _embed(onnx_path, imgs, batch=128):
    s = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    out = []
    for i in range(0, len(imgs), batch):
        b = np.stack(imgs[i:i + batch]).astype(np.float32)
        e = s.run(None, {iname: b})[0]
        out.append(e)
    e = np.concatenate(out, 0)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="+", required=True, help="name=path.onnx ...")
    ap.add_argument("--per-pid-cam", type=int, default=4, help="max crops per (pid,cam)")
    args = ap.parse_args()

    df = pd.read_csv(args.cache_root / args.split / "manifest.csv")
    # balanced sample: cap crops per (pid, cam_id) so retrieval isn't dominated by long tracks
    df = pd.concat([g.sample(min(len(g), args.per_pid_cam), random_state=0)
                    for _, g in df.groupby(["pid", "cam_id"])]).reset_index(drop=True)
    paths = [str((args.cache_root / p).resolve()) for p in df["rel_path"]]
    imgs, keep = [], []
    for i, p in enumerate(paths):
        a = _load(p)
        if a is not None:
            imgs.append(a); keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    pid = df["pid"].to_numpy(); cam = df["cam_id"].to_numpy()
    print(f"[eval] {len(imgs)} crops, {len(np.unique(pid))} ids, {len(np.unique(cam))} cams")

    for spec in args.models:
        name, path = spec.split("=", 1)
        emb = _embed(path, imgs)
        sim = emb @ emb.T
        np.fill_diagonal(sim, -2)
        top1 = aps = n = 0
        for q in range(len(emb)):
            gal = cam != cam[q]            # cross-camera gallery
            if gal.sum() == 0:
                continue
            order = np.argsort(-sim[q][gal])
            gpid = pid[gal][order]
            rel = (gpid == pid[q]).astype(np.float32)
            if rel.sum() == 0:
                continue
            top1 += rel[0]
            cum = np.cumsum(rel)
            prec = cum / (np.arange(len(rel)) + 1)
            aps += (prec * rel).sum() / rel.sum()
            n += 1
        print(f"  {name:6s}  cross-camera top1={top1/n:.4f}  mAP={aps/n:.4f}  (dim {emb.shape[1]}, {n} queries)")


if __name__ == "__main__":
    main()
