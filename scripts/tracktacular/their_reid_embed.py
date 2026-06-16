#!/usr/bin/env python3
"""Extract DENSE per-detection embeddings with the AIC23 winner's own ReID model
(OSNet, synthetic_reid_model_60_epoch.pth) on MMPTracking crops.

This is the "their code, their model" test: replace our sparse (16.8%) in-tracker
ReID embeddings with their dense per-crop OSNet embeddings, then run the anchor
clustering (offline_anchor_faithful). Writes detection_embeddings.npz in the same
format the faithful pipeline reads.

Crops come from cam{N}.mp4 frames using the boxes in cam_*_predictions.csv (640x360).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# torch>=2.6 defaults weights_only=True; their checkpoint has numpy globals.
# Force full load (trusted file we downloaded from the authors).
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

CKPT = "reference/AIC23_Track1_UWIPL_ETRI/deep-person-reid/checkpoints/synthetic_reid_model_60_epoch.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/train")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    try:
        from torchreid.utils import FeatureExtractor
    except ModuleNotFoundError:
        from torchreid.reid.utils import FeatureExtractor
    from src.dataset.mmp_tracking import MMPTrackingShortDataset

    ext = FeatureExtractor(model_name="osnet_x1_0", model_path=CKPT,
                           device="cuda", image_size=(256, 128))
    ds = MMPTrackingShortDataset(str(args.short_root), args.scene)
    cam_ids = ds.get_cam_ids()                      # source_id i -> cam_ids[i]
    pred_dir = Path(args.pred_dir)
    scene_dir = Path(args.short_root) / args.scene

    all_cam, all_frame, all_ltid, all_emb = [], [], [], []
    for src, cam in enumerate(cam_ids):
        csv = pred_dir / f"cam_{src}_predictions.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        by_frame = {f: g for f, g in df.groupby("frame_no_cam")}
        cap = cv2.VideoCapture(str(scene_dir / f"cam{cam}.mp4"))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fidx = -1
        crops, meta = [], []

        def flush():
            if not crops:
                return
            feats = ext(crops).cpu().numpy()       # (n,512)
            for (fr, lt), fe in zip(meta, feats):
                all_cam.append(src); all_frame.append(fr); all_ltid.append(lt)
                all_emb.append(fe)
            crops.clear(); meta.clear()

        while True:
            ok, im = cap.read()
            if not ok:
                break
            fidx += 1
            g = by_frame.get(fidx)
            if g is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for r in g.itertuples():
                x1 = max(0, int(r.left)); y1 = max(0, int(r.top))
                x2 = min(W, int(r.left + r.width)); y2 = min(H, int(r.top + r.height))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crops.append(rgb[y1:y2, x1:x2])
                meta.append((fidx, int(r.local_track_id)))
                if len(crops) >= args.batch:
                    flush()
        flush()
        cap.release()
        print(f"  cam{cam} (src{src}): {sum(1 for c in all_cam if c==src)} crops")

    emb = np.asarray(all_emb, dtype=np.float32)
    np.savez_compressed(pred_dir / "detection_embeddings.npz",
                        cam_id=np.asarray(all_cam, np.int64),
                        frame_no=np.asarray(all_frame, np.int64),
                        local_track_id=np.asarray(all_ltid, np.int64),
                        embeddings=emb)
    print(f"[their-reid] wrote {len(emb)} dense embeddings ({emb.shape[1]}-d) "
          f"-> {pred_dir}/detection_embeddings.npz")


if __name__ == "__main__":
    main()
