#!/usr/bin/env python3
"""Dense per-detection embeddings with the SWIN ReID model (256-d), torch+CUDA,
same I/O as their_reid_embed.py (OSNet). Lets us test using ONE ReID model (Swin)
for BOTH the in-tracker SCT *and* the cross-camera anchor stage — the production
ideal of extracting the embedding once and reusing it (NVIDIA MDX pattern).

Uses torch on GPU (onnxruntime in this env is CPU-only -> 100x slower).

  python scripts/anchor_guided/swin_reid_embed.py \
      --pred-dir output/eval/heldout_64pm_office_0 \
      --out-dir  output/eval/swin_anchor_64pm_office_0 \
      --short-root dataset/MMPTracking_10minute/val --scene 64pm_office_0
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_H, _W = 256, 128
WEIGHTS = "output/reid_10min_labeled_ssd/swin_tiny_mmp_reid_weights.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute/val")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--weights", default=WEIGHTS, help="SwinTinyReID state_dict (.pth)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from scripts.train.finetune_reid_mmp import SwinTinyReID
    from src.dataset.mmp_tracking import MMPTrackingShortDataset

    model = SwinTinyReID(num_classes=70)        # classifier size irrelevant (return_feat)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}  # feats only
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[swin-reid] loaded {args.weights} (missing={len(missing)} unexpected={len(unexpected)})")
    model = model.cuda().eval().half()           # FP16 — ~2x faster, negligible accuracy change
    mean, std = _MEAN.cuda(), _STD.cuda()

    ds = MMPTrackingShortDataset(str(args.short_root), args.scene)
    cam_ids = ds.get_cam_ids()
    pred_dir = Path(args.pred_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(args.short_root) / args.scene

    @torch.no_grad()
    def embed(crops):
        t = torch.empty(len(crops), 3, _H, _W)
        for i, c in enumerate(crops):
            im = cv2.cvtColor(cv2.resize(c, (_W, _H)), cv2.COLOR_BGR2RGB)
            t[i] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        t = ((t.cuda() - mean) / std).half()       # FP16 input to match the FP16 model
        f = model(t, return_feat=True)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        return f.cpu().numpy()

    all_cam, all_frame, all_ltid, all_emb = [], [], [], []
    for src, cam in enumerate(cam_ids):
        csv = pred_dir / f"cam_{src}_predictions.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        by_frame = {f: g for f, g in df.groupby("frame_no_cam")}
        cap = cv2.VideoCapture(str(scene_dir / f"cam{cam}.mp4"))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        crops, meta, fidx = [], [], -1

        def flush():
            if not crops:
                return
            for (fr, lt), fe in zip(meta, embed(crops)):
                all_cam.append(src); all_frame.append(fr); all_ltid.append(lt); all_emb.append(fe)
            crops.clear(); meta.clear()

        while True:
            ok, im = cap.read()
            if not ok:
                break
            fidx += 1
            g = by_frame.get(fidx)
            if g is None:
                continue
            for r in g.itertuples():
                x1 = max(0, int(r.left)); y1 = max(0, int(r.top))
                x2 = min(W, int(r.left + r.width)); y2 = min(H, int(r.top + r.height))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crops.append(im[y1:y2, x1:x2]); meta.append((fidx, int(r.local_track_id)))
                if len(crops) >= args.batch:
                    flush()
        flush(); cap.release()
        print(f"  cam{cam} (src{src}): {sum(1 for c in all_cam if c==src)} crops")

    emb = np.asarray(all_emb, dtype=np.float32)
    np.savez_compressed(out_dir / "detection_embeddings.npz",
                        cam_id=np.asarray(all_cam, np.int64), frame_no=np.asarray(all_frame, np.int64),
                        local_track_id=np.asarray(all_ltid, np.int64), embeddings=emb)
    for f in pred_dir.glob("cam_*_predictions.csv"):
        shutil.copy2(f, out_dir / f.name)
    for f in ("tracklet_bev.csv", "tracklets.csv"):
        if (pred_dir / f).exists():
            shutil.copy2(pred_dir / f, out_dir / f)
    print(f"[swin-reid] wrote {len(emb)} dense embeddings ({emb.shape[1]}-d) -> {out_dir}/detection_embeddings.npz")


if __name__ == "__main__":
    main()
