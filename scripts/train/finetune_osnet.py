#!/usr/bin/env python3
"""Fine-tune OSNet (osnet_x1_0) ReID from a Market-1501 pretrain on the MTMC crop cache.

Companion to finetune_reid.py (Swin from ImageNet) — gives a Market-initialized OSNet
to compare against. Reads the same crop cache that scripts/datasets/mtmc_prepare.py emits
(<root>/<split>/manifest.csv with columns scene,pid,cam_id,frame,rel_path).

  python scripts/train/finetune_osnet.py \
      --cache-root dataset/mtmc_reid_cache \
      --market-weights models/reid/pretrained/osnet_x1_0_market1501.pth \
      --epochs 20 --out output/reid_mtmc_osnet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import torchreid
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights

torch.load = (lambda f=torch.load: (lambda *a, **k: f(*a, **{**k, "weights_only": False})))()


class CropCache(Dataset):
    """Reads the mtmc_prepare crop cache (manifest cols: pid,cam_id,rel_path)."""
    def __init__(self, cache_root: Path, split: str, cap: int, train: bool):
        man = cache_root / split / "manifest.csv"
        df = pd.read_csv(man)
        if cap:
            df = pd.concat([g.sample(min(len(g), cap), random_state=0)
                            for _, g in df.groupby("pid")]).reset_index(drop=True)
        self.paths = [str((cache_root / p).resolve()) for p in df["rel_path"]]
        pids = sorted(df["pid"].unique())
        self.remap = {p: i for i, p in enumerate(pids)}
        self.labels = [self.remap[p] for p in df["pid"]]
        self.num_pids = len(pids)
        aug = [T.Resize((256, 128))]
        if train:
            aug += [T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((256, 128))]
        self.tf = T.Compose(aug + [T.ToTensor(),
                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--market-weights", default="models/reid/pretrained/osnet_x1_0_market1501.pth")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cap", type=int, default=2000, help="max crops per identity")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CropCache(args.cache_root, "train", args.cap, train=True)
    print(f"[osnet] {len(ds)} crops, {ds.num_pids} identities")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    model = build_model("osnet_x1_0", num_classes=ds.num_pids, loss="softmax", pretrained=False)
    load_pretrained_weights(model, args.market_weights)   # Market-1501 backbone (classifier skipped)
    model = model.to(dev).train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dl))
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 1e9
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        avg = tot / max(n, 1)
        print(f"[osnet] epoch {ep+1}/{args.epochs}  loss={avg:.4f}", flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.out / "best.pth")

    # export ONNX (eval -> 512-d feature embedding)
    model.load_state_dict(torch.load(args.out / "best.pth", map_location=dev))
    model.eval()
    dummy = torch.randn(1, 3, 256, 128, device=dev)
    onnx_path = args.out / "osnet_x1_0_mtmc_reid.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"],
                      output_names=["features"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}})
    print(f"[osnet] DONE -> {onnx_path}")


if __name__ == "__main__":
    main()
