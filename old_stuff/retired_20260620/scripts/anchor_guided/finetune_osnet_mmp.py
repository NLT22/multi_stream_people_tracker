#!/usr/bin/env python3
"""Fine-tune the AIC23 authors' OSNet ReID on MMPTracking identities.

Their OSNet (synthetic_reid_model_60_epoch.pth) is trained on AI City synthetic
data => off-domain for MMP. This fine-tunes it on the MMP clean-label ReID crops
(56 identities) so the "fully literal" anchor-guided pipeline gets an in-domain
ReID. Output: checkpoints/osnet_mmp_finetune.pth (backbone weights).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torchreid
torch.load = (lambda f=torch.load: (lambda *a, **k: f(*a, **{**k, "weights_only": False})))()

SYN = "reference/AIC23_Track1_UWIPL_ETRI/deep-person-reid/checkpoints/synthetic_reid_model_60_epoch.pth"


class MMPReid(Dataset):
    def __init__(self, manifest, cap, train=True):
        man = Path(manifest)
        df = pd.read_csv(man)
        if cap:
            idx = []
            for _, g in df.groupby("pid"):
                idx += list(g.sample(min(len(g), cap), random_state=0).index)
            df = df.loc[idx].reset_index(drop=True)
        base = man.parent.parent  # rel_path is "../<cache>/..." relative to cache root
        self.paths = [str((base / p).resolve()) for p in df["rel_path"]]
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
    ap.add_argument("--manifest", default="dataset/MMPTracking_10minute_reid_cache_labeled/train/manifest.csv")
    ap.add_argument("--cap", type=int, default=2000, help="max crops per identity")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="reference/AIC23_Track1_UWIPL_ETRI/deep-person-reid/checkpoints/osnet_mmp_finetune.pth")
    args = ap.parse_args()

    ds = MMPReid(args.manifest, args.cap, train=True)
    print(f"[ft] {len(ds)} crops, {ds.num_pids} identities")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)

    model = torchreid.models.build_model("osnet_x1_0", num_classes=ds.num_pids,
                                         loss="softmax", pretrained=False)
    try:
        from torchreid.utils import load_pretrained_weights
    except ModuleNotFoundError:
        from torchreid.reid.utils import load_pretrained_weights
    load_pretrained_weights(model, SYN)   # backbone init from their synthetic OSNet
    model = model.cuda().train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
              steps_per_epoch=len(dl), epochs=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    nsteps = len(dl)
    for ep in range(args.epochs):
        tot, correct, lsum = 0, 0, 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            lsum += loss.item() * len(y); tot += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            if i % 50 == 0:
                print(f"[ft] ep{ep+1}/{args.epochs} step {i}/{nsteps} "
                      f"loss={lsum/tot:.3f} acc={correct/tot:.3f}", flush=True)
        print(f"[ft] === epoch {ep+1}/{args.epochs} loss={lsum/tot:.3f} "
              f"acc={correct/tot:.3f} ===", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"[ft] saved -> {args.out}")


if __name__ == "__main__":
    main()
