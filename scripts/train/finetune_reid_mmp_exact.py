"""Fine-tune Swin-Tiny ReID from exact MMPTracking zip-derived crops.

Build crops first with:

  python scripts/datasets/mmp_exact_to_reid.py --output-dir dataset/mmp_exact_reid

This trainer intentionally reads the exact-source crop cache manifest, not the
older extracted MMPTracking_10minute MP4/CSV cache.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm
import timm


INPUT_H = 256
INPUT_W = 128
FEAT_DIM = 256
TRIPLET_MARGIN = 0.3


@dataclass(frozen=True)
class CropSample:
    image_path: Path
    pid: int
    cam: int
    scene: str


class ExactReidCropDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None, max_crops_per_pid: int | None = None) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        manifest = root / split / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"missing ReID crop manifest: {manifest}")

        rows_by_pid: dict[int, list[CropSample]] = {}
        with manifest.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                pid = int(row["pid"])
                sample = CropSample(
                    image_path=root / row["rel_path"],
                    pid=pid,
                    cam=int(row["cam"]),
                    scene=row["scene"],
                )
                rows_by_pid.setdefault(pid, []).append(sample)

        samples: list[CropSample] = []
        for pid in sorted(rows_by_pid):
            items = rows_by_pid[pid]
            if max_crops_per_pid is not None and len(items) > max_crops_per_pid:
                rng = random.Random(pid)
                items = rng.sample(items, max_crops_per_pid)
            samples.extend(items)

        raw_pids = sorted({s.pid for s in samples})
        self.pid_to_cls = {pid: idx for idx, pid in enumerate(raw_pids)}
        self.samples = samples
        self._pid_to_idxs: dict[int, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            cls = self.pid_to_cls[sample.pid]
            self._pid_to_idxs.setdefault(cls, []).append(idx)
        self.num_classes = len(raw_pids)

        print(
            f"[reid-exact:{split}] {len(self.samples)} crops, "
            f"{self.num_classes} identities from {manifest}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.pid_to_cls[sample.pid], sample.cam

    def get_pid_indices(self) -> dict[int, list[int]]:
        return self._pid_to_idxs

    def load_image(self, idx: int) -> Image.Image:
        return Image.open(self.samples[idx].image_path).convert("RGB")


class PKSampler(Sampler[int]):
    def __init__(self, dataset: ExactReidCropDataset, p: int, k: int, batches_per_epoch: int) -> None:
        self.pid_to_idxs = dataset.get_pid_indices()
        self.pids = list(self.pid_to_idxs)
        self.p = p
        self.k = k
        if batches_per_epoch <= 0:
            batches_per_epoch = math.ceil(len(dataset) / max(1, p * k))
        self.num_batches = max(1, batches_per_epoch)

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k

    def __iter__(self):
        indices: list[int] = []
        for _ in range(self.num_batches):
            chosen_pids = random.sample(self.pids, min(self.p, len(self.pids)))
            for pid in chosen_pids:
                pool = self.pid_to_idxs[pid]
                if len(pool) >= self.k:
                    indices.extend(random.sample(pool, self.k))
                else:
                    indices.extend(random.choices(pool, k=self.k))
        return iter(indices)


class SwinTinyReID(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM, pretrained: bool = True) -> None:
        super().__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        backbone_dim = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
        )
        nn.init.kaiming_normal_(self.neck[0].weight, mode="fan_out")
        nn.init.constant_(self.neck[1].weight, 1.0)
        nn.init.constant_(self.neck[1].bias, 0.0)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.upsample(x)
        feat = self.neck(self.backbone(x))
        norm_feat = F.normalize(feat, dim=1)
        if return_feat:
            return norm_feat
        return self.classifier(feat), norm_feat


class TripletLoss(nn.Module):
    def __init__(self, margin: float = TRIPLET_MARGIN) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, cams: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(feats, feats)
        same_pid = labels[:, None] == labels[None, :]
        diff_pid = ~same_pid
        diff_cam = cams[:, None] != cams[None, :]
        cross_cam_neg = diff_pid & diff_cam
        has_cross = cross_cam_neg.any(dim=1, keepdim=True)
        neg_mask = torch.where(has_cross, cross_cam_neg, diff_pid)
        pos_dist = (dist * same_pid.float()).max(dim=1).values
        neg_dist = torch.where(neg_mask, dist, torch.full_like(dist, 1e9)).min(dim=1).values
        return F.relu(pos_dist - neg_dist + self.margin).mean()


def make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((INPUT_H + 16, INPUT_W + 8)),
                transforms.RandomCrop((INPUT_H, INPUT_W)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((INPUT_H, INPUT_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def train_one_epoch(model, loader, optimizer, ce_loss, triplet_loss, device, epoch, scaler, accum_steps: int) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_ce = total_tri = correct = total = 0.0
    t0 = time.time()
    pbar = tqdm(loader, desc=f"epoch {epoch}", unit="batch", dynamic_ncols=True)
    for step, (images, pids, cams) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)
        cams = cams.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, feats = model(images)
            loss_ce = ce_loss(logits, pids)
            loss_tri = triplet_loss(feats, pids, cams)
            loss = (loss_ce + loss_tri) / accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float((loss * accum_steps).detach().cpu())
        total_ce += float(loss_ce.detach().cpu())
        total_tri += float(loss_tri.detach().cpu())
        correct += int((logits.argmax(1) == pids).sum())
        total += len(pids)
        pbar.set_postfix(loss=f"{total_loss/step:.3f}", tri=f"{total_tri/step:.3f}", acc=f"{correct/max(1,total):.3f}")
    return {
        "loss": total_loss / max(1, len(loader)),
        "ce": total_ce / max(1, len(loader)),
        "tri": total_tri / max(1, len(loader)),
        "acc": correct / max(1, total),
        "time": time.time() - t0,
    }


@torch.no_grad()
def evaluate_gap(model, dataset: ExactReidCropDataset, device, n_persons: int = 80) -> dict:
    model.eval()
    tf = make_transforms(train=False)
    pid_cam: dict[int, dict[int, list[int]]] = {}
    for idx, sample in enumerate(dataset.samples):
        cls = dataset.pid_to_cls[sample.pid]
        pid_cam.setdefault(cls, {}).setdefault(sample.cam, []).append(idx)
    multi = {pid: cams for pid, cams in pid_cam.items() if len(cams) >= 2}
    if len(multi) < 2:
        return {"pos_mean": 0.0, "neg_mean": 0.0, "gap": -999.0, "pairs": 0}
    sample_pids = random.sample(sorted(multi), min(n_persons, len(multi)))
    emb_cache: dict[int, torch.Tensor] = {}

    def emb(idx: int) -> torch.Tensor:
        cached = emb_cache.get(idx)
        if cached is not None:
            return cached
        image = tf(dataset.load_image(idx)).unsqueeze(0).to(device)
        out = model(image, return_feat=True)[0]
        emb_cache[idx] = out
        return out

    pos_sims: list[float] = []
    neg_sims: list[float] = []
    for pid in sample_pids:
        cams = list(multi[pid])
        c1, c2 = random.sample(cams, 2)
        e1 = emb(random.choice(multi[pid][c1]))
        e2 = emb(random.choice(multi[pid][c2]))
        pos_sims.append(float((e1 * e2).sum()))
    all_pids = list(multi)
    for _ in sample_pids:
        pa, pb = random.sample(all_pids, 2)
        ca = random.choice(list(multi[pa]))
        cb = random.choice(list(multi[pb]))
        ea = emb(random.choice(multi[pa][ca]))
        eb = emb(random.choice(multi[pb][cb]))
        neg_sims.append(float((ea * eb).sum()))
    return {
        "pos_mean": float(np.mean(pos_sims)),
        "neg_mean": float(np.mean(neg_sims)),
        "gap": float(np.mean(pos_sims) - np.mean(neg_sims)),
        "pairs": len(pos_sims),
    }


def export_onnx(model: SwinTinyReID, out_path: Path, device) -> None:
    class FeatureExport(nn.Module):
        def __init__(self, reid_model: SwinTinyReID) -> None:
            super().__init__()
            self.reid_model = reid_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.reid_model(x, return_feat=True)

    export_model = FeatureExport(model).eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=device)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
            opset_version=16,
            dynamo=False,
        )
    print(f"[export] onnx={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", default="dataset/mmp_exact_reid")
    parser.add_argument("--output", default="output/reid_mmp_exact")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pk-p", type=int, default=24)
    parser.add_argument("--pk-k", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--early-stop", type=int, default=6)
    parser.add_argument("--max-crops-per-pid", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_root = Path(args.crop_root)

    train_ds = ExactReidCropDataset(
        crop_root,
        "train",
        transform=make_transforms(train=True),
        max_crops_per_pid=args.max_crops_per_pid,
    )
    val_ds = ExactReidCropDataset(
        crop_root,
        "val",
        transform=None,
        max_crops_per_pid=args.max_crops_per_pid,
    )
    if train_ds.num_classes <= 1 or val_ds.num_classes <= 1:
        raise SystemExit("need at least two train and val identities")

    sampler = PKSampler(train_ds, args.pk_p, args.pk_k, args.batches_per_epoch)
    batch_size = args.pk_p * args.pk_k
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )

    model = SwinTinyReID(
        num_classes=train_ds.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
            {"params": list(model.neck.parameters()) + list(model.classifier.parameters()), "lr": args.lr},
        ],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss = TripletLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"[train] device={device} fp16={bool(scaler)}")
    print(f"[train] train={len(train_ds)} crops/{train_ds.num_classes} ids val={len(val_ds)} crops/{val_ds.num_classes} ids")
    print(f"[train] P={args.pk_p} K={args.pk_k} batch={batch_size} accum={args.accum_steps} batches/epoch={len(loader)}")

    best_gap = -999.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, ce_loss, triplet_loss, device, epoch, scaler, args.accum_steps)
        scheduler.step()
        train_gap = evaluate_gap(model, train_ds, device)
        val_gap = evaluate_gap(model, val_ds, device)
        improved = val_gap["gap"] > best_gap
        no_improve = 0 if improved else no_improve + 1
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={stats['loss']:.4f} ce={stats['ce']:.4f} tri={stats['tri']:.4f} "
            f"acc={stats['acc']:.3f} train_gap={train_gap['gap']:.3f} "
            f"val_pos={val_gap['pos_mean']:.3f} val_neg={val_gap['neg_mean']:.3f} "
            f"val_gap={val_gap['gap']:.3f} no_imp={no_improve}/{args.early_stop} "
            f"time={stats['time']:.0f}s"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "train_gap": train_gap,
            "val_gap": val_gap,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pth")
        if improved:
            best_gap = val_gap["gap"]
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  best val_gap={best_gap:.3f}")
        if args.early_stop > 0 and no_improve >= args.early_stop:
            print("[train] early stop")
            break

    best_ckpt = torch.load(out_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    torch.save(model.state_dict(), out_dir / "swin_tiny_mmp_exact_reid_weights.pth")
    if not args.no_export:
        export_onnx(model, out_dir / "swin_tiny_mmp_exact_reid.onnx", device)
    print(f"[done] output={out_dir}")


if __name__ == "__main__":
    main()
