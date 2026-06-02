"""
Fine-tune Swin-Tiny ReID model on MTA_reid dataset.

Strategy: ID-loss (CrossEntropy on person_id) + Triplet loss (cross-camera hard mining)
Architecture: Swin-Tiny ImageNet pretrained → BNNeck → 256-dim embedding

Pipeline:
  1. Filter MTA_reid images (min_w=25, min_h=50) — removes ~67% tiny/blurry crops
  2. Balance cameras via weighted sampler
  3. Train with combined CE + Triplet loss, 50 epochs
  4. Export to ONNX matching existing nvtracker config (input: [N,3,256,128])

Run:
    python scripts/finetune_reid_mta.py [--epochs 50] [--batch 64] [--output output/reid]

Output:
    output/reid/swin_tiny_mta_reid.pth        — PyTorch checkpoint
    output/reid/swin_tiny_mta_reid.onnx       — ONNX for nvtracker
    output/reid/swin_tiny_mta_reid_engine/    — TRT engine (optional, built separately)
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REID_DIR      = Path("dataset/mta/MTA_reid")
MIN_W         = 25      # filter: min crop width  (pixels)
MIN_H         = 50      # filter: min crop height (pixels)
MIN_IMGS_PID  = 4       # filter: person must have >= this many images after size filter
INPUT_H       = 256     # model input height (nvtracker expects 256x128)
INPUT_W       = 128     # model input width
FEAT_DIM      = 256     # embedding dimension (matches existing nvtracker config)
TRIPLET_MARGIN = 0.3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MtaReidDataset(Dataset):
    """MTA_reid dataset with per-image quality filter."""

    def __init__(self, root: Path, split: str = "train", transform=None,
                 min_w: int = MIN_W, min_h: int = MIN_H,
                 min_imgs_per_pid: int = MIN_IMGS_PID) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []  # (path, pid_idx, cam_id)

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        raw = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))

        # Parse pid and cam from filename: framegta_XXXX_camid_N_pid_M.png
        pid_to_files: dict[int, list[tuple[Path, int]]] = {}
        skipped_size = 0
        for f in raw:
            m_pid = re.search(r"pid_(\d+)", f.name)
            m_cam = re.search(r"camid_(\d+)", f.name)
            if not m_pid or not m_cam:
                continue
            try:
                img = Image.open(f)
                w, h = img.size
            except Exception:
                continue
            if w < min_w or h < min_h:
                skipped_size += 1
                continue
            pid = int(m_pid.group(1))
            cam = int(m_cam.group(1))
            pid_to_files.setdefault(pid, []).append((f, cam))

        # Keep only pids with enough images
        valid_pids = sorted(k for k, v in pid_to_files.items() if len(v) >= min_imgs_per_pid)
        self.pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}
        self.num_classes = len(valid_pids)

        cam_counts: dict[int, int] = {}
        for pid in valid_pids:
            for f, cam in pid_to_files[pid]:
                self.samples.append((f, self.pid_to_idx[pid], cam))
                cam_counts[cam] = cam_counts.get(cam, 0) + 1

        print(f"[reid_data] {split}: {len(raw)} raw → "
              f"{skipped_size} size-filtered → "
              f"{len(self.samples)} kept  "
              f"({self.num_classes} persons)")
        print(f"[reid_data]   Camera dist: " +
              "  ".join(f"cam{c}={n}" for c, n in sorted(cam_counts.items())))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid_idx, cam = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pid_idx, cam

    def make_weights_for_balanced_cameras(self) -> list[float]:
        """Per-sample weight so each camera is sampled equally."""
        cam_counts: dict[int, int] = {}
        for _, _, cam in self.samples:
            cam_counts[cam] = cam_counts.get(cam, 0) + 1
        n_cams = len(cam_counts)
        weights = []
        for _, _, cam in self.samples:
            weights.append(n_cams / cam_counts[cam])
        return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SwinTinyReID(nn.Module):
    """Swin-Tiny backbone + BNNeck, matching the deployed nvtracker config."""

    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM,
                 pretrained: bool = True) -> None:
        super().__init__()
        # swin_tiny_patch4_window7_224: patch=4, window=7 → feature map must be
        # divisible by 7. With input 224×224: 224/4=56 patches, 56%7=0 ✓
        # With input 256×128: 256/4=64, 128/4=32 → 64%7≠0, 32%7≠0 → error.
        # Solution: resize internally to 224×224 in backbone only.
        # Transforms still produce 256×128 (matching nvtracker ONNX input),
        # and we add an Upsample layer before the backbone.
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear",
                                    align_corners=False)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        # Gradient checkpointing reduces VRAM ~40% at cost of ~20% speed
        self.backbone.set_grad_checkpointing(enable=True)
        backbone_dim = self.backbone.num_features  # 768

        # BNNeck: project to feat_dim, normalize for metric learning
        self.neck = nn.Sequential(
            nn.Linear(backbone_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
        )
        nn.init.kaiming_normal_(self.neck[0].weight, mode="fan_out")
        nn.init.constant_(self.neck[1].weight, 1.0)
        nn.init.constant_(self.neck[1].bias, 0.0)

        # ID classifier (only used during training)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.upsample(x)              # [N, 3, 256, 128] → [N, 3, 224, 224]
        feat = self.backbone(x)           # [N, backbone_dim]
        feat = self.neck(feat)            # [N, feat_dim]
        if return_feat:
            return F.normalize(feat, dim=1)
        logits = self.classifier(feat)    # [N, num_classes]
        norm_feat = F.normalize(feat, dim=1)
        return logits, norm_feat


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """Batch-hard triplet loss with cross-camera negative mining."""

    def __init__(self, margin: float = TRIPLET_MARGIN) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor,
                cams: torch.Tensor | None = None) -> torch.Tensor:
        n = feats.size(0)
        dist = torch.cdist(feats, feats)  # [N, N]

        same_pid = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
        diff_pid = ~same_pid

        # For cross-camera mining: prefer hard negatives from different cameras
        if cams is not None:
            diff_cam = cams.unsqueeze(1) != cams.unsqueeze(0)
            # Mask: prefer cross-cam negatives, fall back to any if none available
            cross_cam_neg = diff_pid & diff_cam
            has_cross = cross_cam_neg.any(dim=1, keepdim=True)
            neg_mask = torch.where(has_cross, cross_cam_neg, diff_pid)
        else:
            neg_mask = diff_pid

        # Hard positive: max dist among same pid
        pos_dist = (dist * same_pid.float()).max(dim=1).values
        # Hard negative: min dist among valid negatives
        neg_dist = torch.where(neg_mask, dist, torch.full_like(dist, 1e9)).min(dim=1).values

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((INPUT_H + 16, INPUT_W + 8)),
            transforms.RandomCrop((INPUT_H, INPUT_W)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, ce_loss, triplet_loss,
                    device, epoch, ce_weight=1.0, tri_weight=1.0,
                    scaler=None, accum_steps=1) -> dict:
    model.train()
    total_ce = total_tri = total_loss = n_correct = n_total = 0
    t0 = time.time()
    optimizer.zero_grad()

    for step, (imgs, pids, cams) in enumerate(loader):
        imgs = imgs.to(device)
        pids = pids.to(device)
        cams = cams.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, feats = model(imgs)
            loss_ce  = ce_loss(logits, pids)
            loss_tri = triplet_loss(feats, pids, cams)
            loss     = (ce_weight * loss_ce + tri_weight * loss_tri) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_ce   += loss_ce.item()
        total_tri  += loss_tri.item()
        total_loss += (loss * accum_steps).item()
        n_correct  += (logits.argmax(1) == pids).sum().item()
        n_total    += len(pids)

    n = len(loader)
    return {
        "epoch": epoch,
        "loss":  total_loss / n,
        "ce":    total_ce   / n,
        "tri":   total_tri  / n,
        "acc":   n_correct  / n_total,
        "time":  time.time() - t0,
    }


@torch.no_grad()
def evaluate_cross_cam_sim(model, dataset, device, n_persons=30) -> float:
    """Sample cross-camera pairs and compute mean positive similarity."""
    model.eval()
    tf = make_transforms(train=False)

    # Group by pid and cam
    pid_cam: dict[int, dict[int, list[Path]]] = {}
    for path, pid_idx, cam in dataset.samples:
        pid_cam.setdefault(pid_idx, {}).setdefault(cam, []).append(path)

    # Keep pids visible in >=2 cameras
    multi = {p: cams for p, cams in pid_cam.items() if len(cams) >= 2}
    pids  = random.sample(sorted(multi.keys()), min(n_persons, len(multi)))

    sims = []
    for pid in pids:
        cams = list(multi[pid].keys())
        random.shuffle(cams)
        c1, c2 = cams[0], cams[1]
        f1 = random.choice(multi[pid][c1])
        f2 = random.choice(multi[pid][c2])

        def _emb(path):
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            return model(img, return_feat=True)[0]

        e1, e2 = _emb(f1), _emb(f2)
        sims.append(float((e1 * e2).sum()))

    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: SwinTinyReID, out_path: Path, device) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=device)
    # Use legacy exporter (torch>=2.x defaults to dynamo which needs onnxscript)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy, True),          # return_feat=True → normalized embedding only
            str(out_path),
            input_names=["input"],
            output_names=["fc_pred"],
            dynamic_axes={"input": {0: "batch"}, "fc_pred": {0: "batch"}},
            opset_version=16,
            dynamo=False,
        )
    print(f"[export] ONNX saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune Swin-Tiny ReID on MTA_reid")
    p.add_argument("--reid-dir", default=str(REID_DIR))
    p.add_argument("--output",   default="output/reid")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--batch",    type=int, default=16,
                   help="Batch size per step (default 16 for 4GB VRAM). "
                        "Use --accum-steps to simulate larger batches.")
    p.add_argument("--accum-steps", type=int, default=2,
                   help="Gradient accumulation steps (default 2 → effective batch=32)")
    p.add_argument("--lr",       type=float, default=3.5e-4)
    p.add_argument("--min-w",    type=int, default=MIN_W)
    p.add_argument("--min-h",    type=int, default=MIN_H)
    p.add_argument("--min-imgs-pid", type=int, default=MIN_IMGS_PID)
    p.add_argument("--ce-weight",    type=float, default=1.0)
    p.add_argument("--tri-weight",   type=float, default=1.0)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume",   default=None, metavar="CKPT")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds = MtaReidDataset(
        Path(args.reid_dir), split="train",
        transform=make_transforms(train=True),
        min_w=args.min_w, min_h=args.min_h,
        min_imgs_per_pid=args.min_imgs_pid,
    )
    weights  = train_ds.make_weights_for_balanced_cameras()
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader   = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SwinTinyReID(
        num_classes=train_ds.num_classes,
        feat_dim=FEAT_DIM,
        pretrained=not args.no_pretrained,
    ).to(device)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train] Resumed from {args.resume} (epoch {start_epoch})")

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    # Lower LR for backbone, higher for neck+classifier
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": list(model.neck.parameters()) +
                   list(model.classifier.parameters()), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    ce_loss      = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss = TripletLoss(margin=TRIPLET_MARGIN)

    # Mixed precision scaler (FP16) — saves ~40% VRAM
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    eff_batch = args.batch * args.accum_steps

    print(f"\n[train] device={device}  persons={train_ds.num_classes}"
          f"  samples={len(train_ds)}")
    print(f"[train] batch={args.batch}  accum={args.accum_steps}"
          f"  effective_batch={eff_batch}  epochs={args.epochs}")
    print(f"[train] Filter: min_w={args.min_w} min_h={args.min_h}"
          f"  min_imgs_pid={args.min_imgs_pid}")
    print(f"[train] Loss: CE={args.ce_weight} Triplet={args.tri_weight}"
          f"  FP16={'yes' if scaler else 'no'}  grad_ckpt=yes\n")

    best_sim = -1.0

    for epoch in range(start_epoch, args.epochs + 1):
        stats = train_one_epoch(
            model, loader, optimizer, ce_loss, triplet_loss, device, epoch,
            ce_weight=args.ce_weight, tri_weight=args.tri_weight,
            scaler=scaler, accum_steps=args.accum_steps,
        )
        scheduler.step()

        cross_sim = evaluate_cross_cam_sim(model, train_ds, device)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
              f"tri={stats['tri']:.4f}  acc={stats['acc']:.3f}  "
              f"cross_cam_sim={cross_sim:.3f}  "
              f"({stats['time']:.0f}s)")

        # Save checkpoint
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "cross_cam_sim": cross_sim, "args": vars(args)}
        torch.save(ckpt, out_dir / "last.pth")

        if cross_sim > best_sim:
            best_sim = cross_sim
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  ✓ New best cross-cam sim: {best_sim:.3f}")

    # ── Export best model ────────────────────────────────────────────────────
    print(f"\n[export] Loading best model (cross_cam_sim={best_sim:.3f})")
    best_ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    onnx_path = out_dir / "swin_tiny_mta_reid.onnx"
    export_onnx(model, onnx_path, device)

    # Also save weights-only for easy loading
    torch.save(model.state_dict(), out_dir / "swin_tiny_mta_reid_weights.pth")

    print(f"\n[done] Output dir: {out_dir}")
    print(f"  Checkpoint: {out_dir}/best.pth")
    print(f"  ONNX:       {onnx_path}")
    print(f"\nTo use with nvtracker, update tracker config:")
    print(f"  onnxFile: \"{onnx_path.resolve()}\"")
    print(f"  modelEngineFile: \"\" # leave empty to rebuild TRT engine")
    print(f"  reidFeatureSize: {FEAT_DIM}")


if __name__ == "__main__":
    main()
