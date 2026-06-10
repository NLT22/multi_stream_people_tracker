"""
Fine-tune Swin-Tiny ReID on MMPTracking_short dataset.

Data pipeline:
  - For each scene × camera × frame: crop person bounding boxes from GT
  - Sample every --sample-rate frames (default 5 → 5fps)
  - Min crop size filter: --min-w, --min-h
  - Global person ID = scene_idx * 1000 + local_pid (keeps scene identities separate)
  - Train scenes: first N-1 scenes per environment; val = last scene per env

Architecture: same Swin-Tiny + BNNeck as MTA script — compatible ONNX output.

Run (fast start, fine-tune from MTA model):
    python scripts/finetune_reid_mmp.py \\
        --resume output/reid_v2/best.pth \\
        --epochs 40 --pk-p 24 --pk-k 4

Run (from ImageNet pretrained):
    python scripts/finetune_reid_mmp.py --epochs 60

Output:
    output/reid_mmp/best.pth
    output/reid_mmp/swin_tiny_mmp_reid.onnx   ← drop into nvtracker config
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import timm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_H       = 256
INPUT_W       = 128
FEAT_DIM      = 256
TRIPLET_MARGIN = 0.3

# Environments and their scenes (same as mmp_to_yolo.py)
ENVS = {
    "cafe_shop":       ["cafe_shop_0","cafe_shop_1","cafe_shop_2","cafe_shop_3"],
    "industry_safety": ["industry_safety_0","industry_safety_1","industry_safety_2",
                        "industry_safety_3","industry_safety_4"],
    "lobby":           ["lobby_0","lobby_1","lobby_2","lobby_3"],
    "office":          ["office_0","office_1","office_2"],
    "retail":          ["retail_0","retail_1","retail_2","retail_3",
                        "retail_4","retail_5","retail_6","retail_7"],
}


def _train_val_scenes() -> tuple[list[str], list[str]]:
    train, val = [], []
    for scenes in ENVS.values():
        val.append(scenes[-1])
        train.extend(scenes[:-1])
    return train, val


def _resolve_short_root(short_root: Path, scenes: list[str]) -> Path:
    if any((short_root / scene).exists() for scene in scenes):
        return short_root

    print(f"[ERROR] No known MMPTracking_short scenes found under: {short_root}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Dataset — crop persons directly from video frames
# ---------------------------------------------------------------------------

class MMPReidDataset(Dataset):
    """
    Crops person bounding boxes from MMPTracking_short GT for ReID training.

    Samples every `sample_rate` frames.
    Global person ID = scene_global_offset + local_pid so identities from
    different scenes never collide.
    """

    def __init__(
        self,
        short_root: Path,
        scenes: list[str],
        transform=None,
        sample_rate: int = 5,
        min_w: int = 20,
        min_h: int = 40,
        min_imgs_per_pid: int = 4,
        split_name: str = "data",
        prefer_clean_gt: bool = False,
    ) -> None:
        self.transform = transform
        # (frame_array_or_path, global_pid, cam_id)
        self.samples: list[tuple] = []

        # Map global_pid → list[sample_idx]
        self._pid_to_idxs: dict[int, list[int]] = {}

        total_raw = total_kept = 0
        scene_pid_to_global: dict[tuple[str, int], int] = {}

        for scene in scenes:
            scene_dir = short_root / scene
            if not scene_dir.exists():
                print(f"  [SKIP] scene not found: {scene_dir}")
                continue
            # Build crop list: load video, decode every sample_rate-th frame
            import pandas as pd

            found_csv = False
            for csv_path in sorted(scene_dir.glob("gt_cam*.csv")):
                stem_tail = csv_path.stem.replace("gt_cam", "")
                if not stem_tail.isdigit():
                    continue
                cam_id = int(stem_tail)
                if prefer_clean_gt:
                    clean_path = scene_dir / f"gt_cam{cam_id}_clean.csv"
                    if clean_path.exists():
                        csv_path = clean_path
                found_csv = True
                vid_path = scene_dir / f"cam{cam_id}.mp4"
                if not vid_path.exists():
                    print(f"  [WARN] video not found: {vid_path}")
                    continue

                df = pd.read_csv(csv_path)

                # Group boxes by frame for fast lookup
                frame_boxes: dict[int, list[tuple]] = {}
                for _, row in df.iterrows():
                    f = int(row["frame"])
                    frame_boxes.setdefault(f, []).append((
                        int(row["person_id"]),
                        float(row["left"]), float(row["top"]),
                        float(row["width"]), float(row["height"]),
                    ))

                # Build global pid map for this scene
                local_to_global: dict[int, int] = {}

                cap = cv2.VideoCapture(str(vid_path))
                frame_no = 0
                while True:
                    ret, frame_img = cap.read()
                    if not ret:
                        break
                    if frame_no % sample_rate == 0 and frame_no in frame_boxes:
                        for (local_pid, x1, y1, w, h) in frame_boxes[frame_no]:
                            total_raw += 1
                            if w < min_w or h < min_h:
                                continue
                            # Clamp to frame
                            x1c = max(0, int(x1))
                            y1c = max(0, int(y1))
                            x2c = min(frame_img.shape[1], int(x1 + w))
                            y2c = min(frame_img.shape[0], int(y1 + h))
                            if x2c <= x1c or y2c <= y1c:
                                continue

                            crop = frame_img[y1c:y2c, x1c:x2c]
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                            # Assign global pid
                            if local_pid not in local_to_global:
                                key = (scene, local_pid)
                                if key not in scene_pid_to_global:
                                    scene_pid_to_global[key] = len(scene_pid_to_global)
                                local_to_global[local_pid] = scene_pid_to_global[key]
                            gid = local_to_global[local_pid]

                            idx = len(self.samples)
                            self.samples.append((crop_rgb, gid, cam_id))
                            self._pid_to_idxs.setdefault(gid, []).append(idx)
                            total_kept += 1
                    frame_no += 1
                cap.release()
            if not found_csv:
                print(f"  [WARN] no gt_cam*.csv files found in: {scene_dir}")

        # Filter: keep only pids with enough samples
        valid_pids = sorted(
            pid for pid, idxs in self._pid_to_idxs.items()
            if len(idxs) >= min_imgs_per_pid
        )
        valid_set = set(valid_pids)
        keep_idxs = [i for i, (_, gid, _) in enumerate(self.samples) if gid in valid_set]

        filtered_samples = [self.samples[i] for i in keep_idxs]
        self.samples = filtered_samples

        # Re-map global pids → compact 0-based class indices
        self.pid_to_cls = {pid: i for i, pid in enumerate(valid_pids)}
        self.num_classes = len(valid_pids)

        # Rebuild pid→idxs with new indices
        self._pid_to_idxs = {}
        for i, (_, gid, _) in enumerate(self.samples):
            cls = self.pid_to_cls[gid]
            self._pid_to_idxs.setdefault(cls, []).append(i)

        print(f"[reid_data:{split_name}] {len(scenes)} scenes: "
              f"{total_raw} raw crops → {total_kept} size-filtered → "
              f"{len(self.samples)} kept "
              f"({self.num_classes} persons, min_imgs={min_imgs_per_pid})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        crop_rgb, gid, cam_id = self.samples[idx]
        img = Image.fromarray(crop_rgb)
        if self.transform:
            img = self.transform(img)
        cls = self.pid_to_cls[gid]
        return img, cls, cam_id

    def get_pid_indices(self) -> dict[int, list[int]]:
        return self._pid_to_idxs


# ---------------------------------------------------------------------------
# P×K Sampler
# ---------------------------------------------------------------------------

class PKSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: MMPReidDataset,
        p: int,
        k: int,
        batches_per_epoch: int,
    ) -> None:
        self.pid_to_idxs = dataset.get_pid_indices()
        self.pids = list(self.pid_to_idxs.keys())
        self.p = p
        self.k = k
        if batches_per_epoch <= 0:
            batch_size = max(1, p * k)
            batches_per_epoch = math.ceil(len(dataset) / batch_size)
        self.num_batches = max(1, batches_per_epoch)

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k

    def __iter__(self):
        indices = []
        for _ in range(self.num_batches):
            batch_pids = random.sample(self.pids, min(self.p, len(self.pids)))
            for pid in batch_pids:
                pool = self.pid_to_idxs[pid]
                chosen = (random.sample(pool, self.k) if len(pool) >= self.k
                          else random.choices(pool, k=self.k))
                indices.extend(chosen)
        return iter(indices)


# ---------------------------------------------------------------------------
# Model (identical to finetune_reid_mta.py — same ONNX format)
# ---------------------------------------------------------------------------

class SwinTinyReID(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear",
                                    align_corners=False)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        backbone_dim = self.backbone.num_features  # 768
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
        feat = self.backbone(x)
        feat = self.neck(feat)
        if return_feat:
            return F.normalize(feat, dim=1)
        logits = self.classifier(feat)
        return logits, F.normalize(feat, dim=1)

    def replace_classifier(self, num_classes: int) -> None:
        """Swap out classifier head when fine-tuning from a different dataset."""
        feat_dim = self.classifier.in_features
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.01)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    def __init__(self, margin: float = TRIPLET_MARGIN) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor,
                cams: torch.Tensor | None = None) -> torch.Tensor:
        dist = torch.cdist(feats, feats)
        same_pid = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff_pid = ~same_pid
        if cams is not None:
            diff_cam = cams.unsqueeze(1) != cams.unsqueeze(0)
            cross_cam_neg = diff_pid & diff_cam
            has_cross = cross_cam_neg.any(dim=1, keepdim=True)
            neg_mask = torch.where(has_cross, cross_cam_neg, diff_pid)
        else:
            neg_mask = diff_pid
        pos_dist = (dist * same_pid.float()).max(dim=1).values
        neg_dist = torch.where(neg_mask, dist,
                               torch.full_like(dist, 1e9)).min(dim=1).values
        return F.relu(pos_dist - neg_dist + self.margin).mean()


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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, ce_loss, triplet_loss,
                    device, epoch, scaler=None, accum_steps=1) -> dict:
    model.train()
    total_ce = total_tri = total_loss = n_correct = n_total = 0
    t0 = time.time()
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch",
                dynamic_ncols=True, leave=False)
    for step, (imgs, pids, cams) in enumerate(pbar):
        imgs = imgs.to(device)
        pids = pids.to(device)
        cams = cams.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, feats = model(imgs)
            loss_ce  = ce_loss(logits, pids)
            loss_tri = triplet_loss(feats, pids, cams)
            loss     = (loss_ce + loss_tri) / accum_steps

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
        pbar.set_postfix(
            loss=f"{total_loss/(step+1):.3f}",
            tri=f"{total_tri/(step+1):.3f}",
            acc=f"{n_correct/n_total:.3f}",
        )
    pbar.close()
    n = max(1, len(loader))
    return {
        "epoch": epoch, "loss": total_loss/n, "ce": total_ce/n,
        "tri": total_tri/n, "acc": n_correct/max(1, n_total),
        "time": time.time() - t0,
    }


@torch.no_grad()
def evaluate_cross_cam_sim(model, dataset: MMPReidDataset, device,
                           n_persons: int = 80) -> dict:
    """Mean same-pid cross-camera similarity vs mean diff-pid similarity."""
    model.eval()
    tf = make_transforms(train=False)

    # Group by (pid_cls, cam)
    pid_cam: dict[int, dict[int, list[int]]] = {}
    for i, (_, gid, cam) in enumerate(dataset.samples):
        cls = dataset.pid_to_cls[gid]
        pid_cam.setdefault(cls, {}).setdefault(cam, []).append(i)

    multi = {p: cams for p, cams in pid_cam.items() if len(cams) >= 2}
    if len(multi) < 2:
        return {"pos_mean": 0.0, "neg_mean": 0.0, "gap": -999.0, "pairs": 0}

    sample_pids = random.sample(sorted(multi.keys()), min(n_persons, len(multi)))

    def _emb(idx):
        crop_rgb, _, _ = dataset.samples[idx]
        img = tf(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        return model(img, return_feat=True)[0]

    pos_sims, neg_sims = [], []
    emb_cache: dict[int, torch.Tensor] = {}

    for pid in sample_pids:
        cams = list(multi[pid].keys())
        random.shuffle(cams)
        c1, c2 = cams[0], cams[1]
        i1 = random.choice(multi[pid][c1])
        i2 = random.choice(multi[pid][c2])
        e1 = emb_cache.setdefault(i1, _emb(i1))
        e2 = emb_cache.setdefault(i2, _emb(i2))
        pos_sims.append(float((e1 * e2).sum()))

    # Negative pairs: random pairs from different pids
    all_pids = list(multi.keys())
    for _ in range(len(sample_pids)):
        pa, pb = random.sample(all_pids, 2)
        cam_a = random.choice(list(multi[pa].keys()))
        cam_b = random.choice(list(multi[pb].keys()))
        ia = random.choice(multi[pa][cam_a])
        ib = random.choice(multi[pb][cam_b])
        ea = emb_cache.setdefault(ia, _emb(ia))
        eb = emb_cache.setdefault(ib, _emb(ib))
        neg_sims.append(float((ea * eb).sum()))

    return {
        "pos_mean": float(np.mean(pos_sims)),
        "neg_mean": float(np.mean(neg_sims)),
        "gap":      float(np.mean(pos_sims) - np.mean(neg_sims)),
        "pairs":    len(pos_sims),
    }


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

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
            export_model, dummy, str(out_path),
            input_names=["input"], output_names=["features"],
            dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
            opset_version=16, dynamo=False,
        )
    print(f"[export] ONNX saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune Swin-Tiny ReID on MMPTracking_short")
    p.add_argument("--short-root", default="dataset/MMPTracking_short")
    p.add_argument("--output",     default="output/reid_mmp")
    p.add_argument("--epochs",     type=int, default=40)
    p.add_argument("--pk-p",       type=int, default=24,
                   help="P in P×K sampler: persons per batch (default 24)")
    p.add_argument("--pk-k",       type=int, default=4,
                   help="K in P×K sampler: images per person (default 4 → batch=96)")
    p.add_argument("--accum-steps", type=int, default=2,
                   help="Gradient accumulation (default 2 → eff. batch=192)")
    p.add_argument("--lr",          type=float, default=3.5e-4)
    p.add_argument("--sample-rate", type=int, default=5,
                   help="Sample every Nth frame from video (default 5)")
    p.add_argument("--min-w",       type=int, default=20)
    p.add_argument("--min-h",       type=int, default=40)
    p.add_argument("--min-imgs-pid", type=int, default=4)
    p.add_argument("--early-stop",  type=int, default=8)
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--batches-per-epoch", type=int, default=200,
                   help="PK batches per epoch. Use 0 to cover roughly all crops once "
                        "(default 200; old behavior was only num_persons // P batches).")
    p.add_argument("--grad-ckpt",   action="store_true")
    p.add_argument("--train-all-nonretail", action="store_true",
                   help="Train on ALL non-retail scenes (incl. the usual val "
                        "scenes). Legitimate for a fixed-camera deployment where "
                        "the target scenes are known. Val monitors the same set.")
    p.add_argument("--train-all", action="store_true",
                   help="Train on ALL scenes including retail.")
    p.add_argument("--scan-root", default=None, metavar="DIR",
                   help="Use a dataset laid out as DIR/{train,val}/<scene>/cam*.mp4 "
                        "(e.g. MMPTracking_10minute). Overrides --short-root and the "
                        "built-in scene split with the on-disk train/val dirs.")
    p.add_argument("--exclude-retail", action="store_true",
                   help="With --scan-root: drop scene dirs whose name contains "
                        "'retail' from both train and val.")
    p.add_argument("--prefer-clean-gt", action="store_true",
                   help="Use gt_cam<N>_clean.csv instead of gt_cam<N>.csv when available.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume",      default=None, metavar="CKPT",
                   help="Resume from checkpoint (.pth). If from MTA model, "
                        "the classifier head is replaced automatically.")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_root:
        root = Path(args.scan_root)

        def _scan(split: str) -> list[str]:
            d = root / split
            names = sorted(s.name for s in d.iterdir() if s.is_dir()) if d.exists() else []
            if args.exclude_retail:
                names = [n for n in names if "retail" not in n]
            return [f"{split}/{n}" for n in names]

        train_scenes, val_scenes = _scan("train"), _scan("val")
        short_root = root
        print(f"[reid] scan-root: {root}"
              f"{' (excluding retail)' if args.exclude_retail else ''}")
    else:
        if args.train_all:
            all_scenes = [s for scenes in ENVS.values() for s in scenes]
            train_scenes, val_scenes = all_scenes, list(all_scenes)
            print("[reid] train-all: training on every scene including retail")
        elif args.train_all_nonretail:
            nonretail = [s for env, scenes in ENVS.items() if env != "retail"
                         for s in scenes]
            train_scenes, val_scenes = nonretail, list(nonretail)
            print("[reid] train-all-nonretail: training on every non-retail scene")
        else:
            train_scenes, val_scenes = _train_val_scenes()
        short_root = _resolve_short_root(Path(args.short_root),
                                         train_scenes + val_scenes)
    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val   scenes ({len(val_scenes)}):   {val_scenes}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds = MMPReidDataset(
        short_root, train_scenes,
        transform=make_transforms(train=True),
        sample_rate=args.sample_rate,
        min_w=args.min_w, min_h=args.min_h,
        min_imgs_per_pid=args.min_imgs_pid,
        split_name="train",
        prefer_clean_gt=args.prefer_clean_gt,
    )

    if train_ds.num_classes == 0:
        print("[ERROR] No valid persons found. Lower --min-h/--min-w or check dataset.")
        raise SystemExit(1)

    val_ds = MMPReidDataset(
        short_root, val_scenes,
        transform=None,
        sample_rate=args.sample_rate,
        min_w=args.min_w, min_h=args.min_h,
        min_imgs_per_pid=args.min_imgs_pid,
        split_name="val",
        prefer_clean_gt=args.prefer_clean_gt,
    )
    if val_ds.num_classes == 0:
        print("[ERROR] No valid validation persons found. Check MMPTracking_short val scenes.")
        raise SystemExit(1)

    batch_sz  = args.pk_p * args.pk_k
    sampler   = PKSampler(
        train_ds,
        p=args.pk_p,
        k=args.pk_k,
        batches_per_epoch=args.batches_per_epoch,
    )
    loader    = DataLoader(train_ds, batch_size=batch_sz, sampler=sampler,
                           num_workers=args.workers, pin_memory=True,
                           drop_last=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SwinTinyReID(
        num_classes=train_ds.num_classes,
        feat_dim=FEAT_DIM,
        pretrained=not args.no_pretrained,
    ).to(device)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        # Try to load; if classifier shape mismatches, replace head then reload
        try:
            model.load_state_dict(state, strict=True)
            print(f"[train] Loaded checkpoint: {args.resume}")
        except RuntimeError:
            # Head size mismatch (different number of classes) — replace classifier
            # and drop the old classifier weights (strict=False does NOT ignore
            # size mismatches, only missing/unexpected keys).
            model.replace_classifier(train_ds.num_classes)
            model.to(device)
            state = {k: v for k, v in state.items()
                     if not k.startswith("classifier.")}
            model.load_state_dict(state, strict=False)
            print(f"[train] Loaded checkpoint (head dropped+replaced for {train_ds.num_classes} classes): {args.resume}")
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"[train] Resuming from epoch {start_epoch}")

    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(enable=True)
        print("[train] Gradient checkpointing: ENABLED")

    # Differential LR: lower for backbone
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
    scaler       = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n[train] device={device}"
          f"  train_persons={train_ds.num_classes}  train_samples={len(train_ds)}"
          f"  val_persons={val_ds.num_classes}  val_samples={len(val_ds)}")
    print(f"[train] PK: P={args.pk_p} K={args.pk_k} batch={batch_sz}"
          f"  accum={args.accum_steps}  eff_batch={batch_sz*args.accum_steps}")
    print(f"[train] batches/epoch={len(loader)}"
          f"  sampled_crops/epoch={len(loader) * batch_sz}")
    print(f"[train] epochs={args.epochs}  early_stop={args.early_stop}"
          f"  FP16={'yes' if scaler else 'no'}\n")

    best_gap   = -999.0
    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        stats = train_one_epoch(
            model, loader, optimizer, ce_loss, triplet_loss,
            device, epoch, scaler=scaler, accum_steps=args.accum_steps,
        )
        scheduler.step()

        train_sim = evaluate_cross_cam_sim(model, train_ds, device)
        val_sim = evaluate_cross_cam_sim(model, val_ds, device)
        improved  = val_sim["gap"] > best_gap
        no_improve = 0 if improved else no_improve + 1

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
              f"tri={stats['tri']:.4f}  acc={stats['acc']:.3f}  "
              f"train_gap={train_sim['gap']:.3f}  "
              f"val_pos={val_sim['pos_mean']:.3f}  val_neg={val_sim['neg_mean']:.3f}  "
              f"val_gap={val_sim['gap']:.3f}  "
              f"no_imp={no_improve}/{args.early_stop}  "
              f"({stats['time']:.0f}s)")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "train_sim_stats": train_sim,
            "val_sim_stats": val_sim,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pth")
        if improved:
            best_gap = val_sim["gap"]
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  ✓ New best val_gap: {best_gap:.3f}  "
                  f"(pos={val_sim['pos_mean']:.3f}  neg={val_sim['neg_mean']:.3f})")

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"\n[train] Early stop.")
            break

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"\n[export] Loading best model (val_gap={best_gap:.3f})")
    best_ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    onnx_path = out_dir / "swin_tiny_mmp_reid.onnx"
    export_onnx(model, onnx_path, device)
    torch.save(model.state_dict(), out_dir / "swin_tiny_mmp_reid_weights.pth")

    print(f"\n[done] Output: {out_dir}")
    print(f"  ONNX: {onnx_path}")
    print(f"\nTo use in nvtracker config, set:")
    print(f"  onnxFile: \"{onnx_path.resolve()}\"")
    print(f"  reidFeatureSize: {FEAT_DIM}")


if __name__ == "__main__":
    main()
