"""
Fine-tune YOLO11n on MMPTracking_short and export ONNX for DeepStream.

Two-step workflow:
  1. Build YOLO dataset:   python scripts/datasets/mmp_to_yolo.py
  2. Train + export ONNX:  python scripts/train/train_yolo_mmp.py

Run:
    python scripts/train/train_yolo_mmp.py \\
        [--data dataset/mmp_yolo/dataset.yaml] \\
        [--epochs 30] [--batch 16] [--imgsz 640] [--device 0] \\
        [--weights yolo11n.pt]   # or output/train/yolo11n_mta/weights/best.pt
        [--resume]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


ONNX_DEST = Path("models/yolov11/yolo11n_mmp.onnx")

HINT = """
Next — run in DeepStream:

  python -m src.main \\
      --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \\
      --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \\
      --export-predictions output/eval/mmp_lobby0 \\
      --no-sync --no-display

Then evaluate:
  python -m src.eval.metrics_mmp \\
      --short-root dataset/MMPTracking_short \\
      --scene lobby_0 \\
      --pred-dir output/eval/mmp_lobby0
"""


def _is_writable_dir(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK)
    existing_parent = path.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    return os.access(existing_parent, os.W_OK)


def _fail_if_unwritable(path: Path, description: str) -> None:
    if _is_writable_dir(path):
        return
    print(f"[ERROR] {description} is not writable: {path}")
    print("This is usually caused by an earlier `sudo docker compose run` creating root-owned files.")
    print("Fix ownership from the repo root:")
    print("  sudo chown -R $USER:$USER output dataset/mmp_yolo models/yolov11")
    print("Or choose a user-writable run directory, e.g.:")
    print("  python scripts/train/train_yolo_mmp.py --project runs/train --name yolo11n_mmp")
    raise SystemExit(1)


def _resolve_yolo_data_yaml(data_path: Path) -> Path:
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = Path(data.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    train_rel = Path(data.get("train", "images/train"))
    val_rel = Path(data.get("val", "images/val"))
    train_path = train_rel if train_rel.is_absolute() else root / train_rel
    val_path = val_rel if val_rel.is_absolute() else root / val_rel
    if train_path.exists() and val_path.exists():
        return data_path

    local_root = data_path.parent.resolve()
    local_train = train_rel if train_rel.is_absolute() else local_root / train_rel
    local_val = val_rel if val_rel.is_absolute() else local_root / val_rel
    if local_train.exists() and local_val.exists():
        local_yaml = data_path.with_name(f"{data_path.stem}.local{data_path.suffix}")
        data["path"] = str(local_root)
        local_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"[data] Rewriting dataset root for this environment: {local_yaml}")
        print(f"       original path={root}")
        print(f"       local path={local_root}")
        return local_yaml

    print(f"[ERROR] YOLO images not found from dataset yaml: {data_path}")
    print(f"  Tried: {train_path} and {val_path}")
    print(f"  Also tried local root: {local_train} and {local_val}")
    print("  Run dataset conversion again or check dataset/mmp_yolo/images/{train,val}.")
    raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO11n on MMPTracking_short and export ONNX")
    p.add_argument("--data",    default="dataset/mmp_yolo/dataset.yaml")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="output/train")
    p.add_argument("--name",    default="yolo11n_mmp")
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--weights", default="yolo11n.pt",
                   help="Starting weights. Use MTA best.pt to warm-start from "
                        "an already-finetuned person detector.")
    p.add_argument("--freeze",  type=int, default=0,
                   help="Freeze first N backbone layers (0=train all). "
                        "E.g. --freeze 10 to only train detection head.")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping: stop after N epochs with no mAP50 improvement "
                        "(default 10). Set 0 to disable.")
    p.add_argument("--onnx-dest", default=str(ONNX_DEST),
                   help="Where to copy the exported ONNX (default overwrites the "
                        "deployed yolo11n_mmp.onnx).")
    p.add_argument("--cache", default=None, choices=[None, "ram", "disk"],
                   help="Cache decoded images to avoid re-reading the dataset "
                        "from disk every epoch. Use 'ram' when the dataset lives "
                        "on a slow/external drive (removes per-epoch I/O stalls).")
    args = p.parse_args()
    onnx_dest = Path(args.onnx_dest)

    data_path   = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    save_dir = project_dir / args.name

    if not data_path.exists():
        print(f"[ERROR] dataset.yaml not found: {data_path}")
        print("  Run first: python scripts/datasets/mmp_to_yolo.py")
        raise SystemExit(1)

    data_path = _resolve_yolo_data_yaml(data_path)
    _fail_if_unwritable(save_dir, "Training output directory")
    _fail_if_unwritable(data_path.parent / "labels", "YOLO labels/cache directory")
    _fail_if_unwritable(onnx_dest.parent.resolve(), "ONNX destination directory")

    if args.resume:
        last_ckpt = project_dir / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"[ERROR] No checkpoint to resume: {last_ckpt}")
            raise SystemExit(1)
        print(f"[train] Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[train] Starting from {args.weights}")
        model = YOLO(args.weights)

    freeze_arg = args.freeze if args.freeze > 0 else None
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        freeze=freeze_arg,
        patience=args.patience,
        cache=args.cache if args.cache else False,
        verbose=True,
        # Augmentation: conservative for real-world indoor
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
        degrees=0.0,
        translate=0.1,
        scale=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
    )

    best_pt = project_dir / args.name / "weights" / "best.pt"
    print(f"\n[train] Best weights: {best_pt}")

    print("\n[export] Exporting to ONNX ...")
    best_model = YOLO(str(best_pt))
    export_path = best_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        dynamic=True,
        simplify=True,
    )
    exported_onnx = Path(str(export_path))
    onnx_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_onnx, onnx_dest)
    print(f"[export] Copied to {onnx_dest}")
    print(HINT)


if __name__ == "__main__":
    main()
