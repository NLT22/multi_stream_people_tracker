"""
Fine-tune YOLO11n on MMPTracking_short and export ONNX for DeepStream.

Two-step workflow:
  1. Build YOLO dataset:   python scripts/mmp_to_yolo.py
  2. Train + export ONNX:  python scripts/train_yolo_mmp.py

Run:
    python scripts/train_yolo_mmp.py \\
        [--data dataset/mmp_yolo/dataset.yaml] \\
        [--epochs 30] [--batch 16] [--imgsz 640] [--device 0] \\
        [--weights yolo11n.pt]   # or output/train/yolo11n_mta/weights/best.pt
        [--resume]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

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
    args = p.parse_args()

    data_path   = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()

    if not data_path.exists():
        print(f"[ERROR] dataset.yaml not found: {data_path}")
        print("  Run first: python scripts/mmp_to_yolo.py")
        raise SystemExit(1)

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
    ONNX_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_onnx, ONNX_DEST)
    print(f"[export] Copied to {ONNX_DEST}")
    print(HINT)


if __name__ == "__main__":
    main()
