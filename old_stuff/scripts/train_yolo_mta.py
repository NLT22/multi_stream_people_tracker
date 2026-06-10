"""
Fine-tune YOLO11n on MTA dataset and export to ONNX for DeepStream.

Workflow:
  1. Load YOLO11n pretrained weights (auto-download from Ultralytics)
  2. Fine-tune on dataset/mta_yolo/ (created by mta_to_yolo.py)
  3. Export best.pt → best.onnx (dynamic-batch, opset 12)
  4. Copy ONNX to models/yolov11/yolo11n_mta.onnx

Run:
    python scripts/train/train_yolo_mta.py \\
        [--data dataset/mta_yolo/dataset.yaml] \\
        [--epochs 50] [--batch 16] [--imgsz 640] [--device 0] \\
        [--resume]     # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ONNX_DEST = Path("models/yolov11/yolo11n_mta.onnx")
NVINFER_HINT = """
Next steps — run the fine-tuned model in DeepStream:

  # Use the MTA-specific nvinfer config:
  python -m src.main \\
      --mta-dataset dataset/mta/MTA_ext_short/test \\
      --nvinfer-config configs/models/nvinfer_yolov11_mta.yml \\
      --export-predictions output/eval/mta_finetuned \\
      --no-sync --no-display

  # Then evaluate tracking metrics:
  python -m src.eval.metrics \\
      --gt-dir dataset/mta/MTA_ext_short/test \\
      --pred-dir output/eval/mta_finetuned
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO11n on MTA and export ONNX for DeepStream")
    p.add_argument("--data", default="dataset/mta_yolo/dataset.yaml",
                   help="Path to dataset.yaml (created by mta_to_yolo.py)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="0",
                   help="GPU device id or 'cpu'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="output/train")
    p.add_argument("--name",    default="yolo11n_mta")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--weights", default="yolo11n.pt",
                   help="Base weights to fine-tune from (default: yolo11n.pt)")
    args = p.parse_args()

    # Use absolute paths so Ultralytics doesn't prepend runs/detect/
    data_path   = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()

    if not data_path.exists():
        print(f"[ERROR] dataset.yaml not found: {data_path}")
        print("  Run first: python scripts/datasets/mta_to_yolo.py --mta-root dataset/mta/MTA_ext_short")
        raise SystemExit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.resume:
        last_ckpt = project_dir / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"[ERROR] No checkpoint to resume from: {last_ckpt}")
            raise SystemExit(1)
        print(f"[train] Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[train] Starting from {args.weights}")
        model = YOLO(args.weights)

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
        # Single-class fine-tune: freeze backbone, only train head
        # Comment out to train all layers
        # freeze=10,
        verbose=True,
    )

    best_pt = project_dir / args.name / "weights" / "best.pt"
    print(f"\n[train] Best weights: {best_pt}")

    # ── Export ONNX ────────────────────────────────────────────────────────────
    print("\n[export] Exporting to dynamic-batch ONNX …")
    best_model = YOLO(str(best_pt))
    export_path = best_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        dynamic=True,
        simplify=True,
    )
    exported_onnx = Path(str(export_path))
    print(f"[export] ONNX: {exported_onnx}")

    # ── Copy to models/ ────────────────────────────────────────────────────────
    ONNX_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_onnx, ONNX_DEST)
    print(f"[export] Copied to {ONNX_DEST}")

    print(NVINFER_HINT)


if __name__ == "__main__":
    main()
