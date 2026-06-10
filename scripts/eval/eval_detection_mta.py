"""
Compare detection mAP: COCO-pretrained YOLO11n vs MTA fine-tuned.

Runs Ultralytics validation on the MTA val split and prints a side-by-side
comparison table.

Run:
    # After mta_to_yolo.py and train_yolo_mta.py:
    python scripts/eval/eval_detection_mta.py

    # Custom paths:
    python scripts/eval/eval_detection_mta.py \\
        --data  dataset/mta_yolo/dataset.yaml \\
        --baseline  yolo11n.pt \\
        --finetuned output/train/yolo11n_mta/weights/best.pt \\
        [--imgsz 640] [--device 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _eval(weights: str, data: str, imgsz: int, device: str) -> dict:
    model   = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device,
                        split="val", verbose=False)
    return {
        "mAP50":    metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "P":        metrics.box.mp,
        "R":        metrics.box.mr,
    }


def _fmt(v: float) -> str:
    return f"{v * 100:.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare mAP: baseline YOLO11n vs MTA fine-tuned")
    p.add_argument("--data",      default="dataset/mta_yolo/dataset.yaml")
    p.add_argument("--baseline",  default="yolo11n.pt",
                   help="Baseline pretrained weights (COCO)")
    p.add_argument("--finetuned", default="runs/detect/output/train/yolo11n_mta/weights/best.pt",
                   help="Fine-tuned weights")
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] dataset.yaml not found: {data_path}")
        print("  Run: python scripts/datasets/mta_to_yolo.py --mta-root dataset/mta/MTA_ext_short")
        raise SystemExit(1)

    ft_path = Path(args.finetuned)
    if not ft_path.exists():
        print(f"[ERROR] Fine-tuned weights not found: {ft_path}")
        print("  Run: python scripts/train/train_yolo_mta.py")
        raise SystemExit(1)

    print(f"[eval] Dataset : {data_path}")
    print(f"[eval] Baseline: {args.baseline}")
    print(f"[eval] Finetuned: {ft_path}\n")

    print("Evaluating baseline …")
    base = _eval(args.baseline, str(data_path), args.imgsz, args.device)

    print("Evaluating fine-tuned …")
    fine = _eval(str(ft_path), str(data_path), args.imgsz, args.device)

    # ── Print comparison table ─────────────────────────────────────────────────
    col_w = 14
    header = f"{'Metric':<12} {'Baseline':>{col_w}} {'Fine-tuned':>{col_w}} {'Delta':>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for key in ("mAP50", "mAP50-95", "P", "R"):
        b = base[key]
        f = fine[key]
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"{key:<12} {_fmt(b):>{col_w}} {_fmt(f):>{col_w}} {sign}{_fmt(delta):>{col_w}}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
