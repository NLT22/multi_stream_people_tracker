"""Evaluate a YOLO detector on the official MMPTracking zip-derived val set."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/mmp_exact_yolo/dataset.yaml")
    parser.add_argument("--weights", default="models/yolov11/yolo11n_mmp.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="output/eval_exact")
    parser.add_argument("--name", default="yolo_val")
    args = parser.parse_args()

    data = Path(args.data).resolve()
    weights = Path(args.weights).resolve()
    project = Path(args.project).resolve()
    if not data.exists():
        raise SystemExit(f"dataset yaml not found: {data}")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    print(f"[eval] data={data}")
    print(f"[eval] weights={weights}")
    model = YOLO(str(weights))
    results = model.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
    )
    print(f"[eval] save_dir={results.save_dir}")
    print(f"[eval] metrics={results.results_dict}")


if __name__ == "__main__":
    main()
