"""Export YOLO11n-pose to ONNX for the (not-yet-wired) pose SGIE — improvement #2.

Produces a dynamic-batch ONNX next to the weights so DeepStream can run the pose
model as a secondary GIE on tracked person crops. This only prepares the model;
nothing in the live pipeline references it yet (see src/reid/pose.py and
configs/models/nvinfer_yolo11n_pose_sgie.yml).

    python scripts/export_yolo_pose.py        # -> models/pose/yolo11n-pose.onnx

Requires ultralytics (training-time dependency; use the venv or a PyTorch env).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export YOLO11n-pose to ONNX")
    p.add_argument("--weights", default="yolo11n-pose.pt",
                   help="Pose weights (auto-downloaded by ultralytics if absent)")
    p.add_argument("--out-dir", default="models/pose")
    p.add_argument("--imgsz", type=int, default=256,
                   help="SGIE crop input size (square); person crops are small")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. `pip install ultralytics` "
            "(training-time dependency)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=True,      # dynamic batch for DeepStream SGIE batching
        simplify=True,
    )
    dst = out_dir / Path(onnx_path).name
    if Path(onnx_path).resolve() != dst.resolve():
        shutil.move(str(onnx_path), str(dst))
    print(f"[export] wrote {dst}")
    print("[export] NOTE: not referenced by any active preset. To use it you "
          "still need a YOLO-pose keypoint output parser for the SGIE — see "
          "configs/models/nvinfer_yolo11n_pose_sgie.yml.")


if __name__ == "__main__":
    main()
