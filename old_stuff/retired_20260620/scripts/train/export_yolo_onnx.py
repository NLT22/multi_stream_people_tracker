"""Export an Ultralytics YOLO model to dynamic-batch ONNX.

This script is called inside the DeepStream Docker image by
scripts/setup/prepare_models.sh. Keeping it as a real file avoids fragile heredoc
stdin handling in `docker compose run` on different hosts.
"""

import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    subdir = os.environ["SUBDIR"]
    weights = os.environ["WEIGHTS"]
    onnx_name = os.environ["ONNX_NAME"]

    out_dir = Path("models") / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    local_weights = out_dir / weights
    model = YOLO(str(local_weights) if local_weights.exists() else weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            dynamic=True,
            simplify=True,
        )
    )

    target = out_dir / onnx_name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())

    print(f"[prepare_models] Exported {target}")


if __name__ == "__main__":
    main()
