#!/usr/bin/env bash
# =============================================================================
# Prepare model files that are intentionally not committed to git.
#
# Default Docker demo requires:
#   models/yolov8/yolov8n.onnx
#
# Milestone 8 ReID also requires:
#   models/reid/resnet50_market1501.etlt
#
# Usage:
#   ./scripts/prepare_models.sh          # prepare YOLOv8 ONNX only
#   ./scripts/prepare_models.sh --reid   # YOLOv8 ONNX + ReID ETLT
#   ./scripts/prepare_models.sh --all    # same as --reid
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREPARE_YOLO=1
PREPARE_REID=0

for arg in "$@"; do
  case "$arg" in
    --yolo) PREPARE_YOLO=1 ;;
    --reid|--all) PREPARE_YOLO=1; PREPARE_REID=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--yolo] [--reid|--all]"
      exit 2
      ;;
  esac
done

prepare_yolo() {
  local model_dir="$ROOT_DIR/models/yolov8"
  local onnx="$model_dir/yolov8n.onnx"

  mkdir -p "$model_dir"
  if [[ -f "$onnx" ]]; then
    echo "[prepare_models] OK $onnx"
    return
  fi

  echo "[prepare_models] Missing YOLOv8 ONNX: $onnx"
  echo "[prepare_models] Exporting dynamic-batch ONNX inside the Docker image..."

  command -v docker >/dev/null || {
    echo "[ERROR] docker is required to export YOLOv8 ONNX automatically."
    echo "Install Docker or manually export models/yolov8/yolov8n.onnx with:"
    echo "  from ultralytics import YOLO"
    echo "  YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12, dynamic=True, simplify=True)"
    exit 1
  }
  docker compose version >/dev/null || {
    echo "[ERROR] docker compose is required."
    exit 1
  }

  mkdir -p "$ROOT_DIR/videos"
  X11_SOCKET_DIR="${X11_SOCKET_DIR:-$ROOT_DIR/.docker-x11}"
  mkdir -p "$X11_SOCKET_DIR"
  VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}" X11_SOCKET_DIR="$X11_SOCKET_DIR" \
    docker compose build tracker
  VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}" X11_SOCKET_DIR="$X11_SOCKET_DIR" \
    docker compose run --rm --no-deps tracker \
    bash -lc "python3 -m pip install --no-cache-dir ultralytics onnx onnxslim && \
python3 - <<'PY'
from pathlib import Path
from ultralytics import YOLO

out_dir = Path('models/yolov8')
out_dir.mkdir(parents=True, exist_ok=True)
pt = out_dir / 'yolov8n.pt'
model = YOLO(str(pt) if pt.exists() else 'yolov8n.pt')
exported = Path(model.export(
    format='onnx',
    imgsz=640,
    opset=12,
    dynamic=True,
    simplify=True,
))
target = out_dir / 'yolov8n.onnx'
if exported.resolve() != target.resolve():
    target.write_bytes(exported.read_bytes())
print(f'[prepare_models] Exported {target}')
PY"

  test -f "$onnx" || {
    echo "[ERROR] YOLOv8 ONNX export did not create $onnx"
    exit 1
  }
}

prepare_reid() {
  local model_dir="$ROOT_DIR/models/reid"
  local etlt="$model_dir/resnet50_market1501.etlt"
  local url="https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt"

  mkdir -p "$model_dir"
  if [[ -f "$etlt" ]]; then
    echo "[prepare_models] OK $etlt"
    return
  fi

  command -v wget >/dev/null || {
    echo "[ERROR] wget is required to download the ReID model."
    exit 1
  }

  echo "[prepare_models] Downloading ReID model from NVIDIA NGC..."
  wget -q --show-progress "$url" -O "$etlt"
  echo "[prepare_models] Saved $etlt"
}

if [[ "$PREPARE_YOLO" == "1" ]]; then
  prepare_yolo
fi

if [[ "$PREPARE_REID" == "1" ]]; then
  prepare_reid
fi

echo "[prepare_models] Done."
