#!/usr/bin/env bash
# =============================================================================
# Prepare model files that are intentionally not committed to git.
#
# Main demo (`python -m src.main`) requires:
#   models/yolov11/yolo11n.onnx            (--yolov11)
#   models/reid/swin_tiny_market1501_aicity156_featuredim256.onnx      (--reid-swin)
#
# Usage:
#   ./scripts/prepare_models.sh                # YOLO11 + ReID Swin-Tiny (main default)
#   ./scripts/prepare_models.sh --default      # same as no args
#   ./scripts/prepare_models.sh --yolo         # YOLOv8 ONNX only
#   ./scripts/prepare_models.sh --yolov11      # YOLO11n ONNX only
#   ./scripts/prepare_models.sh --reid-swin    # ReID Swin-Tiny ONNX only
#   ./scripts/prepare_models.sh --reid         # YOLOv8 + ResNet50 ReID ETLT
#   ./scripts/prepare_models.sh --all          # everything (both detectors + both ReID)
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREPARE_YOLO=0
PREPARE_YOLOV11=0
PREPARE_REID=0
PREPARE_REID_SWIN=0

if [[ $# -eq 0 ]]; then
  PREPARE_YOLOV11=1
  PREPARE_REID_SWIN=1
fi

for arg in "$@"; do
  case "$arg" in
    --yolo) PREPARE_YOLO=1 ;;
    --yolov11) PREPARE_YOLOV11=1 ;;
    --reid) PREPARE_YOLO=1; PREPARE_REID=1 ;;
    --reid-swin) PREPARE_REID_SWIN=1 ;;
    --default) PREPARE_YOLOV11=1; PREPARE_REID_SWIN=1 ;;
    --all) PREPARE_YOLO=1; PREPARE_YOLOV11=1; PREPARE_REID=1; PREPARE_REID_SWIN=1 ;;
    -h|--help)
      sed -n '1,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--yolo] [--yolov11] [--reid] [--reid-swin] [--default] [--all]"
      exit 2
      ;;
  esac
done

# Export a YOLO model to dynamic-batch ONNX inside the Docker image.
#   $1 = model subdir under models/ (e.g. "yolov8", "yolov11")
#   $2 = ultralytics weights name (e.g. "yolov8n.pt", "yolo11n.pt")
#   $3 = output ONNX filename (e.g. "yolov8n.onnx", "yolo11n.onnx")
_export_yolo_onnx() {
  local subdir="$1" weights="$2" onnx_name="$3"
  local model_dir="$ROOT_DIR/models/$subdir"
  local onnx="$model_dir/$onnx_name"

  mkdir -p "$model_dir"
  if [[ -f "$onnx" ]]; then
    echo "[prepare_models] OK $onnx"
    return
  fi

  echo "[prepare_models] Missing ONNX: $onnx"
  echo "[prepare_models] Exporting dynamic-batch ONNX inside the Docker image..."

  command -v docker >/dev/null || {
    echo "[ERROR] docker is required to export $weights automatically."
    echo "Install Docker or manually export models/$subdir/$onnx_name with:"
    echo "  from ultralytics import YOLO"
    echo "  YOLO('$weights').export(format='onnx', imgsz=640, opset=12, dynamic=True, simplify=True)"
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
    SUBDIR="$subdir" WEIGHTS="$weights" ONNX_NAME="$onnx_name" \
    docker compose run --rm --no-deps \
    -e SUBDIR -e WEIGHTS -e ONNX_NAME tracker \
    bash -lc "python3 -m pip install --no-cache-dir ultralytics onnx onnxslim && \
python3 - <<'PY'
import os
from pathlib import Path
from ultralytics import YOLO

subdir, weights, onnx_name = os.environ['SUBDIR'], os.environ['WEIGHTS'], os.environ['ONNX_NAME']
out_dir = Path('models') / subdir
out_dir.mkdir(parents=True, exist_ok=True)
pt = out_dir / weights
model = YOLO(str(pt) if pt.exists() else weights)
exported = Path(model.export(
    format='onnx',
    imgsz=640,
    opset=12,
    dynamic=True,
    simplify=True,
))
target = out_dir / onnx_name
if exported.resolve() != target.resolve():
    target.write_bytes(exported.read_bytes())
print(f'[prepare_models] Exported {target}')
PY"

  test -f "$onnx" || {
    echo "[ERROR] ONNX export did not create $onnx"
    exit 1
  }
}

prepare_yolo() {
  _export_yolo_onnx "yolov8" "yolov8n.pt" "yolov8n.onnx"
}

prepare_yolov11() {
  _export_yolo_onnx "yolov11" "yolo11n.pt" "yolo11n.onnx"
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

prepare_reid_swin() {
  local model_dir="$ROOT_DIR/models/reid"
  local onnx="$model_dir/swin_tiny_market1501_aicity156_featuredim256.onnx"
  local url="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet_transformer/deployable_v1.0/files?redirect=true&path=swin_tiny_market1501_aicity156_featuredim256.onnx"

  mkdir -p "$model_dir"
  if [[ -f "$onnx" ]]; then
    echo "[prepare_models] OK $onnx"
    return
  fi

  command -v wget >/dev/null || {
    echo "[ERROR] wget is required to download the ReID Swin-Tiny model."
    exit 1
  }

  echo "[prepare_models] Downloading ReID Swin-Tiny from NVIDIA NGC..."
  wget -q --show-progress --content-disposition "$url" -O "$onnx"
  echo "[prepare_models] Saved $onnx"
  echo "[prepare_models] NOTE: verify the ONNX input shape matches inferDims in"
  echo "                 configs/tracker/nvdeepsort_reid_swin.yaml (384x128 vs 256x128)."
}

if [[ "$PREPARE_YOLO" == "1" ]]; then
  prepare_yolo
fi

if [[ "$PREPARE_YOLOV11" == "1" ]]; then
  prepare_yolov11
fi

if [[ "$PREPARE_REID" == "1" ]]; then
  prepare_reid
fi

if [[ "$PREPARE_REID_SWIN" == "1" ]]; then
  prepare_reid_swin
fi

echo "[prepare_models] Done."
