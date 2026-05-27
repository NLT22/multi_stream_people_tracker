#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DO_BUILD=0
DO_RUN=0
for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    --run) DO_RUN=1 ;;
    --all) DO_BUILD=1; DO_RUN=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: VIDEO_DIR=/path/to/videos $0 [--build] [--run|--all]"
      exit 2
      ;;
  esac
done

echo "== Host checks =="
command -v docker >/dev/null || { echo "Missing docker"; exit 1; }
docker compose version >/dev/null || { echo "Missing docker compose"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Missing nvidia-smi"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo ""
echo "== Project files =="
required=(
  Dockerfile
  docker-compose.yml
  configs/sources/video_files_docker.txt
  configs/models/nvinfer_yolov8_people.yml
  models/yolov8/yolov8n.onnx
  models/yolov8/libnvds_infercustomparser_yolov8.so
)
for path in "${required[@]}"; do
  test -e "$path" || { echo "Missing $path"; exit 1; }
  echo "OK $path"
done

echo ""
echo "== Video mount =="
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos}"
if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "VIDEO_DIR does not exist: $VIDEO_DIR"
  echo "Set it with: VIDEO_DIR=/absolute/path/to/videos $0 --all"
  exit 1
fi
echo "VIDEO_DIR=$VIDEO_DIR"

echo ""
echo "== Compose config =="
VIDEO_DIR="$VIDEO_DIR" docker compose config >/dev/null
echo "OK docker compose config"

if [[ "$DO_BUILD" == "1" ]]; then
  echo ""
  echo "== Docker build =="
  VIDEO_DIR="$VIDEO_DIR" docker compose build
fi

if [[ "$DO_RUN" == "1" ]]; then
  echo ""
  echo "== Container import smoke test =="
  VIDEO_DIR="$VIDEO_DIR" docker compose run --rm tracker \
    python3 -c "import pyservicemaker; import yaml; from src.pipeline.model_utils import infer_person_class_id; print('person_class_id=', infer_person_class_id('configs/models/nvinfer_yolov8_people.yml'))"
fi

echo ""
echo "Docker smoke test completed."
