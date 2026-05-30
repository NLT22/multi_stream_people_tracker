#!/usr/bin/env bash
# =============================================================================
# Download and unpack the NVIDIA DeepStream multi-view demo dataset.
#
# Source:
#   https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/blob/master/deepstream-tracker-3d-multi-view/assets/datasets.zip
#
# The zip is intentionally not committed to this repository. This script creates:
#   dataset/mtmc_4cam/
#   dataset/mtmc_12cam/
#   dataset/Wildtrack/
#
# Usage:
#   ./scripts/prepare_dataset.sh
#   DATASET_DIR=/absolute/path/to/dataset ./scripts/prepare_dataset.sh
#   ./scripts/prepare_dataset.sh --force
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache}"
ZIP_PATH="$CACHE_DIR/nvidia_deepstream_datasets.zip"
RAW_URL="https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/raw/master/deepstream-tracker-3d-multi-view/assets/datasets.zip"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      sed -n '1,17p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: DATASET_DIR=/path/to/dataset $0 [--force]"
      exit 2
      ;;
  esac
done

has_default_dataset() {
  [[ -f "$DATASET_DIR/mtmc_4cam/videos/Warehouse_Synthetic_Cam001.mp4" ]] &&
  [[ -f "$DATASET_DIR/mtmc_4cam/videos/Warehouse_Synthetic_Cam002.mp4" ]] &&
  [[ -f "$DATASET_DIR/mtmc_4cam/videos/Warehouse_Synthetic_Cam003.mp4" ]] &&
  [[ -f "$DATASET_DIR/mtmc_4cam/videos/Warehouse_Synthetic_Cam004.mp4" ]]
}

if [[ "$FORCE" != "1" ]] && has_default_dataset; then
  echo "[prepare_dataset] OK $DATASET_DIR/mtmc_4cam/videos"
  echo "[prepare_dataset] Use --force to re-download/re-extract."
  exit 0
fi

command -v unzip >/dev/null || {
  echo "[ERROR] unzip is required."
  echo "Install it with: sudo apt-get install unzip"
  exit 1
}

mkdir -p "$DATASET_DIR" "$CACHE_DIR"

if [[ "$FORCE" == "1" || ! -f "$ZIP_PATH" ]]; then
  echo "[prepare_dataset] Downloading NVIDIA dataset zip (~161 MB)..."
  if command -v curl >/dev/null; then
    curl -L --fail "$RAW_URL" -o "$ZIP_PATH"
  elif command -v wget >/dev/null; then
    wget -O "$ZIP_PATH" "$RAW_URL"
  else
    echo "[ERROR] curl or wget is required to download the dataset."
    exit 1
  fi
else
  echo "[prepare_dataset] Reusing cached zip: $ZIP_PATH"
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[prepare_dataset] Extracting..."
unzip -q "$ZIP_PATH" -d "$TMP_DIR"

copy_dataset_dir() {
  local name="$1"
  local src
  src="$(find "$TMP_DIR" -type d -name "$name" | head -n1 || true)"
  if [[ -z "$src" ]]; then
    echo "[prepare_dataset] WARN missing $name in zip"
    return
  fi

  rm -rf "$DATASET_DIR/$name"
  if command -v rsync >/dev/null; then
    rsync -a "$src/" "$DATASET_DIR/$name/"
  else
    mkdir -p "$DATASET_DIR/$name"
    cp -a "$src/." "$DATASET_DIR/$name/"
  fi
  echo "[prepare_dataset] OK $DATASET_DIR/$name"
}

copy_dataset_dir mtmc_4cam
copy_dataset_dir mtmc_12cam
copy_dataset_dir Wildtrack

if ! has_default_dataset; then
  echo "[ERROR] mtmc_4cam videos were not found after extraction."
  echo "Inspect extracted zip structure or download manually from:"
  echo "  $RAW_URL"
  exit 1
fi

echo ""
echo "[prepare_dataset] Done."
echo "[prepare_dataset] Default 4-camera video folder:"
echo "  $DATASET_DIR/mtmc_4cam/videos"
