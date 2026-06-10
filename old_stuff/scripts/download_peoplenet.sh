#!/bin/bash
# =============================================================================
# Download PeopleNet v2.6.3 from NVIDIA NGC
# =============================================================================
# PeopleNet is a ResNet34-based person detector trained by NVIDIA TAO.
# It is NOT bundled with DeepStream and must be downloaded separately.
#
# Prerequisites:
#   - NGC CLI installed: https://ngc.nvidia.com/setup/installers/cli
#   - NGC API key configured: `ngc config set`
#   - OR: download manually from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet
#
# Output:
#   models/peoplenet/resnet34_peoplenet.onnx   ← used by nvinfer_peoplenet.yml
#
# Run from project root:
#   bash scripts/setup/download_peoplenet.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models/peoplenet"

echo "[download_peoplenet] Saving to: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_DIR/resnet34_peoplenet.onnx" ]; then
    echo "[download_peoplenet] resnet34_peoplenet.onnx already exists. Skipping download."
    echo "[download_peoplenet] Delete models/peoplenet/resnet34_peoplenet.onnx to re-download."
    exit 0
fi

NGC_URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=resnet34_peoplenet.onnx"

echo "[download_peoplenet] Downloading PeopleNet v2.6.3 from NGC (no login required)..."
if ! wget -q --show-progress --content-disposition \
        "$NGC_URL" \
        -O "$MODEL_DIR/resnet34_peoplenet.onnx"; then
    echo ""
    echo "[ERROR] wget failed. Possible causes:"
    echo "  - No internet connection"
    echo "  - NGC API changed (check: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)"
    echo ""
    echo "Manual download fallback:"
    echo "  wget --content-disposition \\"
    echo "    '$NGC_URL' \\"
    echo "    -O models/peoplenet/resnet34_peoplenet.onnx"
    exit 1
fi

echo ""
echo "[download_peoplenet] Done!"
echo "  Model: $MODEL_DIR/resnet34_peoplenet.onnx"
echo ""
echo "Next steps:"
echo "  1. First pipeline run will build the TRT engine (~2 min) and cache it:"
echo "     models/peoplenet/resnet34_peoplenet.onnx_b4_gpu0_fp16.engine"
echo "  2. Use PeopleNet in the main app with --nvinfer-config:"
echo "     python -m src.main \\"
echo "         --nvinfer-config configs/models/nvinfer_peoplenet.yml"
echo "  3. The main app infers PeopleNet's person class id from the label file."
