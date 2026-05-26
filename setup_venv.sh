#!/bin/bash
# =============================================================================
# Setup virtual environment for multi_stream_people_tracker
# =============================================================================
# Run once from the project root:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#
# Then activate before running any milestone or the main app:
#   source venv/bin/activate
# =============================================================================

set -e  # exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Creating Python virtual environment ==="
python3 -m venv venv

echo "=== Activating venv ==="
source venv/bin/activate

echo "=== Installing pyservicemaker from DeepStream SDK wheel ==="
# pyservicemaker is installed system-wide by the DeepStream installer but is
# NOT accessible from a standard venv. We must install it explicitly.
# The .whl file is bundled with DeepStream 9.0 at the path below.
PSMAKER_WHL=$(ls /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl 2>/dev/null | head -n1)

if [ -z "$PSMAKER_WHL" ]; then
    echo "[ERROR] pyservicemaker wheel not found."
    echo "        Expected: /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl"
    echo "        Make sure DeepStream 9.0 is installed."
    exit 1
fi

echo "        Installing: $PSMAKER_WHL"
pip install "$PSMAKER_WHL" pyyaml

echo "=== Installing project requirements ==="
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start working:"
echo "  source venv/bin/activate"
echo ""
echo "To run Milestone 1 (replace with your video path):"
echo "  python milestones/01_single_video_display.py --input /path/to/video.mp4"
echo ""
echo "To run the full pipeline:"
echo "  python -m src.main --config configs/pipeline.yaml"
