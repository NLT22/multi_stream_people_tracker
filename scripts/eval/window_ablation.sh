#!/bin/bash
# Buffered ID window-size ablation: 200 / 300 / 400 frames.
#
# Strategy: one pipeline run with chunk=100 frames (--live-buffered-window 100),
# then reprocess the same chunks three times with window-chunks 2/3/4.
# Retail keeps 4× the base window (same ratio as the canonical eval).
#
# Usage:
#   bash scripts/eval/window_ablation.sh [--skip-pipeline]
#
# --skip-pipeline: reuse an existing export (already ran step 1).

set -euo pipefail

SKIP_PIPELINE=0
for arg in "$@"; do [[ "$arg" == "--skip-pipeline" ]] && SKIP_PIPELINE=1; done

PYTHON=./venv/bin/python3
source venv/bin/activate

EXPORT=output/eval/window_ablation
VALROOT=dataset/MMPTracking_10minute/val
MAP="cafe=64pm_cafe_shop_0:0-3,lobby=64pm_lobby_0:4-7,office=64pm_office_0:8-11,industry=64pm_industry_safety_0:12-15,retail=64pm_retail_0:16-19"
CAM_GROUPS="cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"

mkdir -p "$EXPORT"

# ---------------------------------------------------------------------------
# Step 1: single-pass pipeline with 100-frame chunks
# ---------------------------------------------------------------------------
if [[ $SKIP_PIPELINE -eq 0 ]]; then
  echo "=== Step 1: pipeline run (chunk=100 frames) ==="
  $PYTHON -m src.main \
    --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
    --sources configs/sources/val_20cam_mixed.txt \
    --no-display --no-sync \
    --export-predictions "$EXPORT" \
    --live-buffered-window 100
  echo "=== Pipeline done ==="
else
  echo "=== Skipping pipeline run (--skip-pipeline) ==="
fi

CHUNK_COUNT=$(ls "$EXPORT"/det_emb_chunk_*.npz 2>/dev/null | wc -l)
echo "Found $CHUNK_COUNT embedding chunks in $EXPORT"

# ---------------------------------------------------------------------------
# Step 2: post-process 3x and score each
# ---------------------------------------------------------------------------
for WINDOW in 200 300 400; do
  CHUNKS=$((WINDOW / 100))
  RETAIL_CHUNKS=$((CHUNKS * 4))
  ASSIGN="$EXPORT/_eval_assign.csv"

  echo ""
  echo "======================================================================"
  echo "  Window = $WINDOW frames  (default=$CHUNKS chunks, retail=$RETAIL_CHUNKS chunks)"
  echo "======================================================================"

  $PYTHON -m src.mtmc.live_buffered \
    --export-dir "$EXPORT" \
    --once \
    --window-chunks "$CHUNKS" \
    --groups "$CAM_GROUPS" \
    --assign-csv "$ASSIGN"

  echo "--- Global IDF1 (window=${WINDOW}f) ---"
  $PYTHON scripts/eval/score_longrun_idf1.py \
    --export-dir "$EXPORT" \
    --map "$MAP" \
    --val-root "$VALROOT"
  echo ""
done

echo "=== Ablation complete ==="
