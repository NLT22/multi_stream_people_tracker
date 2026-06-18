#!/usr/bin/env bash
# QUEUED experiment: NvDCF Single-View-3D tracking (stateEstimatorType:3, ground-plane
# foot via MMP calibration) vs the 2D tracker, on 64pm_office_0.
# Compares the SAME downstream (Swin cross-camera anchor) so it isolates the SCT 3D effect.
# Baseline (2D tracker, Swin-anchor) = 0.880.
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
VROOT=dataset/MMPTracking_10minute/val; S=64pm_office_0
P=output/eval/sv3dt_$S; LOG=output/logs/sv3dt_test.log; : > "$LOG"

echo "[1] SCT export with SV3DT tracker (stateEstimatorType:3)..." | tee -a "$LOG"
python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml \
  --mmp-short-dataset "$VROOT:$S" \
  --nvinfer-config configs/models/nvinfer_yolov11_10min_clean_fp32nms07.yml \
  --tracker-config configs/tracker/nvdcf_sv3dt_office.yaml \
  --no-display --no-sync --export-predictions "$P" >> "$LOG" 2>&1
echo "  export exit=$? preds=$(ls $P/cam_*_predictions.csv 2>/dev/null | wc -l)" | tee -a "$LOG"

if [ "$(ls $P/cam_*_predictions.csv 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[2] Swin cross-camera embed + anchor + eval..." | tee -a "$LOG"
  python scripts/anchor_guided/swin_reid_embed.py --pred-dir "$P" --out-dir "${P}_e" \
    --short-root "$VROOT" --scene "$S" >> "$LOG" 2>&1
  python -m src.eval.offline_anchor_faithful --pred-dir "${P}_e" --out-dir "${P}_a" \
    --short-root "$VROOT" --scene "$S" --oracle-k >> "$LOG" 2>&1
  V=$(python -m src.eval.metrics_mmp --short-root "$VROOT" --scene "$S" --pred-dir "${P}_a" 2>/dev/null | grep -oE "Global IDF1: [0-9.]+")
  echo "=== SV3DT tracker -> $V   (2D-tracker Swin-anchor baseline = 0.8801) ===" | tee -a "$LOG"
  rm -rf "${P}_e"
else
  echo "  EXPORT FAILED — check $LOG (SV3DT config / camInfo format)" | tee -a "$LOG"
fi
echo "SV3DT_TEST_DONE" | tee -a "$LOG"
