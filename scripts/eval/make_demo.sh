#!/usr/bin/env bash
# One-command demo: annotated video (boxes + Global IDs + trajectories + live
# heatmap overlay + ROI counts) AND offline per-camera heatmaps, from one run.
#
#   scripts/eval/make_demo.sh [sources_file] [out_dir] [preset]
#
# Examples:
#   scripts/eval/make_demo.sh configs/sources/val_20cam_mixed.txt output/demo
#   scripts/eval/make_demo.sh configs/sources/val_20cam_mixed.txt output/demo quality
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null || true

SRC="${1:-configs/sources/val_20cam_mixed.txt}"
OUT="${2:-output/demo}"
PRESET="${3:-reid0}"
case "$PRESET" in
  reid0)   PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml ;;
  quality) PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml ;;
  *) echo "unknown preset '$PRESET' (use reid0|quality)"; exit 2 ;;
esac
ANALYTICS="${ANALYTICS:-configs/analytics/nvdsanalytics_mmp.txt}"
mkdir -p "$OUT/export"

echo "[demo] preset=$PRESET sources=$SRC -> $OUT"
python scripts/setup/validate_config.py --config "$PIPECFG" || {
  echo "[demo] config validation failed"; exit 3; }

# 1. annotated demo video (headless) + per-detection export
ANALYTICS_ARG=()
[ -f "$ANALYTICS" ] && ANALYTICS_ARG=(--nvdsanalytics-config "$ANALYTICS")
python -m src.main --config "$PIPECFG" --sources "$SRC" \
  --no-display --heatmap-overlay --save-video "$OUT/demo.mp4" \
  --export-predictions "$OUT/export" --live-buffered-window 200 \
  "${ANALYTICS_ARG[@]}"

# 2. offline per-camera occupancy heatmaps + montage from the export
python scripts/eval/heatmap_from_export.py \
  --export-dir "$OUT/export" --out-dir "$OUT/heatmap"

echo "[demo] DONE"
echo "[demo]   video        : $OUT/demo.mp4"
echo "[demo]   heatmaps     : $OUT/heatmap/ (per-cam PNG + montage.png + occupancy_stats.json)"
echo "[demo]   BEV floor map: scripts/eval/bev_heatmap_from_export.py --export-dir $OUT/export --calib <env>/calibrations.json --cams ..."
