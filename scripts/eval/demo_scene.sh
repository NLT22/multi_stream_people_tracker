#!/usr/bin/env bash
# Per-scene demo set (matches the office demo): analytics video + buffered-ID OSD
# (camera view) + BEV top-down tracking + per-camera/BEV heatmaps. One fast GPU run
# (analytics + export) then offline render (buffered IDs via live_buffered).
#
#   scripts/eval/demo_scene.sh <scene> <calib-env> <group> [track_frames]
#   scripts/eval/demo_scene.sh 64pm_cafe_shop_0 cafe_shop cafe 1500
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.   # venv_visualize imports src.reid.geometry
SCENE="$1"; ENV="$2"; GROUP="$3"; TRACKF="${4:-1500}"
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
# per-env analytics (ROI/line/threshold tuned per scene); fall back to the generic one
ANALYTICS="configs/analytics/nvdsanalytics_${ENV}.txt"
[ -f "$ANALYTICS" ] || ANALYTICS=configs/analytics/nvdsanalytics_mmp.txt
VIDEO_DIR="dataset/MMPTracking_10minute/val/$SCENE"
CALIB="dataset/MMPTracking/MMPTracking_validation/validation/calibrations/$ENV/calibrations.json"
OUT=output/demo
WORK="$OUT/_work/$SCENE"
rm -rf "$WORK"; mkdir -p "$WORK" "$OUT/heatmap_$SCENE"
ls "$VIDEO_DIR"/cam*.mp4 | sort > "$WORK/sources.txt"
echo "[demo:$SCENE] sources=$(wc -l < "$WORK/sources.txt") calib=$ENV group=$GROUP"

# 1) GPU run (fast, --no-sync): analytics overlay video + per-detection export (+chunks)
./venv/bin/python -m src.main --config "$PIPECFG" --sources "$WORK/sources.txt" \
  --no-display --no-sync --show-trajectories \
  --trim-seconds $((TRACKF / 10)) \
  --nvdsanalytics-config "$ANALYTICS" \
  --save-video "$OUT/${SCENE}_analytics.mp4" \
  --export-predictions "$WORK" --live-buffered-window 200 > "$WORK/run.log" 2>&1
echo "[demo:$SCENE] analytics video + export done ($(grep -ac FPS "$WORK/run.log") fps samples)"

# 2) buffered (anchor-guided) global IDs from the export
./venv/bin/python -m src.mtmc.live_buffered --export-dir "$WORK" --groups "$GROUP:0-3" \
  --window-chunks 4 --assign-thr 0.40 --once \
  --assign-csv "$WORK/_eval_assign.csv" --gids-csv "$WORK/gids.csv" > "$WORK/buf.log" 2>&1
echo "[demo:$SCENE] buffered IDs: $(tail -n +2 "$WORK/gids.csv" 2>/dev/null | cut -d, -f4 | sort -u | wc -l) distinct"

# 3) offline render: buffered-ID camera view + BEV tracking + heatmaps
./venv/bin/python scripts/eval/venv_visualize.py \
  --export-dir "$WORK" --video-dir "$VIDEO_DIR" --calib "$CALIB" --cams 0 1 2 3 \
  --assign-csv "$WORK/_eval_assign.csv" --cam-tracking \
  --track-frames "$TRACKF" --video-steps 2 --out-dir "$WORK/viz" > "$WORK/viz.log" 2>&1

# 4) collect named artifacts to match the office demo layout
[ -f "$WORK/viz/cam_tracking.mp4" ] && cp "$WORK/viz/cam_tracking.mp4" "$OUT/${SCENE}_live_buffered_osd.mp4"
[ -f "$WORK/viz/bev_tracking.mp4" ] && cp "$WORK/viz/bev_tracking.mp4" "$OUT/${SCENE}_tracking_bev.mp4"
cp "$WORK/viz"/cam_*_{occupancy,footfall,dwelltime}.png "$OUT/heatmap_$SCENE/" 2>/dev/null
cp "$WORK/viz"/bev_{occupancy,footfall,dwelltime}.png "$OUT/heatmap_$SCENE/" 2>/dev/null

echo "[demo:$SCENE] artifacts:"
ls -la "$OUT/${SCENE}"_*.mp4 2>/dev/null | awk '{print "   ", $5, $9}'
echo "    heatmaps: $(ls "$OUT/heatmap_$SCENE"/*.png 2>/dev/null | wc -l) pngs"
