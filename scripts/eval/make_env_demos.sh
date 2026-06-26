#!/usr/bin/env bash
# Per-environment demo bundle, organized under output/demo/<env>/:
#   <env>_osd_buffered.mp4   real DeepStream OSD video, FULL length, Buffered IDs
#   heatmap/                 occupancy/footfall/dwell per-cam + BEV (venv_visualize)
#
# Uses the deployed production preset (retail-clean detector). Reuses the
# full-length retail-clean exports for buffered IDs, so only the OSD render runs
# on GPU. One representative (best-IDF1) scene per environment.
#
# Usage: scripts/eval/make_env_demos.sh
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.   # venv_visualize.py imports src.reid.geometry
PY=./venv/bin/python3
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
EXPROOT=output/eval/full_mmp_val_retailclean
VALROOT=dataset/MMPTracking_10minute/val
CALROOT=dataset/MMPTracking/MMPTracking_validation/validation/calibrations

# env -> representative scene (highest Global IDF1)
ENVS=(cafe_shop lobby office industry_safety retail)
declare -A SCENE=(
  [cafe_shop]=64pm_cafe_shop_1 [lobby]=64pm_lobby_3 [office]=64pm_office_0
  [industry_safety]=64pm_industry_safety_2 [retail]=64pm_retail_3
)

for ENV in "${ENVS[@]}"; do
  S="${SCENE[$ENV]}"
  EXPORT="$EXPROOT/$S"
  SRC="configs/sources/val_full_mmp_${S}.txt"
  DEST="output/demo/$ENV"
  WORK="$DEST/_work"
  mkdir -p "$DEST/heatmap" "$WORK"
  if ! ls "$EXPORT"/det_emb_chunk_*.npz >/dev/null 2>&1; then
    echo "[env:$ENV] no export at $EXPORT — skip"; continue
  fi
  NCAM=$(ls "$EXPORT"/cam_*_predictions.csv 2>/dev/null | wc -l)
  CAMS=$(seq 0 $((NCAM-1)))
  CALIB="$CALROOT/$ENV/calibrations.json"
  echo "=== [env:$ENV] scene=$S cams=$NCAM ==="

  # 1) Buffered global IDs (+ per-detection assign for footfall heatmap)
  $PY -m src.mtmc.live_buffered --export-dir "$EXPORT" \
      --window-chunks 1 --assign-thr 0.50 --once \
      --gids-csv "$WORK/gids.csv" --assign-csv "$WORK/assign.csv" \
      --log-csv "$WORK/lb.log" > "$WORK/buf.log" 2>&1
  NGID=$(tail -n +2 "$WORK/gids.csv" 2>/dev/null | cut -d, -f4 | sort -un | wc -l)
  echo "[env:$ENV] buffered IDs: $NGID"

  # 2) FULL-length real-OSD video with Buffered IDs + trajectory trails (no trim).
  #    This is the single OSD video (replaces the old OpenCV *_live_buffered_osd.mp4).
  echo "[env:$ENV] rendering full-length OSD video (with trajectories) ..."
  $PY -m src.main --config "$PIPECFG" --sources "$SRC" \
      --no-display --no-sync --show-trajectories \
      --buffered-remap "$WORK/gids.csv" \
      --save-video "$DEST/${ENV}_osd_buffered.mp4" > "$WORK/render.log" 2>&1
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 \
        "$DEST/${ENV}_osd_buffered.mp4" 2>/dev/null)
  echo "[env:$ENV] OSD video -> ${ENV}_osd_buffered.mp4 (${DUR}s)"

  # 3) Analytics overlay video (gst-nvdsanalytics: ROI / line-crossing / overcrowding)
  AN="configs/analytics/nvdsanalytics_${ENV}.txt"; [ -f "$AN" ] || AN=configs/analytics/nvdsanalytics_mmp.txt
  echo "[env:$ENV] analytics video ..."
  $PY -m src.main --config "$PIPECFG" --sources "$SRC" \
      --no-display --no-sync --show-trajectories --nvdsanalytics-config "$AN" \
      --save-video "$DEST/${ENV}_analytics.mp4" > "$WORK/analytics.log" 2>&1

  # 4) Heatmaps (occupancy/footfall/dwell per-cam + BEV) and the BEV top-down
  #    tracking video, via venv_visualize.
  echo "[env:$ENV] heatmaps + BEV ..."
  $PY scripts/eval/venv_visualize.py --export-dir "$EXPORT" \
      --video-dir "$VALROOT/$S" --calib "$CALIB" --cams $CAMS \
      --assign-csv "$WORK/assign.csv" --video-steps 2 --track-frames 1500 \
      --out-dir "$DEST/heatmap" > "$WORK/heatmap.log" 2>&1 \
      && echo "[env:$ENV] heatmaps -> $DEST/heatmap/ ($(ls "$DEST/heatmap"/*.png 2>/dev/null | wc -l) png)" \
      || { echo "[env:$ENV] heatmap FAILED"; tail -5 "$WORK/heatmap.log"; }
  # promote the BEV tracking video; drop only the timelapse + the redundant OpenCV OSD
  [ -f "$DEST/heatmap/bev_tracking.mp4" ] && mv -f "$DEST/heatmap/bev_tracking.mp4" "$DEST/${ENV}_tracking_bev.mp4"
  rm -f "$DEST/heatmap/timelapse_"*.mp4 "$DEST/heatmap/cam_tracking.mp4" \
        "$DEST/${ENV}_live_buffered_osd.mp4"
  rm -rf "$WORK"
  echo "[env:$ENV] videos: $(ls "$DEST"/*.mp4 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
done
echo "ALL_ENV_DEMOS_DONE"
