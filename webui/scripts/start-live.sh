#!/usr/bin/env bash
# Live, end-to-end: real RTSP streams -> DeepStream pipeline (buffered-ID OSD)
# -> HLS into webui/public/live/ so the browser console plays the actual
# NVIDIA pipeline output with anchor-guided Buffered IDs drawn on it.
#
# Chain:
#   cam*.mp4 --ffmpeg -re--> MediaMTX (RTSP :8554)
#            --rtsp--> src.main (YOLO+NvDCF+SGIE, OSD = buffered remap)
#            --hlssink2--> webui/public/live/stream.m3u8  --> <video> (hls.js)
#   src.mtmc.live_buffered re-clusters embeddings -> rewrites the (cam,track)->GID
#   map every 2s; the pipeline reads it (--buffered-remap) and draws Buffered IDs.
#
# Usage:
#   Loop local videos as RTSP via MediaMTX (self-contained demo, default):
#     webui/scripts/start-live.sh [scene_dir|.txt] [envmap] [preset]
#     webui/scripts/start-live.sh dataset/MMPTracking_10minute/val/64pm_office_0 office:0-3 reid0
#
#   LIVE RTSP cameras directly — NO MediaMTX, no looping (--rtsp first):
#     webui/scripts/start-live.sh --rtsp [rtsp_list.txt] [envmap] [preset]
#     Reads rtsp:// URLs (one per line) from rtsp_list.txt; default
#     configs/sources/rtsp_cameras.txt. Use this with real cameras.
#
# Stop with Ctrl+C (tears down MediaMTX (if used) + consumer).
set -u
cd "$(dirname "$0")/../.."                      # repo root
PYTHON=./venv/bin/python3                        # explicit; `python` may not be on PATH

# --rtsp [list.txt]: skip MediaMTX/looping and ingest live RTSP cameras directly.
LIVE_RTSP=0
RTSP_LIST="configs/sources/rtsp_cameras.txt"
if [ "${1:-}" = "--rtsp" ]; then
  LIVE_RTSP=1; shift
  case "${1:-}" in *.txt) RTSP_LIST="$1"; shift ;; esac
fi

# Default = ALL 20 cameras (mixed list) so the whole console is one live stream.
# Pass a single scene dir for a 4/5-cam run instead. (Loop mode only; ignored for --rtsp.)
SCENE="${1:-configs/sources/val_20cam_mixed.txt}"
ENVMAP="${2:-}"                                  # NOT GROUPS (bash builtin); auto if empty
PRESET="${3:-reid0}"
PORT="${PORT:-8554}"
WINDOW="${WINDOW:-100}"
# 2 chunks (~22s) maps tracks to Buffered IDs ~2x faster than 4 — snappier live
# OSD, especially in retail where local tracks fragment quickly. Raise for slightly
# tighter offline-quality clustering; lower for an even snappier live demo.
WCHUNKS="${WCHUNKS:-2}"
OUT="output/live_stream"
HLS="webui/public/live"
GIDS="$OUT/export/gids.csv"

case "$PRESET" in
  reid0)   PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml ;;
  quality) PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml ;;
  *) echo "unknown preset '$PRESET'"; exit 2 ;;
esac

mkdir -p "$OUT/export" "$HLS"
rm -f "$HLS"/*.ts "$HLS"/stream.m3u8 2>/dev/null || true

MTX="scripts/eval/mediamtx_loop.sh"
MTX_STARTED=0
cleanup() {
  set +e
  [ -n "${CPID:-}" ] && kill "$CPID" 2>/dev/null
  [ "$MTX_STARTED" = "1" ] && bash "$MTX" stop >/dev/null 2>&1
  # Remove HLS segments so the console doesn't replay a stale, finished stream
  # (which would play briefly then freeze/blank). No stream = clean REPLAY fallback.
  rm -f "$HLS"/*.ts "$HLS"/stream.m3u8 2>/dev/null
  echo "[live] stopped (live segments cleared)."
}
trap cleanup INT TERM EXIT

# 1) sources -> RTSP URI list.
if [ "$LIVE_RTSP" = "1" ]; then
  # Live cameras: read rtsp:// URLs directly. No MediaMTX, no looping.
  if [ ! -f "$RTSP_LIST" ]; then
    echo "[live] ERROR: rtsp list '$RTSP_LIST' not found."; exit 2
  fi
  mapfile -t SRCS < <(grep -vE '^\s*#|^\s*$' "$RTSP_LIST")
  ncam=${#SRCS[@]}
  if [ "$ncam" -eq 0 ]; then
    echo "[live] ERROR: no rtsp:// URLs in $RTSP_LIST — add your cameras (one per line)."; exit 2
  fi
  [ -z "$ENVMAP" ] && ENVMAP="live:0-$((ncam-1))"
  echo "[live] LIVE RTSP mode: $ncam cameras from $RTSP_LIST ($ENVMAP) — no MediaMTX/looping"
  echo "[live] ${SRCS[*]}"
else
  # Demo: loop local videos -> RTSP (real-time paced, forever) via MediaMTX.
  # SCENE may be a scene dir (its cam*.mp4) OR a .txt list of video paths (20-cam mixed).
  echo "[live] starting RTSP streams from $SCENE ..."
  bash "$MTX" start "$SCENE" "$PORT" >/dev/null
  MTX_STARTED=1
  if [ -f "$SCENE" ] && echo "$SCENE" | grep -q '\.txt$'; then
    ncam=$(grep -vcE '^\s*#|^\s*$' "$SCENE")
    [ -z "$ENVMAP" ] && ENVMAP="cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
  else
    ncam=$(ls "$SCENE"/cam*.mp4 2>/dev/null | wc -l)
    [ -z "$ENVMAP" ] && ENVMAP="$(basename "$SCENE" | sed 's/^64pm_//;s/_0$//'):0-$((ncam-1))"
  fi
  SRCS=(); for i in $(seq 1 "$ncam"); do SRCS+=("rtsp://localhost:$PORT/$(printf cam%02d "$i")"); done
  echo "[live] $ncam RTSP cameras ($ENVMAP): ${SRCS[*]}"
  sleep 3
fi

# 2) buffered-ID consumer: re-cluster embeddings, rewrite (cam,track)->GID map
echo "[live] starting anchor-guided Buffered-ID consumer ..."
$PYTHON -m src.mtmc.live_buffered --export-dir "$OUT/export" --groups "$ENVMAP" \
  --window-chunks "$WCHUNKS" --assign-thr 0.40 --poll-interval 2 --max-idle 600 \
  --gids-csv "$GIDS" --log-csv "$OUT/buf.csv" > "$OUT/consumer.log" 2>&1 &
CPID=$!
sleep 2

# 3) pipeline: RTSP in -> buffered-ID OSD -> live HLS for the web console
echo "[live] starting pipeline -> HLS at $HLS/stream.m3u8"
echo "[live] open the console (npm run dev) and go to #live → PIPELINE LIVE"
$PYTHON -m src.main --config "$PIPECFG" --sources "${SRCS[@]}" \
  --no-display --no-sync --show-trajectories \
  --stream-hls "$HLS" \
  --export-predictions "$OUT/export" --live-buffered-window "$WINDOW" \
  --buffered-remap "$GIDS"
