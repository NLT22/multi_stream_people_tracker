#!/usr/bin/env bash
# MediaMTX RTSP loop launcher (production_todo §1): turn a scene's camera mp4s into
# N looped, real-time-paced RTSP streams for a live / days-long stability run.
#
#   start:  scripts/eval/mediamtx_loop.sh start dataset/MMPTracking_10minute/val/64pm_office_0
#   stop :  scripts/eval/mediamtx_loop.sh stop
#
# Then point the pipeline at the printed rtsp:// list:
#   python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
#     --sources rtsp://localhost:8554/cam01 ... --no-display --no-sync
set -u
CMD="${1:-}"; SCENE="${2:-}"; PORT="${3:-8554}"
PIDFILE="/tmp/mediamtx_loop_ffmpeg.pids"

need() { command -v "$1" >/dev/null || { echo "ERROR: $1 not found"; exit 1; }; }

if [ "$CMD" = "stop" ]; then
  [ -f "$PIDFILE" ] && { xargs -r kill 2>/dev/null < "$PIDFILE"; rm -f "$PIDFILE"; echo "stopped ffmpeg loops"; }
  docker rm -f mediamtx_loop 2>/dev/null && echo "stopped MediaMTX container" || true
  exit 0
fi

if [ "$CMD" != "start" ] || [ -z "$SCENE" ]; then
  echo "usage: $0 start <scene_dir> [port] | stop"; exit 1
fi
need ffmpeg; need docker
# SCENE may be a scene dir (its cam*.mp4) OR a .txt list of video paths (multi-scene 20-cam)
if [ -f "$SCENE" ] && echo "$SCENE" | grep -q '\.txt$'; then
  cams=$(grep -vE '^\s*#|^\s*$' "$SCENE")
else
  [ -d "$SCENE" ] || { echo "ERROR: scene dir / list not found: $SCENE"; exit 1; }
  cams=$(ls "$SCENE"/cam*.mp4 2>/dev/null | sort)
fi
[ -n "$cams" ] || { echo "ERROR: no videos found for $SCENE"; exit 1; }

# 1. MediaMTX (idempotent)
if ! docker ps --format '{{.Names}}' | grep -q '^mediamtx_loop$'; then
  docker run -d --rm --name mediamtx_loop -p "$PORT:8554" bluenviron/mediamtx >/dev/null
  echo "started MediaMTX on :$PORT"; sleep 2
fi

# 2. one looped, real-time ffmpeg per camera (-re paces to native fps, -c copy = no re-encode)
# sequential cam01..camNN names (avoids collisions when a list spans multiple scenes)
: > "$PIDFILE"; urls=(); i=0
for f in $cams; do
  i=$((i + 1)); name=$(printf "cam%02d" "$i")
  ffmpeg -re -stream_loop -1 -i "$f" -c copy -f rtsp "rtsp://localhost:$PORT/$name" \
    -loglevel error >/dev/null 2>&1 &
  echo $! >> "$PIDFILE"
  urls+=("rtsp://localhost:$PORT/$name")
done
echo "looping $(echo "$cams" | wc -l) cameras -> RTSP (forever, native fps)"
echo; echo "pipeline command:"
echo "  python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \\"
echo "    --sources ${urls[*]} --no-display --no-sync"
echo; echo "stop with: $0 stop"
