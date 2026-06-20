#!/usr/bin/env bash
# Multi-environment continuous RTSP loop (production_todo §1/§6, "many envs, each
# loops & changes scene continuously"). For every environment, each camera streams
# a CONCAT PLAYLIST of all that env's scenes back-to-back, looped forever — so the
# day-long feed shows changing scenes/people per camera (realistic), not one clip.
#
#   start: scripts/eval/mediamtx_multienv.sh start dataset/MMPTracking_10minute/val [port] [envs...]
#   stop : scripts/eval/mediamtx_multienv.sh stop
# Prints the rtsp:// list (paths: <env>_cam<N>) for the pipeline.
set -u
CMD="${1:-}"; ROOT="${2:-}"; PORT="${3:-8554}"
PIDFILE="/tmp/mediamtx_multienv.pids"; PLDIR="output/_playlists"

if [ "$CMD" = "stop" ]; then
  [ -f "$PIDFILE" ] && { xargs -r kill 2>/dev/null < "$PIDFILE"; rm -f "$PIDFILE"; echo "stopped ffmpeg loops"; }
  docker rm -f mediamtx_loop 2>/dev/null && echo "stopped MediaMTX" || true
  exit 0
fi
[ "$CMD" = "start" ] && [ -d "$ROOT" ] || { echo "usage: $0 start <root> [port] [envs...]"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found"; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
shift 3 2>/dev/null || shift $#; ENVS_FILTER="$*"      # optional explicit env list

env_of() { echo "$1" | sed 's/^[0-9]*[a-z]*_//; s/_[0-9]*$//'; }   # 64pm_cafe_shop_0 -> cafe_shop

# group scenes by env
declare -A ENV_SCENES
for d in "$ROOT"/*/; do
  s=$(basename "$d"); [ "$s" = "calibrations" ] && continue
  ls "$d"cam*.mp4 >/dev/null 2>&1 || continue
  e=$(env_of "$s")
  [ -n "$ENVS_FILTER" ] && ! echo " $ENVS_FILTER " | grep -q " $e " && continue
  ENV_SCENES[$e]="${ENV_SCENES[$e]:-} $d"
done
[ ${#ENV_SCENES[@]} -gt 0 ] || { echo "ERROR: no envs found under $ROOT"; exit 1; }

# MediaMTX
if ! docker ps --format '{{.Names}}' | grep -q '^mediamtx_loop$'; then
  docker run -d --rm --name mediamtx_loop -p "$PORT:8554" bluenviron/mediamtx >/dev/null
  echo "started MediaMTX on :$PORT"; sleep 2
fi

mkdir -p "$PLDIR"; : > "$PIDFILE"; urls=(); ncam=0
for e in "${!ENV_SCENES[@]}"; do
  scenes=(${ENV_SCENES[$e]})
  cams=$(ls "${scenes[0]}"cam*.mp4 | xargs -n1 basename | sed 's/cam//;s/.mp4//' | sort -n)
  for c in $cams; do
    pl="$PLDIR/${e}_cam${c}.txt"; : > "$pl"
    for sc in "${scenes[@]}"; do
      f="${sc}cam${c}.mp4"; [ -f "$f" ] && echo "file '$(realpath "$f")'" >> "$pl"
    done
    [ -s "$pl" ] || continue
    ffmpeg -re -stream_loop -1 -f concat -safe 0 -i "$pl" -c copy \
      -f rtsp "rtsp://localhost:$PORT/${e}_cam${c}" -loglevel error >/dev/null 2>&1 &
    echo $! >> "$PIDFILE"
    urls+=("rtsp://localhost:$PORT/${e}_cam${c}"); ncam=$((ncam + 1))
  done
  echo "  env $e: ${#scenes[@]} scenes x cams [$cams] cycling"
done
printf '%s\n' "${urls[@]}" > "$PLDIR/urls.txt"     # for the orchestrator
echo "[multienv] $ncam cameras across ${#ENV_SCENES[@]} envs, each looping its scenes"
echo; echo "pipeline command:"
echo "  python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \\"
echo "    --sources ${urls[*]} --no-display --no-sync"
echo; echo "stop: $0 stop"
