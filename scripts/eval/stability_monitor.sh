#!/usr/bin/env bash
# Long-run stability monitor (production_todo §6): samples the running pipeline every
# --interval seconds and appends a CSV row, so a days-long run can be checked for:
#   - FPS stability (no degradation)         - VRAM/RSS creep (memory leaks)
#   - Global-ID growth (gallery leak: total GIDs should PLATEAU, not climb forever)
#   - GPU utilization
#
#   scripts/eval/stability_monitor.sh --log output/logs/run20cam.log \
#       --pid <pipeline_pid> --out output/logs/stability.csv --interval 30
# Stops automatically when the pipeline PID exits. Plot/inspect the CSV anytime.
set -u
LOG=""; PID=""; OUT="output/logs/stability.csv"; INT=30
while [ $# -gt 0 ]; do case "$1" in
  --log) LOG="$2"; shift 2;; --pid) PID="$2"; shift 2;;
  --out) OUT="$2"; shift 2;; --interval) INT="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done

mkdir -p "$(dirname "$OUT")"
echo "ts,elapsed_s,gpu_util,gpu_mem_mb,rss_mb,fps,n_gids" > "$OUT"
t0=$(date +%s)
echo "[monitor] -> $OUT every ${INT}s (stops when pid $PID exits)"
while true; do
  [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null && { echo "[monitor] pid $PID gone — stop"; break; }
  now=$(date +%s); el=$((now - t0))
  read gu gm < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
  rss=$( [ -n "$PID" ] && ps -o rss= -p "$PID" 2>/dev/null | awk '{print int($1/1024)}' || echo "")
  # last FPS and GID-count printed by the pipeline (gallery probe logs these)
  fps=$( [ -n "$LOG" ] && grep -aoE "FPS:[[:space:]]+[0-9.]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$" || echo "")
  gid=$( [ -n "$LOG" ] && grep -aoE "total_gids_ever_assigned=[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$" || echo "")
  echo "$(date -Is),$el,${gu:-},${gm:-},${rss:-},${fps:-},${gid:-}" >> "$OUT"
  sleep "$INT"
done
echo "[monitor] done; $(wc -l < "$OUT") rows in $OUT"
