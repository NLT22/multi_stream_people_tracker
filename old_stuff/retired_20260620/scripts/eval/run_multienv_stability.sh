#!/usr/bin/env bash
# Multi-env continuous long-day STABILITY run (production_todo §1/§6).
# Each env's cameras cycle through that env's scenes forever (changing scenes/people),
# fed live via MediaMTX -> online pipeline -> stability monitor. Waits for the FP16
# Swin batch first (no GPU contention). Runs until stopped.
#   stop: scripts/eval/mediamtx_multienv.sh stop ; pkill -f 'src.main.*rtsp'
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
ROOT="${1:-dataset/MMPTracking_10minute/val}"
PORT="${2:-8554}"
RUNLOG=output/logs/run_multienv.log
STABCSV=output/logs/stability_multienv.csv
mkdir -p output/logs

echo "[multienv] waiting for FP16 Swin batch (VAL_SWIN_DONE)..."
while ! grep -q VAL_SWIN_DONE output/logs/val_swin.log 2>/dev/null; do
  pgrep -f "[r]un_val_swin.sh" >/dev/null || { echo "[multienv] swin not running/done — proceeding"; break; }
  sleep 30
done
echo "[multienv] GPU free — launching."

bash scripts/eval/mediamtx_multienv.sh start "$ROOT" "$PORT" || { echo "mediamtx failed"; exit 1; }
sleep 5
mapfile -t URLS < output/_playlists/urls.txt
[ ${#URLS[@]} -gt 0 ] || { echo "no URLs"; exit 1; }
echo "[multienv] ${#URLS[@]} streams -> pipeline"

python -m src.main --config configs/pipelines/pipeline_mmp_realtime_20cam.yaml \
  "${URLS[@]}" --no-display > "$RUNLOG" 2>&1 &
PPID_=$!
sleep 10
echo "[multienv] pipeline pid=$PPID_ -> $RUNLOG"
bash scripts/eval/stability_monitor.sh --pid "$PPID_" --log "$RUNLOG" \
  --out "$STABCSV" --interval 30 &
echo "[multienv] RUNNING. watch: column -t -s, $STABCSV | tail"
echo "[multienv] STOP:  scripts/eval/mediamtx_multienv.sh stop ; kill $PPID_"
wait $PPID_
echo "[multienv] pipeline exited"
