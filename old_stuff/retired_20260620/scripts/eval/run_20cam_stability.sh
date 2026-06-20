#!/usr/bin/env bash
# 20-camera long-day STABILITY run (production_todo §6 + §1).
# Waits for the FP16 Swin batch to finish (no GPU contention), then:
#   MediaMTX (20 looped RTSP, forever) -> online 20-cam pipeline -> stability monitor.
# Runs continuously (RTSP never ends) until you stop it.
#   stop:  scripts/eval/mediamtx_loop.sh stop ; pkill -f 'src.main.*cam01'
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
LIST=configs/sources/val_20cam.txt
PORT=8554
RUNLOG=output/logs/run20cam.log
STABCSV=output/logs/stability_20cam.csv
mkdir -p output/logs

# 1. wait for the FP16 Swin batch (avoid GPU contention)
echo "[20cam] waiting for FP16 Swin batch (VAL_SWIN_DONE)..."
while ! grep -q VAL_SWIN_DONE output/logs/val_swin.log 2>/dev/null; do
  pgrep -f "[r]un_val_swin.sh" >/dev/null || { echo "[20cam] swin batch not running and not done — proceeding"; break; }
  sleep 30
done
echo "[20cam] GPU free — launching."

# 2. MediaMTX 20 looped RTSP streams (cam01..cam20)
bash scripts/eval/mediamtx_loop.sh start "$LIST" "$PORT" || { echo "[20cam] mediamtx start failed"; exit 1; }
sleep 5
URLS=""; for i in $(seq -w 1 20); do URLS="$URLS rtsp://localhost:$PORT/cam$i"; done

# 3. online 20-cam pipeline (perf preset; rtsp = live = runs forever)
echo "[20cam] starting pipeline -> $RUNLOG"
python -m src.main --config configs/pipelines/pipeline_mmp_realtime_20cam.yaml \
  $URLS --no-display > "$RUNLOG" 2>&1 &
PPID_=$!
sleep 10
echo "[20cam] pipeline pid=$PPID_"

# 4. stability monitor (logs FPS/VRAM/RSS/GID every 30s until pipeline exits)
bash scripts/eval/stability_monitor.sh --pid "$PPID_" --log "$RUNLOG" \
  --out "$STABCSV" --interval 30 &
echo "[20cam] monitor -> $STABCSV"
echo "[20cam] RUNNING. Watch: column -t -s, $STABCSV | tail"
echo "[20cam] STOP:  scripts/eval/mediamtx_loop.sh stop ; kill $PPID_"
wait $PPID_
echo "[20cam] pipeline exited"
