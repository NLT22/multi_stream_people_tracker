#!/usr/bin/env bash
# Long-duration FULL-SYSTEM eval (production_todo §6, option B): runs the real
# online pipeline (perception + Swin ReID) continuously on looped multi-env video,
# with the LIVE buffered cross-camera consumer clustering rolling windows, plus a
# stability monitor. Measures identity health + system stability over a long run.
#
#   scripts/eval/run_long_eval.sh [duration_seconds]   # default 7200 (2h)
#
# Outputs:
#   output/logs/long_buffered.csv  - per-window: active/total global IDs, latency
#   output/logs/long_stability.csv - GPU/VRAM/RSS/FPS over time
#   output/logs/long_pipe.log      - pipeline log
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
DUR="${1:-7200}"
SRCLIST=configs/sources/val_10cam_mixed.txt
EXPORT=output/eval/long_run
PIPELOG=output/logs/long_pipe.log
mkdir -p output/logs "$EXPORT"
rm -f "$EXPORT"/det_emb_chunk_*.npz "$EXPORT"/cam_*_predictions.csv

# point the online preset's source list at the mixed-env cams
cp configs/sources/video_files.txt configs/sources/video_files.txt.longbak
cp "$SRCLIST" configs/sources/video_files.txt
restore() { cp configs/sources/video_files.txt.longbak configs/sources/video_files.txt 2>/dev/null; \
            rm -f configs/sources/video_files.txt.longbak; }
trap restore EXIT

echo "[long-eval] $(wc -l < $SRCLIST) cams, looped, duration ${DUR}s"

# 1. live online pipeline (gallery+ReID on, loop, flush embedding chunks every 200f)
python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online.yaml \
  --no-display --no-sync --loop-video \
  --export-predictions "$EXPORT" --live-buffered-window 200 \
  > "$PIPELOG" 2>&1 &
PPID_=$!
echo "[long-eval] pipeline pid=$PPID_ -> $PIPELOG"
sleep 20      # let engines build + first chunks appear

# 2. live buffered cross-camera consumer (rolling-window re-cluster)
python -m src.mtmc.live_buffered --export-dir "$EXPORT" --window-chunks 1 \
  --assign-thr 0.40 --duration "$DUR" --max-idle 180 \
  --log-csv output/logs/long_buffered.csv --gids-csv output/logs/long_gids.csv \
  > output/logs/long_buffered_stdout.log 2>&1 &
CPID=$!

# 3. stability monitor
bash scripts/eval/stability_monitor.sh --pid "$PPID_" --log "$PIPELOG" \
  --out output/logs/long_stability.csv --interval 30 \
  > output/logs/long_monitor_stdout.log 2>&1 &
MPID=$!

echo "[long-eval] consumer pid=$CPID  monitor pid=$MPID  — running ${DUR}s"
# run for the duration, then tear everything down
sleep "$DUR"
echo "[long-eval] duration reached — stopping"
kill "$PPID_" "$CPID" "$MPID" 2>/dev/null
sleep 5
restore
echo "[long-eval] DONE. buffered=output/logs/long_buffered.csv  stability=output/logs/long_stability.csv"
