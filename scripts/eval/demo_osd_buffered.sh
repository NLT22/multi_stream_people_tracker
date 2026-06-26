#!/usr/bin/env bash
# Real DeepStream OSD demo with Buffered (anchor-guided) Global IDs.
#
# Renders the genuine nvosdbin output (boxes + Global-ID labels) to video, with
# the labels driven by the buffered/anchor-guided global IDs (not the volatile
# online IDs). Uses the production pipeline preset, so it reflects the currently
# deployed detector (retail-clean YOLO11n) — phantom shelf boxes are gone, so no
# static-FP filter is applied.
#
# Two passes:
#   1. export per-detection embeddings -> live_buffered --once -> gids.csv
#   2. re-run the pipeline with --buffered-remap gids.csv --save-video
#      (nvosdbin draws a box for EVERY detection every frame; tracks not yet
#       clustered show "ID:..." then converge — boxes never vanish mid-video).
#
# Usage:
#   scripts/eval/demo_osd_buffered.sh <name> <sources.txt> <trim_s> [reuse_export_dir] [groups]
# Examples:
#   scripts/eval/demo_osd_buffered.sh 64pm_cafe_shop_1 \
#       configs/sources/val_full_mmp_64pm_cafe_shop_1.txt 150 \
#       output/eval/full_mmp_val_retailclean/64pm_cafe_shop_1
#   scripts/eval/demo_osd_buffered.sh live_20cam configs/sources/val_20cam_mixed.txt 90 "" \
#       "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
set -u
cd "$(dirname "$0")/../.."
PY=./venv/bin/python3
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml

# NOTE: do NOT name this variable GROUPS — that is a bash special/readonly array
# (the current user's group IDs); assigning it leaks the gid (e.g. 1000) into
# --groups. Use GRP. (Same lesson as run_live_osd.sh.)
NAME="$1"; SOURCES="$2"; TRIM="${3:-150}"; REUSE="${4:-}"; GRP="${5:-}"
OUT=output/DEMO
WORK="$OUT/_work/$NAME"
mkdir -p "$WORK" "$OUT"

# ---- Pass 1: get per-detection embeddings (reuse an existing export if given) ----
if [ -n "$REUSE" ] && ls "$REUSE"/det_emb_chunk_*.npz >/dev/null 2>&1; then
  echo "[demo:$NAME] reusing export $REUSE"
  EXPORT="$REUSE"
else
  echo "[demo:$NAME] exporting embeddings (trim ${TRIM}s) ..."
  $PY -m src.main --config "$PIPECFG" --sources "$SOURCES" \
      --no-display --no-sync --trim-seconds "$TRIM" \
      --export-predictions "$WORK/export" --live-buffered-window 200 \
      > "$WORK/pass1.log" 2>&1
  EXPORT="$WORK/export"
fi

# ---- Buffered global IDs (anchor-guided), no FP filter (detector is clean) ----
echo "[demo:$NAME] live_buffered --once -> gids.csv"
if [ -n "$GRP" ]; then
  $PY -m src.mtmc.live_buffered --export-dir "$EXPORT" \
      --window-chunks 1 --assign-thr 0.50 --once --groups "$GRP" \
      --gids-csv "$WORK/gids.csv" --log-csv "$WORK/lb.log" > "$WORK/buf.log" 2>&1
else
  $PY -m src.mtmc.live_buffered --export-dir "$EXPORT" \
      --window-chunks 1 --assign-thr 0.50 --once \
      --gids-csv "$WORK/gids.csv" --log-csv "$WORK/lb.log" > "$WORK/buf.log" 2>&1
fi
NGID=$(tail -n +2 "$WORK/gids.csv" 2>/dev/null | cut -d, -f4 | sort -un | wc -l)
echo "[demo:$NAME] distinct buffered IDs: $NGID"
if [ "${NGID:-0}" -lt 1 ]; then
  echo "[demo:$NAME] ABORT — no buffered IDs (see $WORK/buf.log)"; tail -4 "$WORK/buf.log"; exit 1
fi

# ---- Pass 2: real OSD render with buffered IDs ----
echo "[demo:$NAME] rendering real-OSD video (trim ${TRIM}s) ..."
$PY -m src.main --config "$PIPECFG" --sources "$SOURCES" \
    --no-display --no-sync --trim-seconds "$TRIM" \
    --buffered-remap "$WORK/gids.csv" \
    --save-video "$OUT/${NAME}_osd_buffered.mp4" \
    > "$WORK/pass2.log" 2>&1

if [ -f "$OUT/${NAME}_osd_buffered.mp4" ]; then
  echo "[demo:$NAME] DONE -> $OUT/${NAME}_osd_buffered.mp4 ($(du -h "$OUT/${NAME}_osd_buffered.mp4" | cut -f1))"
else
  echo "[demo:$NAME] FAILED — see $WORK/pass2.log"; tail -5 "$WORK/pass2.log"
fi
