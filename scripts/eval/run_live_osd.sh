#!/usr/bin/env bash
# Live OSD with BUFFERED (anchor-guided) Global IDs — route (a).
#
# Runs the pipeline and the live_buffered consumer together: the consumer
# re-clusters the exported embeddings (anchor-guided) and rewrites a
# (cam,local_track_id)->global_id map; the pipeline reads that map (--buffered-remap)
# and draws the authoritative buffered IDs on the OSD instead of the volatile online
# IDs. Runs REALTIME so the consumer keeps pace; buffered IDs appear after ~1 window.
#
#   scripts/eval/run_live_osd.sh [sources_file] [out_dir] [preset] [env_map]
#   scripts/eval/run_live_osd.sh configs/sources/val_20cam_mixed.txt output/live_osd reid0 \
#       "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null || true

SRC="${1:-dataset/MMPTracking_10minute/val/64pm_office_0}"   # dir or sources .txt
OUT="${2:-output/live_osd}"
PRESET="${3:-reid0}"
ENVMAP="${4:-office:0-3}"                                     # NOTE: not GROUPS (bash builtin)
WINDOW="${WINDOW:-100}"
WCHUNKS="${WCHUNKS:-4}"
case "$PRESET" in
  reid0)   PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml ;;
  quality) PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml ;;
  *) echo "unknown preset '$PRESET' (reid0|quality)"; exit 2 ;;
esac

# sources: a dir of cam*.mp4, or a ready-made sources file
if [ -d "$SRC" ]; then
  SRCLIST="$OUT/_sources.txt"; mkdir -p "$OUT"; ls "$SRC"/cam*.mp4 | sort > "$SRCLIST"
else
  SRCLIST="$SRC"
fi
mkdir -p "$OUT/export"
GIDS="$OUT/export/gids.csv"

cleanup(){ set +e; [ -n "${CPID:-}" ] && kill "$CPID" 2>/dev/null; }
trap cleanup INT TERM EXIT

echo "[live-osd] preset=$PRESET sources=$SRCLIST env=$ENVMAP -> $OUT/live_osd.mp4"
# 1) consumer: anchor-guided re-cluster -> rewrite the buffered remap continuously
python -m src.mtmc.live_buffered --export-dir "$OUT/export" --groups "$ENVMAP" \
  --window-chunks "$WCHUNKS" --assign-thr 0.40 --poll-interval 2 --max-idle 300 \
  --gids-csv "$GIDS" --log-csv "$OUT/buf.csv" > "$OUT/consumer.log" 2>&1 &
CPID=$!
sleep 3

# 2) pipeline REALTIME (no --no-sync) so the consumer keeps pace; OSD reads the remap
python -m src.main --config "$PIPECFG" --sources "$SRCLIST" \
  --no-display --show-trajectories \
  --save-video "$OUT/live_osd.mp4" \
  --export-predictions "$OUT/export" --live-buffered-window "$WINDOW" \
  --buffered-remap "$GIDS"

cleanup; trap - INT TERM EXIT
echo "[live-osd] DONE -> $OUT/live_osd.mp4  (buffered IDs: $(tail -n +2 "$GIDS" 2>/dev/null | cut -d, -f4 | sort -u | wc -l) distinct)"
