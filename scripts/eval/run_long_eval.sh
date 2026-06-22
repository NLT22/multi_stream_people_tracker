#!/usr/bin/env bash
# Long-duration production-style eval/soak.
#
# Usage:
#   scripts/eval/run_long_eval.sh [duration_seconds] [sources_file] [env_map]
#
# Example:
#   PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
#   bash scripts/eval/run_long_eval.sh \
#     600 configs/sources/val_20cam_mixed.txt \
#     "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
#
# Outputs:
#   $LOGDIR/long_buffered.csv  - live buffered identity-health rows
#   $LOGDIR/long_stability.csv - GPU/VRAM/RSS/FPS rows
#   $LOGDIR/long_pipe.log      - DeepStream pipeline log
#   output/eval/long_run           - per-camera CSVs + det_emb_chunk_*.npz
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null || true

DUR="${1:-7200}"
SRCLIST="${2:-configs/sources/val_20cam_mixed.txt}"
ENV_MAP="${3:-}"
PIPECFG="${PIPECFG:-configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml}"
EXPORT="${EXPORT:-output/eval/long_run}"
LIVE_BUFFERED_WINDOW="${LIVE_BUFFERED_WINDOW:-200}"
WINDOW_CHUNKS="${WINDOW_CHUNKS:-1}"
WINDOW_CHUNKS_SPEC="${WINDOW_CHUNKS_SPEC:-retail:4,default:1}"
ASSIGN_THR="${ASSIGN_THR:-0.40}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
MAIN_EXTRA_ARGS="${MAIN_EXTRA_ARGS:-}"

# Run-dir naming (production_todo 4.3): USE_RUN_DIR=1 stores export + logs under
# output/runs/<ts>_<preset>/ with a run_manifest.json. Default keeps legacy paths.
PRESET="$(basename "$PIPECFG" .yaml | sed 's/^pipeline_mmp_nvdcf_online_//')"
if [ "${USE_RUN_DIR:-0}" = "1" ]; then
  RUN_DIR="${RUN_DIR:-output/runs/$(date +%Y%m%d_%H%M%S)_${PRESET}}"
  EXPORT="$RUN_DIR/export"
  LOGDIR="$RUN_DIR"
else
  LOGDIR="output/logs"
fi
PIPELOG="${PIPELOG:-$LOGDIR/long_pipe.log}"

mkdir -p "$LOGDIR" "$EXPORT"
rm -f "$EXPORT"/det_emb_chunk_*.npz "$EXPORT"/det_emb_chunk_*.tmp.npz \
      "$EXPORT"/cam_*_predictions.csv "$EXPORT"/tracklets.csv \
      "$EXPORT"/_eval_assign.csv "$EXPORT"/_eval_buf.csv \
      $LOGDIR/long_buffered.csv $LOGDIR/long_gids.csv \
      $LOGDIR/long_stability.csv "$PIPELOG"

if [ ! -f "$SRCLIST" ]; then
  echo "[long-eval] missing source list: $SRCLIST" >&2
  exit 2
fi
if [ ! -f "$PIPECFG" ]; then
  echo "[long-eval] missing pipeline config: $PIPECFG" >&2
  exit 2
fi

CAM_COUNT=$(grep -cvE '^[[:space:]]*(#|$)' "$SRCLIST")
echo "[long-eval] cams=$CAM_COUNT sources=$SRCLIST duration=${DUR}s"
echo "[long-eval] pipeline=$PIPECFG"
[ -n "$ENV_MAP" ] && echo "[long-eval] env_map=$ENV_MAP"
echo "[long-eval] export=$EXPORT live_buffered_window=$LIVE_BUFFERED_WINDOW window_chunks=$WINDOW_CHUNKS"
[ -n "$WINDOW_CHUNKS_SPEC" ] && echo "[long-eval] window_chunks_spec=$WINDOW_CHUNKS_SPEC"

# Pre-flight guardrail (production_todo 4.1) — abort on a broken preset unless skipped.
if [ "${SKIP_VALIDATE:-0}" != "1" ]; then
  VAL_ARGS=(--config "$PIPECFG")
  [ -n "$ENV_MAP" ] && VAL_ARGS+=(--sources "$SRCLIST" --env-map "$ENV_MAP")
  if ! python scripts/setup/validate_config.py "${VAL_ARGS[@]}"; then
    echo "[long-eval] config validation FAILED — aborting (SKIP_VALIDATE=1 to override)" >&2
    exit 3
  fi
fi

# Run manifest (production_todo 4.3).
python scripts/eval/write_run_manifest.py \
  --config "$PIPECFG" --sources "$SRCLIST" --env-map "$ENV_MAP" --duration "$DUR" \
  --extra "{\"export\":\"$EXPORT\",\"live_buffered_window\":$LIVE_BUFFERED_WINDOW,\"assign_thr\":$ASSIGN_THR,\"window_chunks_spec\":\"$WINDOW_CHUNKS_SPEC\",\"preset\":\"$PRESET\"}" \
  --out "$EXPORT/run_manifest.json" || true

cleanup() {
  set +e
  [ -n "${PPID_:-}" ] && kill "$PPID_" 2>/dev/null
  [ -n "${CPID:-}" ] && kill "$CPID" 2>/dev/null
  [ -n "${MPID:-}" ] && kill "$MPID" 2>/dev/null
}
trap cleanup INT TERM EXIT

python -m src.main \
  --config "$PIPECFG" \
  --sources "$SRCLIST" \
  --no-display --no-sync --loop-video \
  --export-predictions "$EXPORT" \
  --live-buffered-window "$LIVE_BUFFERED_WINDOW" \
  $MAIN_EXTRA_ARGS \
  > "$PIPELOG" 2>&1 &
PPID_=$!
echo "[long-eval] pipeline pid=$PPID_ -> $PIPELOG"
sleep 20

BUFFER_ARGS=(
  --export-dir "$EXPORT"
  --window-chunks "$WINDOW_CHUNKS"
  --assign-thr "$ASSIGN_THR"
  --duration "$DUR"
  --max-idle 180
  --log-csv $LOGDIR/long_buffered.csv
  --gids-csv $LOGDIR/long_gids.csv
  --assign-csv "$EXPORT/_eval_assign.csv"
)
[ -n "$ENV_MAP" ] && BUFFER_ARGS+=(--groups "$ENV_MAP")
[ -n "$WINDOW_CHUNKS_SPEC" ] && BUFFER_ARGS+=(--group-window-chunks "$WINDOW_CHUNKS_SPEC")

python -m src.mtmc.live_buffered "${BUFFER_ARGS[@]}" \
  > $LOGDIR/long_buffered_stdout.log 2>&1 &
CPID=$!

bash scripts/eval/stability_monitor.sh \
  --pid "$PPID_" \
  --log "$PIPELOG" \
  --out $LOGDIR/long_stability.csv \
  --interval "$MONITOR_INTERVAL" \
  > $LOGDIR/long_monitor_stdout.log 2>&1 &
MPID=$!

echo "[long-eval] consumer pid=$CPID monitor pid=$MPID — running ${DUR}s"
sleep "$DUR"
echo "[long-eval] duration reached — stopping"
cleanup
sleep 5
trap - INT TERM EXIT

echo "[long-eval] DONE"
echo "[long-eval] buffered=$LOGDIR/long_buffered.csv"
echo "[long-eval] stability=$LOGDIR/long_stability.csv"
echo "[long-eval] pipe_log=$PIPELOG"
