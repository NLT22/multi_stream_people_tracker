#!/usr/bin/env bash
# Full val-set (64pm, unseen people) offline anchor-guided eval.
# Per scene: [retail: clean GT stride-1] -> SCT export (FP32/nms0.7) -> dense OSNet
# embed -> anchor (oracle-k) -> metrics. Reuses existing anchor outputs.
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
VROOT=dataset/MMPTracking_10minute/val
DET=configs/models/nvinfer_yolov11_10min_clean_fp32nms07.yml
TRK=configs/tracker/nvdcf_accuracy_mmp_recall_clean.yaml
OSNET=models/reid_osnet_mmp/osnet_mmp_retail.pth
LOG=output/logs/val_full.log
CSV=output/eval/val_full_results.csv
: > "$LOG"; echo "scene,env,idf1" > "$CSV"

scenes=$(ls -d "$VROOT"/*/ | xargs -n1 basename | grep -v calib)
for S in $scenes; do
  P=output/eval/heldout_$S
  env=$(echo "$S" | sed 's/^64pm_//; s/_[0-9]*$//')
  echo "######## $S (env=$env) ########" | tee -a "$LOG"
  GTSUF=""
  if echo "$S" | grep -q retail; then
    GTSUF="--gt-suffix _clean"
    # stride-1 clean GT for fair retail eval (idempotent overwrite)
    python scripts/datasets/clean_gt_csv.py --root dataset/MMPTracking_10minute \
      --scene "$S" --stride 1 >> "$LOG" 2>&1
  fi
  if [ ! -f "${P}_anchor/cam_0_predictions.csv" ]; then
    python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml \
      --mmp-short-dataset "$VROOT:$S" --nvinfer-config "$DET" --tracker-config "$TRK" \
      --no-display --no-sync --export-predictions "$P" >> "$LOG" 2>&1
    python scripts/anchor_guided/their_reid_embed.py --pred-dir "$P" --scene "$S" \
      --short-root "$VROOT" --ckpt "$OSNET" >> "$LOG" 2>&1
    python -m src.eval.offline_anchor_faithful --pred-dir "$P" --out-dir "${P}_anchor" \
      --short-root "$VROOT" --scene "$S" --oracle-k >> "$LOG" 2>&1
  else
    echo "  (reusing existing anchor output)" | tee -a "$LOG"
  fi
  IDF1=$(python -m src.eval.metrics_mmp --short-root "$VROOT" --scene "$S" $GTSUF \
    --pred-dir "${P}_anchor" 2>/dev/null | grep -oE "Global IDF1: [0-9.]+" | grep -oE "[0-9.]+$")
  echo "RESULT $S $env $IDF1" | tee -a "$LOG"
  echo "$S,$env,$IDF1" >> "$CSV"
done

echo "==== PER-ENV MEAN ====" | tee -a "$LOG"
python3 - "$CSV" <<'PY' | tee -a "$LOG"
import csv,sys,collections,statistics
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['idf1']]
by=collections.defaultdict(list)
for r in rows: by[r['env']].append(float(r['idf1']))
for e in sorted(by):
    v=by[e]; print(f"{e:18s} n={len(v)} mean={statistics.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")
allv=[float(r['idf1']) for r in rows]
print(f"{'OVERALL':18s} n={len(allv)} mean={statistics.mean(allv):.3f}")
PY
echo "VAL_FULL_DONE" | tee -a "$LOG"
