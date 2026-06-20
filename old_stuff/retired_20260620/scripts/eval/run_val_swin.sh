#!/usr/bin/env bash
# Full val-set (64pm) benchmark with SWIN cross-camera embeddings (ONE-model path).
# Reuses the existing SCT exports (output/eval/heldout_64pm_*) and only swaps the
# cross-camera ReID from OSNet -> Swin (the same model already in the NvDCF tracker).
# Per scene: swin dense embed (torch+GPU) -> anchor (oracle-k) -> metrics.
set -u
cd "$(dirname "$0")/../.."
source venv/bin/activate 2>/dev/null
VROOT=dataset/MMPTracking_10minute/val
LOG=output/logs/val_swin.log
CSV=output/eval/val_swin_results.csv
: > "$LOG"; echo "scene,env,idf1" > "$CSV"

for S in $(ls -d "$VROOT"/*/ | xargs -n1 basename | grep -v calib); do
  SRC=output/eval/heldout_$S
  OUT=output/eval/swinval_$S
  env=$(echo "$S" | sed 's/^64pm_//; s/_[0-9]*$//')
  [ -f "$SRC/cam_0_predictions.csv" ] || { echo "SKIP $S (no SCT export)" | tee -a "$LOG"; continue; }
  echo "######## $S (env=$env) ########" | tee -a "$LOG"
  rm -rf "$OUT" "${OUT}_a"
  python scripts/anchor_guided/swin_reid_embed.py --pred-dir "$SRC" --out-dir "$OUT" \
    --short-root "$VROOT" --scene "$S" >> "$LOG" 2>&1
  python -m src.eval.offline_anchor_faithful --pred-dir "$OUT" --out-dir "${OUT}_a" \
    --short-root "$VROOT" --scene "$S" --oracle-k >> "$LOG" 2>&1
  GT=""; echo "$S" | grep -q retail && GT="--gt-suffix _clean"
  IDF1=$(python -m src.eval.metrics_mmp --short-root "$VROOT" --scene "$S" $GT \
    --pred-dir "${OUT}_a" 2>/dev/null | grep -oE "Global IDF1: [0-9.]+" | grep -oE "[0-9.]+$")
  echo "RESULT $S $env $IDF1" | tee -a "$LOG"
  echo "$S,$env,$IDF1" >> "$CSV"
  rm -rf "$OUT"   # free the big npz; keep the anchor output for video/inspection
done

echo "==== PER-ENV MEAN (Swin cross-camera) ====" | tee -a "$LOG"
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
echo "VAL_SWIN_DONE" | tee -a "$LOG"
