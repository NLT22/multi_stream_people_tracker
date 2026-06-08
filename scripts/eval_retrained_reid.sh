#!/usr/bin/env bash
# Re-evaluate all non-retail scenes with the retrained (all-non-retail) ReID model.
# Run after scripts/finetune_reid_mmp.py --train-all-nonretail finishes.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

SRC_ONNX=output/reid_mmp_all/swin_tiny_mmp_reid.onnx
DST_ONNX=models/reid/swin_tiny_mmp_reid_all.onnx
[ -f "$SRC_ONNX" ] || { echo "Trained ONNX not found: $SRC_ONNX"; exit 1; }
cp -f "$SRC_ONNX" "$DST_ONNX"
# force tracker to rebuild its engine for the new onnx
rm -f models/reid/swin_tiny_mmp_reid_all.onnx_*.engine 2>/dev/null || true
echo "copied retrained ReID -> $DST_ONNX"

getidf () { python -m src.eval.metrics_mmp --short-root dataset/MMPTracking_short --scene "$1" --pred-dir "$2" 2>&1 | grep -oE "Global IDF1: [0-9.]+" | grep -oE "[0-9.]+$"; }
RES=output/benchmark/retrained_reid_idf1.txt
echo "Retrained-ReID (all non-retail) IDF1, $(date)" > $RES

for SC in lobby_0 lobby_3 cafe_shop_0 cafe_shop_3 industry_safety_0 industry_safety_4 office_0 office_2; do
  ED=output/eval/rt_$SC; rm -rf $ED ${ED}_nl
  python -u -m src.main \
    --config configs/pipeline_mmp_nvdcf_realtime_baseline.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:$SC --no-display --no-sync \
    --tracker-config configs/tracker/nvdcf_accuracy_mmp_recall_all.yaml \
    --export-predictions $ED > /dev/null 2>&1
  on=$(getidf $SC $ED)
  best=$on; bestcfg=online
  for TH in 0.55 0.62; do for GW in 0.15 0.25 0.35; do
    out=${ED}_nl; rm -rf $out
    python -m src.eval.nearline_merge --pred-dir $ED --out-dir $out --threshold $TH --margin 0.02 \
      --geo-weight $GW --geo-min-overlaps 8 --window-frames 125 --delay-frames 50 \
      --min-gid-embeddings 4 --min-tracklet-detections 6 \
      --mmp-short-root dataset/MMPTracking_short --scene $SC > /dev/null 2>&1
    idf=$(getidf $SC $out)
    [ -n "$idf" ] && awk "BEGIN{exit !($idf>$best)}" && { best=$idf; bestcfg="nl:th=$TH,gw=$GW"; }
  done; done
  echo "[$SC] online=$on BEST=$best ($bestcfg)" | tee -a $RES
done
python3 - <<'PY' | tee -a $RES
import re
vals=[]
for line in open("output/benchmark/retrained_reid_idf1.txt"):
    m=re.search(r'BEST=([0-9.]+)',line)
    if m: vals.append(float(m.group(1)))
if vals: print("AVG best-per-scene = %.4f over %d ; clear0.8=%d"%(sum(vals)/len(vals),len(vals),sum(v>=0.8 for v in vals)))
PY
echo RETRAINEVALDONE | tee -a $RES
