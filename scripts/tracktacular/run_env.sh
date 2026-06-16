#!/usr/bin/env bash
# Full TrackTacular loop for one environment/scene:
#   convert -> train (SegNet) -> test -> 3-way BEV compare.
# Usage: bash scripts/tracktacular/run_env.sh <env> <scene> <num_cameras> [frame_step]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source venv/bin/activate

ENV="$1"; SCENE="$2"; NCAM="$3"; STEP="${4:-2}"
OUT="dataset/worldtrack/mmp_${SCENE}"
ROOT="output/tracktacular/${SCENE}"
TT="reference/TrackTacular/WorldTrack"
echo "=========== $SCENE (env=$ENV, $NCAM cams) ==========="

# 1. convert (skip if done)
if [ ! -f "$OUT/annotations_positions" ] && [ ! -d "$OUT/annotations_positions" ]; then
    python scripts/tracktacular/mmp_to_worldtrack.py --scene "$SCENE" --out "$OUT" --frame-step "$STEP"
fi

# 2. train
( cd "$TT" && python world_track.py fit \
    -c configs/t_fit.yml -c configs/d_mmp_industry.yml -c configs/m_segnet.yml \
    --data.init_args.data_dir "$REPO/$OUT" \
    --model.num_cameras "$NCAM" \
    --trainer.default_root_dir "$REPO/$ROOT" )

# 3. pick best checkpoint by val_center (lowest)
CKDIR="$ROOT/lightning_logs/version_0/checkpoints"
BEST=$(python3 - "$CKDIR" <<'PY'
import sys, glob, re, os
best, bv = None, 1e9
for f in glob.glob(os.path.join(sys.argv[1], "model-epoch=*.ckpt")):
    m = re.search(r"val_center=([0-9.]+)", f)
    if m and float(m.group(1)) < bv:
        bv, best = float(m.group(1)), f
print(best or "")
PY
)
echo "best ckpt: $BEST"
CFG="$ROOT/lightning_logs/version_0/config.yaml"

# 4. test (writes mota_gt/pred under a new version dir)
( cd "$TT" && python world_track.py test -c "$REPO/$CFG" --ckpt "$REPO/$BEST" \
    --trainer.default_root_dir "$REPO/$ROOT" )

# 5. locate newest mota files + 3-way compare
GT=$(ls -t "$ROOT"/lightning_logs/version_*/mota_gt.txt | head -1)
PR=$(ls -t "$ROOT"/lightning_logs/version_*/mota_pred.txt | head -1)
echo "### $SCENE compare:"
python scripts/tracktacular/bev_compare.py --gt "$GT" --tt-pred "$PR" \
    --current-dir "output/eval/clean_${SCENE}" \
    --anchor-dir  "output/eval/anchor_${SCENE}" \
    --env "$ENV" --frame-step "$STEP" | tee "$ROOT/compare.txt"
