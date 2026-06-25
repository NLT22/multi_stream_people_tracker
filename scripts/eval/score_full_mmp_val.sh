#!/usr/bin/env bash
# Score exported full-val predictions with buffered IDs (live_buffered --once).
#
# For each scene that has cam_*_predictions.csv but no _eval_assign.csv yet:
#   1. Run live_buffered --once  → produces _eval_assign.csv
#   2. Score with metrics_mmp    → prints per-scene IDF1
#
# Can be run while run_full_mmp_val.sh is still exporting — it skips in-progress
# scenes (no predictions.csv) and already-scored ones (_eval_assign.csv exists).
#
# Usage:
#   bash scripts/eval/score_full_mmp_val.sh [export_dir] [val_root]
#
# Defaults:
#   export_dir : output/eval/full_mmp_val
#   val_root   : dataset/MMPTracking_10minute/val

set -euo pipefail

EXPORT_ROOT="${1:-output/eval/full_mmp_val}"
VAL_ROOT="${2:-dataset/MMPTracking_10minute/val}"
PYTHON="${PYTHON:-./venv/bin/python3}"

WINDOW_CHUNKS="${WINDOW_CHUNKS:-1}"          # 1 chunk × 200f = 200f window (canonical)
ASSIGN_THR="${ASSIGN_THR:-0.40}"

declare -A SCENE_IDF1

for scene_dir in "${EXPORT_ROOT}"/64pm_*/; do
    [[ -d "${scene_dir}" ]] || continue
    scene=$(basename "${scene_dir}")

    # Skip scenes with no predictions yet (pipeline still running)
    ncam=$(ls "${scene_dir}"cam_*_predictions.csv 2>/dev/null | wc -l)
    [[ "${ncam}" -eq 0 ]] && { echo "[${scene}] no predictions yet, skipping"; continue; }

    echo ""
    echo "── ${scene} ──────────────────────────────"

    # Step 1: live_buffered --once (skip if already done)
    assign_csv="${scene_dir}_eval_assign.csv"
    if [[ ! -f "${assign_csv}" ]]; then
        echo "  live_buffered --once ..."
        "${PYTHON}" -m src.mtmc.live_buffered \
            --export-dir "${scene_dir}" \
            --window-chunks "${WINDOW_CHUNKS}" \
            --assign-thr "${ASSIGN_THR}" \
            --assign-csv "${assign_csv}" \
            --once
        echo "  → ${assign_csv}"
    else
        echo "  _eval_assign.csv already exists, skipping live_buffered."
    fi

    # Step 2: score with metrics_mmp using buffered assignments
    # Remap global_id in predictions using _eval_assign.csv, then score
    score_out="${scene_dir}metrics.json"
    echo "  scoring ..."
    "${PYTHON}" - <<PYEOF
import json, sys
from pathlib import Path
import pandas as pd

scene_dir = Path("${scene_dir}")
val_root  = Path("${VAL_ROOT}")
scene     = "${scene}"
assign_csv = scene_dir / "_eval_assign.csv"
score_out  = scene_dir / "metrics.json"

# Load buffered assignments: group, cam_id, frame_no, local_track_id → global_id
assign = pd.read_csv(assign_csv)
# build lookup: (cam_id, frame_no, local_track_id) → global_id
gid_map = {
    (int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
    for r in assign.itertuples()
}

# Determine cam IDs from prediction files
cam_files = sorted(scene_dir.glob("cam_*_predictions.csv"))
source_ids = [int(p.stem.split("_")[1]) for p in cam_files]

# Map source_id → GT cam_id (1-based, ordered by source_id)
# source_id 0 = cam1.mp4 = gt_cam1.csv, etc.
scene_data = val_root / scene
gt_cam_ids = sorted(int(p.stem[3:]) for p in scene_data.glob("cam*.mp4"))

results_per_cam = []
all_gt, all_pred = {}, {}

for src_id, gt_cam_id in zip(source_ids, gt_cam_ids):
    pred_path = scene_dir / f"cam_{src_id}_predictions.csv"
    # prefer _clean GT
    gt_path = scene_data / f"gt_cam{gt_cam_id}_clean.csv"
    if not gt_path.exists():
        gt_path = scene_data / f"gt_cam{gt_cam_id}.csv"
    if not pred_path.exists() or not gt_path.exists():
        continue

    pred = pd.read_csv(pred_path)
    gt   = pd.read_csv(gt_path)

    # Remap global_id using buffered assignments
    pred["global_id"] = [
        gid_map.get((src_id, int(f), int(t)), -1)
        for f, t in zip(pred["frame_no_cam"], pred["local_track_id"])
    ]
    # Drop unassigned detections
    pred = pred[pred["global_id"] >= 0].copy()
    # Rename frame_no_cam → frame so _eval_global_idf1 can find it
    pred = pred.rename(columns={"frame_no_cam": "frame"})

    all_gt[gt_cam_id]   = gt
    all_pred[gt_cam_id] = pred

try:
    import sys
    sys.path.insert(0, ".")
    from src.eval.mmp_metrics.core import _eval_global_idf1
    result = _eval_global_idf1(all_gt, all_pred, iou_threshold=0.5)
    idf1 = result.get("global_idf1", result.get("mean_idf1", None))
    out = {"scene": scene, "global_idf1": idf1, **result}
    score_out.write_text(json.dumps(out, indent=2))
    print(f"  IDF1 = {idf1:.4f}" if idf1 is not None else "  IDF1 = N/A")
except Exception as e:
    print(f"  scoring failed: {e}", file=sys.stderr)
PYEOF

    if [[ -f "${score_out}" ]]; then
        idf1=$("${PYTHON}" -c "
import json
d=json.load(open('${score_out}'))
v=d.get('idf1') or d.get('global_idf1') or d.get('mean_idf1')
print(f'{v:.4f}' if v else 'N/A')
" 2>/dev/null || echo "N/A")
        SCENE_IDF1["${scene}"]="${idf1}"
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Scored scenes"
echo "=========================================="
printf "%-32s  %s\n" "Scene" "Global IDF1"
printf "%-32s  %s\n" "--------------------------------" "-----------"

declare -A ENV_SUM ENV_COUNT
total=0; count=0
for scene in $(echo "${!SCENE_IDF1[@]}" | tr ' ' '\n' | sort); do
    idf1="${SCENE_IDF1[$scene]}"
    printf "%-32s  %s\n" "${scene}" "${idf1}"
    if [[ "${idf1}" != "N/A" ]]; then
        total=$("${PYTHON}" -c "print(${total} + ${idf1})")
        count=$((count + 1))
        env=$(echo "${scene}" | sed 's/64pm_//' | sed 's/_[0-9]*$//')
        ENV_SUM["${env}"]=$("${PYTHON}" -c "print(${ENV_SUM[${env}]:-0} + ${idf1})")
        ENV_COUNT["${env}"]=$(( ${ENV_COUNT[${env}]:-0} + 1 ))
    fi
done

if [[ ${count} -gt 0 ]]; then
    echo ""
    for env in $(echo "${!ENV_SUM[@]}" | tr ' ' '\n' | sort); do
        mean_env=$("${PYTHON}" -c "print(f'{${ENV_SUM[$env]} / ${ENV_COUNT[$env]}:.4f}')")
        printf "  %-22s  %s  (%d scenes)\n" "${env}" "${mean_env}" "${ENV_COUNT[$env]}"
    done
    mean=$("${PYTHON}" -c "print(f'{${total} / ${count}:.4f}')")
    printf "%-32s  %s\n" "--------------------------------" "-----------"
    printf "%-32s  %s\n" "MEAN (${count} scenes)" "${mean}"
fi
