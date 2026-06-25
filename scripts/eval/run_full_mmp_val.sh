#!/usr/bin/env bash
# Run pipeline eval on the full MMPTracking original validation set (24 scenes).
#
# Uses MMPTracking_10minute/val — all 24 scenes already converted to MP4.
# Processes one scene at a time (4-6 cams each) to stay within VRAM budget.
# Exports predictions per scene then scores with metrics_mmp.
#
# Usage:
#   bash scripts/eval/run_full_mmp_val.sh [output_dir] [config_yaml]
#
# Defaults:
#   output_dir : output/eval/full_mmp_val
#   config_yaml: configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml

set -euo pipefail

OUTDIR="${1:-output/eval/full_mmp_val}"
PIPECFG="${2:-configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml}"
PYTHON="${PYTHON:-./venv/bin/python3}"
SOURCES_DIR="configs/sources"
DATA_ROOT="dataset/MMPTracking_10minute/val"
LIVE_BUFFERED_WINDOW="${LIVE_BUFFERED_WINDOW:-200}"

# Collect per-scene source lists (sorted so envs are grouped)
mapfile -t SCENE_LISTS < <(ls "${SOURCES_DIR}"/val_full_mmp_64pm_*.txt 2>/dev/null | sort)

if [[ ${#SCENE_LISTS[@]} -eq 0 ]]; then
    echo "ERROR: No per-scene source lists found in ${SOURCES_DIR}/"
    echo "Expected: configs/sources/val_full_mmp_64pm_<scene>.txt"
    exit 1
fi

echo "=========================================="
echo " Full MMPTracking val eval"
echo " Scenes : ${#SCENE_LISTS[@]}"
echo " Config : ${PIPECFG}"
echo " Output : ${OUTDIR}"
echo "=========================================="
mkdir -p "${OUTDIR}"

declare -A SCENE_IDF1

for src_file in "${SCENE_LISTS[@]}"; do
    # e.g. val_full_mmp_64pm_cafe_shop_0.txt -> 64pm_cafe_shop_0
    scene=$(basename "${src_file}" .txt | sed 's/val_full_mmp_//')
    pred_dir="${OUTDIR}/${scene}"
    scene_data="${DATA_ROOT}/${scene}"

    echo ""
    echo "── ${scene} ──────────────────────────────"

    # Run pipeline → export predictions
    "${PYTHON}" -m src.main \
        --config "${PIPECFG}" \
        --sources "${src_file}" \
        --no-display --no-sync \
        --export-predictions "${pred_dir}" \
        --live-buffered-window "${LIVE_BUFFERED_WINDOW}"

    # Build GT args — prefer *_clean.csv if present (exact-source relabeled)
    gt_args=()
    for cam_dir in "${scene_data}"/cam*.mp4; do
        cam=$(basename "${cam_dir}" .mp4 | sed 's/cam//')
        clean_csv="${scene_data}/gt_cam${cam}_clean.csv"
        raw_csv="${scene_data}/gt_cam${cam}.csv"
        if [[ -f "${clean_csv}" ]]; then
            gt_args+=(--gt "${cam}:${clean_csv}")
        elif [[ -f "${raw_csv}" ]]; then
            gt_args+=(--gt "${cam}:${raw_csv}")
        fi
    done

    if [[ ${#gt_args[@]} -gt 0 ]]; then
        score_out="${pred_dir}/metrics.json"
        "${PYTHON}" -m src.eval.metrics_mmp \
            --pred-dir "${pred_dir}" \
            "${gt_args[@]}" \
            --out "${score_out}" 2>/dev/null || true

        if [[ -f "${score_out}" ]]; then
            idf1=$("${PYTHON}" -c "
import json
d = json.load(open('${score_out}'))
v = d.get('global_idf1', d.get('mean_idf1', None))
print(f'{v:.4f}' if v is not None else 'N/A')
" 2>/dev/null || echo "N/A")
            SCENE_IDF1["${scene}"]="${idf1}"
            echo "  IDF1=${idf1}"
        fi
    else
        echo "  WARNING: no GT CSVs found for ${scene}, skipping scoring."
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Results summary"
echo "=========================================="
printf "%-32s  %s\n" "Scene" "IDF1"
printf "%-32s  %s\n" "--------------------------------" "------"

# Accumulate per-env for group means
declare -A ENV_SUM ENV_COUNT

total=0; count=0
for scene in $(echo "${!SCENE_IDF1[@]}" | tr ' ' '\n' | sort); do
    idf1="${SCENE_IDF1[$scene]}"
    printf "%-32s  %s\n" "${scene}" "${idf1}"
    if [[ "${idf1}" != "N/A" ]]; then
        total=$("${PYTHON}" -c "print(${total} + ${idf1})")
        count=$((count + 1))
        # env = remove trailing _N from scene name
        env=$(echo "${scene}" | sed 's/_[0-9]*$//')
        ENV_SUM["${env}"]=$("${PYTHON}" -c "print(${ENV_SUM[${env}]:-0} + ${idf1})")
        ENV_COUNT["${env}"]=$(( ${ENV_COUNT[${env}]:-0} + 1 ))
    fi
done

echo ""
echo "Per-environment means:"
for env in $(echo "${!ENV_SUM[@]}" | tr ' ' '\n' | sort); do
    mean_env=$("${PYTHON}" -c "print(f'{${ENV_SUM[$env]} / ${ENV_COUNT[$env]}:.4f}')")
    printf "  %-28s  %s  (%d scenes)\n" "${env}" "${mean_env}" "${ENV_COUNT[$env]}"
done

if [[ ${count} -gt 0 ]]; then
    mean=$("${PYTHON}" -c "print(f'{${total} / ${count}:.4f}')")
    echo ""
    printf "%-32s  %s\n" "--------------------------------" "------"
    printf "%-32s  %s\n" "MEAN all ${count} scenes" "${mean}"
fi

# Save summary
summary="${OUTDIR}/summary.txt"
{
    echo "Full MMPTracking val eval — ${PIPECFG}"
    echo ""
    printf "%-32s  %s\n" "Scene" "IDF1"
    printf "%-32s  %s\n" "--------------------------------" "------"
    for scene in $(echo "${!SCENE_IDF1[@]}" | tr ' ' '\n' | sort); do
        printf "%-32s  %s\n" "${scene}" "${SCENE_IDF1[$scene]}"
    done
    if [[ ${count} -gt 0 ]]; then
        echo ""
        for env in $(echo "${!ENV_SUM[@]}" | tr ' ' '\n' | sort); do
            mean_env=$("${PYTHON}" -c "print(f'{${ENV_SUM[$env]} / ${ENV_COUNT[$env]}:.4f}')")
            printf "  %-28s  %s  (%d scenes)\n" "${env}" "${mean_env}" "${ENV_COUNT[$env]}"
        done
        printf "%-32s  %s\n" "MEAN all ${count} scenes" "${mean}"
    fi
} > "${summary}"
echo ""
echo "Summary written: ${summary}"
