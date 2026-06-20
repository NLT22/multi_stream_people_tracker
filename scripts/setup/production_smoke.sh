#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PYTHON:-./venv/bin/python}"

required=(
  configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml
  configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
  configs/models/nvinfer_yolov11_mmp.yml
  configs/models/nvinfer_reid_swin_sgie_all.yml
  configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml
  configs/tracker/nvdcf_accuracy_mmp_recall_sgie_reid0.yaml
  configs/sources/val_20cam_mixed.txt
  models/yolov11/yolo11n_mmp.onnx
  models/reid/swin_tiny_mmp_reid_all.onnx
  models/yolov8/libnvds_infercustomparser_yolov8.so
  scripts/eval/run_long_eval.sh
)

for path in "${required[@]}"; do
  test -e "$path" || { echo "missing: $path" >&2; exit 1; }
done

"$PY" - <<'PY'
from pathlib import Path
from src.config.runtime import DEFAULT_CONFIG_PATH, _load_defaults

assert DEFAULT_CONFIG_PATH == "configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml"
d = _load_defaults(DEFAULT_CONFIG_PATH)
assert d["nvinfer_config"] == "configs/models/nvinfer_yolov11_mmp.yml"
assert d["reid_sgie_config"] == "configs/models/nvinfer_reid_swin_sgie_all.yml"
assert d["tracker_config"] == "configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml"

import src.mtmc
assert src.mtmc.__all__ == []

for retired in [
    "src/mtmc/incremental_mtmc.py",
    "src/mtmc/run_incremental.py",
    "src/eval/offline_anchor.py",
]:
    assert not Path(retired).exists(), retired

print("production smoke OK")
PY
