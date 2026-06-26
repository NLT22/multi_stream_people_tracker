# Retired 2026-06-26

Artifacts archived during the project cleanup of 2026-06-26 (after the retail-clean
YOLO retrain was deployed). All are **restorable** — `git mv` back to restore.

## `models/yolo11n_mmp_clean.onnx` + `configs/nvinfer_yolov11_mmp_clean.yml`
The **"all-clean" detector**: YOLO11n retrained after running the verifier label-cleaner
(`old_stuff/retired_20260620/scripts/datasets/clean_yolo_labels.py`) on **every**
environment's labels.

**Why retired:** cleaning all environments over-cut recall on non-retail (people
partially occluded by furniture were wrongly dropped) — full-val IDF1 0.786, non-retail
dropped to 0.835. It was **superseded by the retail-targeted clean** detector
(`models/yolov11/yolo11n_mmp_retailclean.onnx`, the deployed model): cleaning only retail
labels and leaving the other environments intact gave full-val IDF1 **0.798** with
non-retail recall preserved. See `report/26062026.md` and the LaTeX report §3.5.1.

**Restore:** `git mv old_stuff/retired_20260626/models/yolo11n_mmp_clean.onnx models/yolov11/`
and the config back to `configs/models/`; the TensorRT engine rebuilds on first run.

## Not archived here (deliberately kept in place)
- `models/yolov11/yolo11n_mmp.onnx` + `configs/models/nvinfer_yolov11_mmp.yml` — the OLD
  amodal-trained detector, kept as the **rollback** for the deployed retail-clean detector.
- Large gitignored data (old eval exports, old demos, the all-clean dataset) was archived
  under `output/_archive_pre_retailclean/` (see the README there), not in git.
