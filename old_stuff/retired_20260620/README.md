# Retired Project Files — 2026-06-20

This directory keeps files removed from the production path while preserving
their history in Git.

Moved here:

- old pipeline presets
- old NvDeepSORT / NvDCF ablation tracker configs
- old detector/ReID nvinfer configs
- training, dataset-conversion, benchmark, and anchor-guided experiment scripts
- old model variants and rejected/redundant ONNX files
- stale setup helpers that depended on archived training scripts
- old incremental MTMC simulator files and historical anchor notes

The active production path remains in the root project:

```text
configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml
configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
configs/models/nvinfer_yolov11_mmp.yml
configs/models/nvinfer_reid_swin_sgie_all.yml
configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml
configs/tracker/nvdcf_accuracy_mmp_recall_sgie_reid0.yaml
models/yolov11/yolo11n_mmp.onnx
models/reid/swin_tiny_mmp_reid_all.onnx
```

To restore a retired file, move it back with `git mv` and re-run the relevant
smoke/eval command.
