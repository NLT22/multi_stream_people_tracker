# Model Artifacts

Production + MTMC ONNX files, tracked via **Git LFS**. Older base models, 10-minute
retrains, duplicate variants, and rejected ablations are archived under
`old_stuff/retired_20260620/models/`. Models split into the two pipelines:
**MMP** (production, MMPTracking) and **MTMC** (AI-City warehouse) — see the root
[README](../README.md) and [CLAUDE.md](../CLAUDE.md).

## MMP (production)

| File | Used by | Notes |
|------|---------|-------|
| `models/yolov11/yolo11n_mmp_retailclean.onnx` | `nvinfer_yolov11_mmp_retailclean.yml` | **Deployed detector** (both production presets). Retrained on retail-cleaned labels (precision 0.62→0.94). |
| `models/yolov11/yolo11n_mmp.onnx` | `nvinfer_yolov11_mmp.yml` | Original YOLO11n detector. **Kept for rollback** only. |
| `models/yolov11/yolo11n_mmp_int8.onnx` | `nvinfer_yolov11_mmp_int8.yml` | INT8-quantized detector (experimental; IDF1 preserved, ~0 FPS gain). |
| `models/reid/swin_tiny_mmp_reid_all.onnx` | `nvinfer_reid_swin_sgie_all.yml` | **Deployed Swin-Tiny ReID** for the SGIE path (top1 0.847 / mAP 0.773). |

## MTMC (warehouse — separate pipeline)

| File | Used by | Notes |
|------|---------|-------|
| `models/yolov11/yolo11n_mtmc.onnx` | `nvinfer_yolov11_mtmc[_640].yml` | MTMC detector @960. |
| `models/yolov11/yolo11n_mtmc_1280.onnx` | `nvinfer_yolov11_mtmc_1280.yml` | MTMC detector @1280 (higher recall, slower). |
| `models/yolov11/yolo11n_mtmc_overfit.onnx` | `nvinfer_yolov11_mtmc_overfit.yml` | Overfit diagnostic variant (not for deployment). |
| `models/reid/swin_tiny_mtmc_reid.onnx` | `nvinfer_reid_swin_sgie_mtmc.yml` | MTMC warehouse Swin ReID (warehouse crops; geometry still drives cross-cam ID). |

## Shared

| File | Used by | Notes |
|------|---------|-------|
| `models/yolov8/libnvds_infercustomparser_yolov8.so` | every detector nvinfer config | Custom YOLO bbox parser. Name is historical; the detectors are YOLO11. |

TensorRT `.engine` files are generated locally on first run and are intentionally
not tracked. They are GPU, driver, TensorRT, batch-size, and precision specific.

## If A Model Is Missing

```bash
git lfs pull
```

If a TensorRT engine is missing or stale, run the pipeline once and DeepStream
will rebuild it.
