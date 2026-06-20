# Model Artifacts

Only production ONNX files remain in the main model folders. Older base models,
10-minute retrains, duplicate variants, and rejected ablations are archived under
`old_stuff/retired_20260620/models/`.

## Active Models

| File | Used by | Notes |
|------|---------|-------|
| `models/yolov11/yolo11n_mmp.onnx` | `configs/models/nvinfer_yolov11_mmp.yml` | YOLO11n detector fine-tuned on MMPTracking_short. |
| `models/reid/swin_tiny_mmp_reid_all.onnx` | `configs/models/nvinfer_reid_swin_sgie_all.yml` | Swin-Tiny ReID model used by the SGIE production path. |
| `models/yolov8/libnvds_infercustomparser_yolov8.so` | detector nvinfer config | Custom YOLO parser library. Name is historical; current detector is YOLO11. |

TensorRT `.engine` files are generated locally on first run and are intentionally
not tracked. They are GPU, driver, TensorRT, batch-size, and precision specific.

## If A Model Is Missing

```bash
git lfs pull
```

If a TensorRT engine is missing or stale, run the pipeline once and DeepStream
will rebuild it.
