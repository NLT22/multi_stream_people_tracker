# Model Artifacts

ONNX files used by the DeepStream pipeline. `.engine` files are **not** committed
— they are GPU/driver-specific and auto-built next to each ONNX on first run.

"Deployed" below means **referenced by a production pipeline config**, traced
through the config chain:

```
configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml   (and ..._online.yaml)
  detector ->  configs/models/nvinfer_yolov11_mmp.yml      ->  yolo11n_mmp.onnx
  reid     ->  configs/tracker/nvdcf_accuracy_mmp_recall.yaml ->  swin_tiny_mmp_reid_all.onnx
```

These two are the regression-anchor models (see `CLAUDE.md` → Regression Anchors).
Everything else is an experiment — **do not point a production config at an
experiment model without re-validating the MMPTracking_short IDF1 anchors.**

## Detectors — `models/yolov11/`

| File | Status | Trained on | Notes |
|------|--------|-----------|-------|
| `yolo11n_mmp.onnx` | **DEPLOYED** | MMPTracking_short (all scenes, orig GT) | Production detector |
| `yolo11n.onnx` | base | COCO | Ultralytics YOLO11n warm-start |
| `yolo11n_10min.onnx` | experiment | MMPTracking_10minute (all scenes) | New larger-data retrain |
| `yolo11n_nonretail.onnx` | experiment | MMPTracking_short, retail excluded | Falsified (report §3): IDF1 neutral |
| `yolo11n_retail_clean.onnx` | experiment | MMPTracking_short, cleaned retail GT | Falsified (report §2): held-out 0.459→0.420 |
| `yolo11n_mmp_416.onnx` | dead dup | — | **Identical md5 to `yolo11n_mmp.onnx`** (not a real 416 model) |

## ReID — `models/reid/`

| File | Status | Trained on | Notes |
|------|--------|-----------|-------|
| `swin_tiny_mmp_reid_all.onnx` | **DEPLOYED** | MMPTracking_short (train-all-nonretail, warm-started) | Production ReID; lobby_0 **0.9031** |
| `swin_tiny_mmp_reid.onnx` | previous deploy | MMPTracking_short | Older ReID; lobby_0 0.8506 |
| `swin_tiny_market1501_aicity156_featuredim256.onnx` | base | Market1501 + AICity | MTA/ImageNet warm-start |
| `swin_tiny_mmp_reid_10min.onnx` | experiment | MMPTracking_10minute (retail excluded) | New larger-data retrain |
| `swin_tiny_mmp_reid_nonretail.onnx` | experiment | MMPTracking_short, retail excluded | Falsified (report §3): over-merge artifact |

> **History:** `nvdcf_accuracy_mmp_recall.yaml` previously pointed at
> `swin_tiny_mmp_reid.onnx` (0.8506) while `swin_tiny_mmp_reid_all.onnx` (0.9031,
> the report's headline number) was only wired into the *quality* config. The
> deployed tracker config now points at `_all`, so the baseline path gets the
> +0.05 too.

## Promoting an experiment to deployed

1. Validate it on the MMPTracking_short anchors (lobby_0, industry_safety_0) and
   confirm Global IDF1 does not regress.
2. Update the production config (`nvinfer_yolov11_mmp.yml` or
   `nvdcf_accuracy_mmp_recall.yaml`) to point at the new ONNX.
3. Update this table and the Regression Anchors in `CLAUDE.md`.
