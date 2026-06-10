# Config reference

Presets (`configs/pipelines/pipeline_*.yaml`) pick a detector (`configs/models/*.yml`),
a tracker (`configs/tracker/*.yaml`), sources, and ReID/gallery tuning. CLI flags
override any value. Paths inside an nvinfer/tracker YAML are **relative to that
file's directory**, not the shell CWD.

> Generated `*.runtime_b*_gpu*.yml` and `*.engine` files are build artifacts
> (gitignored) — never edit them; they are regenerated per batch size / GPU.

## Detector — `configs/models/nvinfer_*.yml`

| Key | Meaning |
|-----|---------|
| `onnx-file` / `model-engine-file` | source model; engine auto-built on first run |
| `network-mode` | **0=FP32, 1=INT8, 2=FP16** (FP16 is the deployed mode) |
| `batch-size` | detector batch (set per stream count at runtime) |
| `infer-dims` | input size, e.g. `3;640;640` — must match the ONNX |
| `pre-cluster-threshold` | detection confidence floor; lower = more recall + more FP |
| `nms-iou-threshold` | overlap above which duplicate boxes are merged |
| `interval` | inference every N+1 frames (`0`=every frame; `2`≈2× FPS) |

## Tracker — `configs/tracker/nvdcf_*.yaml`

| Key | Meaning |
|-----|---------|
| `visualTrackerType` | **1 = legacy DCF (deployed; runs ReID cheaply), 2 = VPI DCF (collapses with ReID ~5 FPS)** |
| `reidType` | `0`=off, `2`=REASSOC (in-tracker ReID for cross-camera embeddings) |
| `outputReidTensor` | `1` exports the ReID embedding tensor the Python gallery reads |
| `reidExtractionInterval` | run the Swin ReID every N frames (cost vs freshness) |
| `onnxFile` | the ReID model (`swin_tiny_mmp_reid_all.onnx` = deployed) |
| `maxShadowTrackingAge` | frames a track survives without a detection before it dies; higher = fewer fragments through occlusion |
| `minDetectorConfidence` / `minTrackerConfidence` | gates for keeping detections / tracks |

## ReID / gallery (`reid:` block in a preset, or CLI)

Cross-camera Global ID tuning — see `src/reid/gallery.py` constants and the table
in the top-level [README](../README.md#reid-stabilization-methods). Key ones:
`similarity_threshold`, `id_switch_margin`, `match_ambiguity_margin`,
`tracklet_window`, `geometry_assignment_mode`, and the micro-batch fusion knobs
(`--micro-batch-fusion`, `--fusion-interval`, `--fusion-threshold`).

## Which preset

| Preset | Use |
|--------|-----|
| `pipeline_mmp_10cam_quality.yaml` | **deployed** MMPTracking_short path (legacy DCF + Swin ReID + fusion) |
| `pipeline_mmp_nvdcf_realtime_baseline.yaml` | frozen baseline for A/B |
| others (`*sgie*`, `*20cam*`, `*retail*`) | experimental — not the blessed path |
