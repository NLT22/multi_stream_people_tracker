# Multi-Stream People Tracker

Production-focused DeepStream 9.0 / pyservicemaker pipeline for multi-camera
people tracking and cross-camera identity assignment on MMPTracking data.

The current system is intentionally narrow:

- detector: YOLO11n fine-tuned on MMPTracking_short
- tracker: NvDCF
- ReID: Swin-Tiny as a secondary nvinfer (SGIE) on person crops
- global IDs: live buffered MTMC grouping, evaluated per environment

Experimental training scripts, older model variants, NvDeepSORT configs, and
ablation presets are archived under `old_stuff/retired_20260620/`.

## Production Presets

| Preset | Use |
|--------|-----|
| `configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml` | Quality default. Best known 20-cam target result. |
| `configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml` | Lower-VRAM/FPS-headroom preset. SGIE still drives global ReID; NvDCF internal ReID is disabled. |

Production model/config chain:

```text
pipeline_mmp_nvdcf_online_sgie.yaml
  detector -> configs/models/nvinfer_yolov11_mmp.yml
           -> models/yolov11/yolo11n_mmp.onnx

  SGIE ReID -> configs/models/nvinfer_reid_swin_sgie_all.yml
            -> models/reid/swin_tiny_mmp_reid_all.onnx

  tracker -> configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml
```

## Current Verified Result

20-camera mixed-environment eval, 600 seconds, grouped per environment:

```text
quality preset:
  avg FPS/cam: 9.99
  avg VRAM:    ~12.7 GB
  mean IDF1:   0.8344

performance preset:
  avg FPS/cam: 10.60
  avg VRAM:    ~9.34 GB
  mean IDF1:   0.8098
```

Retail is still the weak environment. Details and recovery notes are in
[CHANGE.md](CHANGE.md).

## Setup

```bash
./setup_venv.sh
source venv/bin/activate
git lfs pull
```

Requirements: Ubuntu 24.04, NVIDIA driver/CUDA compatible with DeepStream 9.0,
TensorRT, Python 3.12, and the DeepStream pyservicemaker wheel installed by
`setup_venv.sh`.

## Run

Quick non-GPU sanity check:

```bash
scripts/setup/production_smoke.sh
```

Single command using the production default:

```bash
python -m src.main \
  --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  --sources configs/sources/val_20cam_mixed.txt \
  --no-display --no-sync \
  --export-predictions output/eval/manual_run \
  --live-buffered-window 200
```

Long production-style eval:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Performance preset:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

## RTSP Simulation

Use MediaMTX helpers when testing live ingest behavior:

```bash
scripts/eval/mediamtx_loop.sh start dataset/MMPTracking_10minute/val/64pm_office_0
scripts/eval/mediamtx_loop.sh stop
```

The helper prints RTSP URLs and a matching `src.main` command.

## Docker

```bash
docker compose build tracker
docker compose run --rm tracker
```

The compose file now contains only the production tracker service. Training and
dataset-conversion services were archived with the old experiment scripts.

## Layout

```text
configs/    production pipeline/model/tracker/source configs
models/     production ONNX models and YOLO parser library
scripts/    eval and Docker smoke helpers only
src/        production pipeline, config, ReID/gallery, MTMC, metrics
tests/      lightweight regression tests
docs/       production notes and references
report/     dated experiment reports
old_stuff/  archived experiments and retired source files
```

Generated outputs, TensorRT engines, runtime nvinfer configs, Python caches, and
GUI cache files are local artifacts and are gitignored.
