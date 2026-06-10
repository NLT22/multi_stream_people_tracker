# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **DeepStream 9.0** (`/opt/nvidia/deepstream/deepstream-9.0/`) must be installed locally.
- **pyservicemaker** is NOT on PyPI — it is installed from the DeepStream SDK wheel via `setup_venv.sh`.
- Python 3.12, CUDA 13.1, TensorRT 10.14, Ubuntu 24.04 with RTX 3050Ti (4GB VRAM).

## Setup & Running

```bash
# One-time venv setup (installs pyservicemaker from DeepStream SDK wheel)
./setup_venv.sh
source venv/bin/activate

# Run the full pipeline (default: mtmc_4cam preset)
python -m src.main

# Run with MMP dataset scene
python -m src.main \
    --config configs/pipelines/pipeline_mmp.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0
```

## Tests

```bash
source venv/bin/activate
python -m pytest tests/test_geometry.py -v

# Run a single test
python -m pytest tests/test_geometry.py::test_foot_to_world -v
```

There are no other automated tests — the main validation workflow is the pipeline eval loop described in `Old materials/COMMANDS.md`.

## Docker

```bash
# Build DeepStream image
docker compose build tracker

# Run pipeline scene
docker compose run --rm tracker \
    python3 -m src.main \
        --config configs/pipelines/pipeline_mmp.yaml \
        --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
        --no-display --no-sync \
        --export-predictions output/eval/mmp_lobby0

# Train YOLO (PyTorch image, not DeepStream)
docker compose run --rm yolo_train

# Train ReID
docker compose run --rm reid_train_mmp
```

Docker services: `tracker` (DeepStream), `yolo_train` (PyTorch), `reid_train_mmp` (PyTorch). Training services use `shm_size: "16gb"` to avoid DataLoader `/dev/shm` errors.

If you previously ran with `sudo docker compose`, fix file ownership before switching to host venv:
```bash
sudo chown -R $USER:$USER output dataset/mmp_yolo models/yolov11
```

## Eval Pipeline

The typical workflow for one scene:

```bash
# 1. Export predictions
python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync --export-predictions output/eval/mmp_lobby0

# 2. Nearline merge (geometry-assisted)
python -m src.eval.nearline_merge \
    --pred-dir output/eval/mmp_lobby0 --out-dir output/eval/mmp_lobby0_nearline \
    --threshold 0.65 --margin 0.03 --geo-weight 0.25 \
    --mmp-short-root dataset/MMPTracking_short --scene lobby_0

# 3. Eval metrics
python -m src.eval.metrics_mmp \
    --short-root dataset/MMPTracking_short --scene lobby_0 \
    --pred-dir output/eval/mmp_lobby0_nearline
```

See `Old materials/COMMANDS.md` for full commands including MTA, Wildtrack, sweeps, and benchmarks.

## Architecture

### Two-Layer Identity System

1. **NvDeepSORT/NvDCF tracker** (`nvtracker`) — per-camera local track IDs + ReID tensor export.
2. **CrossCameraGalleryProbe** (`src/reid/gallery.py`) — matches tracker embeddings across cameras to produce stable Global IDs.

### Pipeline Topology

```
[nvurisrcbin × N] → [nvstreammux] → [nvinfer/YOLO] → [nvtracker]
                                                           │
                                           [SourceIdCollectorProbe]   ← pre-tiler (source_id is exact here)
                                                           │
                                           [nvmultistreamtiler]
                                                           │
                                           [CrossCameraGalleryProbe] ← post-tiler (draws GID labels)
                                                           │
                                                    [nvosdbin] → sink
```

**Critical**: `SourceIdCollectorProbe` must be attached **pre-tiler** because `source_id` is exact there. Post-tiler, source_id must be geometrically inferred from tile coordinates (unreliable). When `--export-predictions` is used, the gallery probe automatically uses pretiler mode.

### Key Source Files

- `src/main.py` — thin entry point (`main()` orchestration only)
- `src/config/args.py` — CLI argument parsing
- `src/config/runtime.py` — build defaults dict from YAML preset + gallery tuning
- `src/pipeline/runner.py` — assembles all GStreamer/pyservicemaker elements + `run()` (the production builder; replaced the old unused `builder.py`)
- `src/pipeline/probes.py` — metadata probe callbacks (BatchMetadataOperator subclasses)
- `src/pipeline/sources.py` — URI loading for video files, folders, RTSP
- `src/pipeline/engine_prep.py` — dynamic TensorRT engine generation per batch size
- `src/reid/gallery.py` — CrossCameraGalleryProbe + SourceIdCollectorProbe + all ReID tuning constants
- `src/reid/matching.py` — pure cosine / mean-embedding / Hungarian helpers
- `src/reid/geometry.py` — ground-plane geometry from MMPTracking calibration JSONs
- `src/config/loader.py` — PipelineConfig YAML loader
- `src/eval/metrics_mmp.py` — MOTA/IDF1/Global IDF1 for MMPTracking_short
- `src/eval/nearline_merge.py` — delayed geometry+embedding-assisted Global ID remapping

### Config Presets

| File | Dataset | Notes |
|------|---------|-------|
| `configs/pipelines/pipeline.yaml` | mtmc_4cam | default |
| `configs/pipelines/pipeline_mta.yaml` | MTA 6-cam | NvDeepSORT Swin-MTA |
| `configs/pipelines/pipeline_mmp.yaml` | MMPTracking_short | MMP fine-tuned detector + Swin ReID |
| `configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml` | MMPTracking_short | NvDCF realtime, no online merge |
| `configs/pipelines/pipeline_mmp_nvdcf_online.yaml` | MMPTracking_short | NvDCF + online global merge |

### Metadata Iteration

`batch_meta.frame_items` and `frame_meta.object_items` are **iterators, not lists**. Do not call `len()` directly. If multiple passes are needed: `objects = list(frame_meta.object_items)`.

### TensorRT Engines

First run on a new GPU auto-builds `.engine` files (1–3 min for YOLO11n). Engines are saved next to their ONNX files under `models/`. Do not commit `.engine` files — they are GPU/driver-specific and `.gitignore`d.

Config file paths inside nvinfer YAML configs are **relative to the config file's directory**, not the shell CWD.

### VRAM Pressure (RTX 3050Ti 4GB)

- Use `--tile-w 640 --tile-h 360` for smaller tiles.
- Add `interval: 2` in nvinfer config to skip inference frames.
- For ReID training OOM: `--pk-p 16 --pk-k 4 --accum-steps 4 --grad-ckpt`.
- `retail_*` scenes with 6 cameras may need `nvdcf_accuracy_mmp_retail_lowmem.yaml`.

### Training Custom Models

```bash
# YOLO detector on MMPTracking_short
python scripts/mmp_to_yolo.py          # convert dataset
python scripts/train_yolo_mmp.py       # fine-tune YOLO11n

# Swin-Tiny ReID on MMPTracking_short
python scripts/finetune_reid_mmp.py    # outputs output/reid_mmp/swin_tiny_mmp_reid.onnx
```

Best YOLO warm-start: use MTA model (`output/train/yolo11n_mta/weights/best.pt`).
Best ReID warm-start: use MTA ReID model (`output/reid_v2/best.pth`).

## Regression Anchors

| Scene | Preset | Global IDF1 |
|-------|--------|-------------|
| MTA (offline merge) | `pipeline_mta.yaml` + `nvdeepsort_reid_swin_mta.yaml` | 0.5801 |
| `lobby_0` (nearline) | `pipeline_mmp_nvdcf_realtime_baseline.yaml` | 0.8365 |
| `industry_safety_0` (nearline) | same | 0.8360 |

Current nearline best config: `threshold=0.62`, `margin=0.02`, `geo_weight=0.25`, `geo_min_overlaps=8`, `window_frames=125`.
