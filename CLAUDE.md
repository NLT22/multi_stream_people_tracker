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

# Run the production pipeline on the current source list
# Default = reid0 perf preset: same honest IDF1 as the quality preset (~0.81) but
# faster (~10.6 FPS/cam) and leaner (~3.5 GB at maxTargetsPerStream=40).
# See Config Presets / Regression Anchors (VRAM is driven by maxTargetsPerStream).
python -m src.main \
    --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
    --sources configs/sources/val_20cam_mixed.txt \
    --no-display --no-sync \
    --export-predictions output/eval/manual_run \
    --live-buffered-window 200
```

## Tests

```bash
source venv/bin/activate
python -m pytest tests/test_geometry.py -v

# Run a single test
python -m pytest tests/test_geometry.py::test_foot_to_world -v
```

The unit tests under `tests/` (run `python tests/test_*.py` or `pytest`) cover the ReID/gallery/eval logic; the end-to-end validation is the pipeline eval loop (export → nearline merge → `metrics_mmp`). Archived commands live in `old_stuff/COMMANDS.md`.

## Docker

```bash
# Build DeepStream image
docker compose build tracker

# Run production tracker service
docker compose run --rm tracker
```

Docker service: `tracker` (DeepStream). Training services were archived with the
retired training scripts.

If you previously ran with `sudo docker compose`, fix file ownership before switching to host venv:
```bash
sudo chown -R $USER:$USER output dataset/mmp_yolo models/yolov11
```

## Eval Pipeline

Production long eval:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

See `old_stuff/COMMANDS.md` for archived commands (MTA, Wildtrack, sweeps, benchmarks). MTA/Wildtrack/MTMC/FastReID/YOLOv8/pose support has been moved to `old_stuff/` — the pipeline is now MMP-only.

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
- `src/pipeline/runner.py` — assembles all GStreamer/pyservicemaker elements + `run(PipelineRunConfig)` (the production builder)
- `src/pipeline/run_config.py` — `PipelineRunConfig` dataclass (all `run()` parameters)
- `src/pipeline/source_plan.py` — turns args into a `SourcePlan` (sources + GT + geometry)
- `src/pipeline/sources.py` — URI loading for video files, folders, RTSP
- `src/pipeline/engine_prep.py` — dynamic TensorRT engine generation per batch size
- `src/reid/gallery.py` — `CrossCameraGalleryProbe` (thin DeepStream adapter) + `gallery_{rows,conflict,assignment,merge}` mixins
- `src/reid/metadata.py` — `SourceIdCollectorProbe` (pre-tiler source_id + embedding reader)
- `src/reid/config.py` — `ReIDConfig` dataclass (all ReID / Global-ID tuning)
- `src/reid/{gallery_store,tracklet_store,detection_row}.py` — gallery/tracklet state + the per-frame row dataclass
- `src/reid/matching.py` — pure cosine / mean-embedding / Hungarian helpers
- `src/reid/geometry.py` — ground-plane geometry from MMPTracking calibration JSONs
- `src/config/loader.py` — PipelineConfig YAML loader
- `src/eval/mmp_metrics/` — MOTA/IDF1/Global IDF1 engine (`core.py`) + CLI (`cli.py`); `src/eval/metrics_mmp.py` is a thin `-m` shim
- `src/mtmc/live_buffered.py` — production live-buffered MTMC consumer for long eval

### Config Presets

| File | Dataset | Notes |
|------|---------|-------|
| `configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml` | MMPTracking_short / mixed 20cam | **Production default (recommended):** NvDCF internal ReID off, SGIE drives global IDs. Ties the quality preset on IDF1 (~0.81) but faster/leaner. |
| `configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml` | MMPTracking_short / mixed 20cam | Quality preset: YOLO11 + NvDCF (reidType:2) + SGIE ReID. Double-ReID buys ~0 IDF1 over reid0 (global IDs come from SGIE); keep only if local-track continuity matters. |

### Metadata Iteration

`batch_meta.frame_items` and `frame_meta.object_items` are **iterators, not lists**. Do not call `len()` directly. If multiple passes are needed: `objects = list(frame_meta.object_items)`.

### TensorRT Engines

First run on a new GPU auto-builds `.engine` files (1–3 min for YOLO11n). Engines are saved next to their ONNX files under `models/`. Do not commit `.engine` files — they are GPU/driver-specific and `.gitignore`d.

Config file paths inside nvinfer YAML configs are **relative to the config file's directory**, not the shell CWD.

### VRAM Pressure

- **`maxTargetsPerStream` is the dominant VRAM lever** — NvDCF pre-allocates per-target
  state (DCF filters + ReID buffers) for `maxTargetsPerStream × streams`, so 220 vs 40 is
  the difference between ~9–13 GB and ~3.5–4 GB at 20 cams. MMP never needs >40.
- Prefer `pipeline_mmp_nvdcf_online_sgie_reid0.yaml` for lower VRAM.
- Use `--no-display --no-sync` for evaluation/soak runs.
- If more headroom is needed, test SGIE `interval` changes before changing the detector.

### Training Custom Models

Training and dataset-conversion scripts are archived under
`old_stuff/retired_20260620/`. The root project is now production/eval focused.
Restore archived scripts only when intentionally starting a new training cycle.

## Regression Anchors

Use **honest single-pass full-GT** as the canonical measure (every frame processed once, no loop,
no GT trimming) — score with `scripts/eval/score_full_mmp_val.py` AFTER `live_buffered --once` finishes.

**Full val (all 24 scenes, buffered ID, reid0)** — 2026-06-25, `score_full_mmp_val.py`:

| Environment | Scenes | Mean IDF1 |
|-------------|--------|-----------|
| Lobby       | 4      | **0.893** |
| Office      | 3      | **0.878** |
| Industry    | 5      | **0.829** |
| Café        | 4      | **0.823** |
| Retail      | 8      | 0.616     |
| **Overall** | **24** | **0.774** |

**Single-scene (_0 only, 5 scenes)** — older reference numbers:

| Eval | Preset | Mean Global IDF1 |
|-------|--------|-------------|
| honest single-pass full-GT (5 scenes, _0 only) | `..._reid0.yaml` (default) | 0.8109 (~10.6 FPS/cam) |
| honest single-pass full-GT (5 scenes, _0 only) | `..._sgie.yaml` (quality) | 0.8132 (~9.5 FPS/cam) |
| 600s looped, processed-segment (optimistic) | `..._sgie.yaml` | 0.8344 |
| 600s looped, full untrimmed GT (over-penalized) | `..._sgie.yaml` | 0.758 |

The 0.811 single-scene mean is not wrong — it just evaluated only 1 scene per env. The full-val 0.774 is the honest number across all 24 val scenes. Retail pulls it down most (8 scenes, 0.427–0.675 range).

**VRAM depends almost entirely on `maxTargetsPerStream`, not the preset/model** (measured
2026-06-25, 20-cam, nvidia-smi steady-state). NvDCF pre-allocates per-target state (DCF
correlation filters; plus ReID buffers when `reidType:2`) for the full `maxTargetsPerStream ×
streams` capacity, regardless of the actual ~5–15 people/cam:

| Preset | `reidType` | `maxTargetsPerStream` | VRAM (20-cam) |
|--------|-----------|----------------------|---------------|
| reid0 (current default) | 0 | 40 | **~3.5 GB** |
| reid0 (pre-4.4-audit) | 0 | 220 | **~9.4 GB** ← the older figure; not wrong, just `maxTargets=220` |
| quality | 2 | 40 | **~4.2 GB** (the ReID model itself adds only ~0.7 GB) |
| quality (as shipped) | 2 | 220 | **~12.8 GB** |

So the old "9.4 GB" was reid0 at `maxTargetsPerStream=220`; the 4.4 audit cut it to 40 → ~3.5 GB.
Lowering `maxTargetsPerStream` is the single biggest VRAM lever (MMP never needs 220).

Current nearline best config: `threshold=0.62`, `margin=0.02`, `geo_weight=0.25`, `geo_min_overlaps=8`, `window_frames=125`.
