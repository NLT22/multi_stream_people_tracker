# Multi-Stream People Tracker

Real-time multi-camera people tracking and cross-camera re-identification (MTMC)
on **DeepStream 9.0** / pyservicemaker. Per-camera detection + tracking + ReID
embeddings run on the GPU every frame; a decoupled **micro-batch fusion** stage
clusters embeddings across cameras to assign stable Global IDs — the same
architecture used in production MTMC systems (NVIDIA Metropolis, AI City Challenge
winners), not per-frame online matching.

Primary dataset: **MMPTracking_short**. Detector: YOLO11n fine-tuned on MMP.
Tracker: NvDCF (legacy DCF) + Swin-Tiny ReID.

## Results (MMPTracking_short, RTX 5060 Ti)

| Metric | Value |
|--------|-------|
| Global IDF1 (8 non-retail scenes, avg) | **0.81** |
| Throughput @10 cameras | **37.6 FPS/cam** |
| Throughput @20 cameras | **18.8 FPS/cam** (1.88× real-time) |

Per-scene IDF1 and the throughput methodology are in
[Old materials/report/](Old%20materials/report/).

---

## Setup

### Local (DeepStream 9.0 installed on host)

```bash
./setup_venv.sh           # installs pyservicemaker from the DeepStream SDK wheel
source venv/bin/activate
```

Prerequisites: Ubuntu 24.04, NVIDIA driver 590+, CUDA 13.1, DeepStream 9.0,
Python 3.12, TensorRT 10.14.

### Docker

```bash
docker compose build tracker
docker compose run --rm tracker python3 -m src.main \
    --config configs/pipelines/pipeline_mmp_10cam_quality.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync --micro-batch-fusion \
    --save-video output/video/lobby_0.mp4 \
    --export-predictions output/eval/lobby_0
```

The first run builds TensorRT `.engine` files (1–3 min); they persist under
`models/` via the bind mount. `.engine` files are GPU-specific and gitignored.

---

## Run the pipeline

```bash
source venv/bin/activate

python -m src.main \
    --config configs/pipelines/pipeline_mmp_10cam_quality.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync \
    --micro-batch-fusion \
    --save-video output/video/lobby_0.mp4 \
    --export-predictions output/eval/lobby_0
```

- `--micro-batch-fusion` runs the in-pipeline cross-camera fusion (live Global
  IDs). The annotated video shows near-realtime IDs; the exported predictions
  hold the converged authoritative IDs.
- Drop the flag for the plain gallery baseline.
- Scenes: `lobby_0 lobby_3 cafe_shop_0 cafe_shop_3 industry_safety_0
  industry_safety_4 office_0 office_2` (and the `retail_*` set).

### Evaluate

```bash
python -m src.eval.metrics_mmp \
    --short-root dataset/MMPTracking_short --scene lobby_0 \
    --pred-dir output/eval/lobby_0
```

For an offline (or very long stream) cross-camera merge instead of the in-pipeline
fusion, run the validated post-pass on an export:

```bash
python -m src.eval.online_fusion \
    --pred-dir output/eval/lobby_0 --out-dir output/eval/lobby_0_fused \
    --threshold 0.55 --geo-weight 0.25 \
    --mmp-short-root dataset/MMPTracking_short --scene lobby_0
```

---

## Architecture

```
[nvurisrcbin ×N] → [nvstreammux] → [nvinfer/YOLO11] → [nvtracker + Swin ReID]
                                                              │
                                          [SourceIdCollectorProbe]  (pre-tiler, exact source_id)
                                                              │
                                          [nvmultistreamtiler]
                                                              │
                                          [CrossCameraGalleryProbe]  ← per-camera IDs + live
                                                              │         micro-batch fusion → Global IDs
                                                       [nvosdbin] → sink / video / CSV
```

1. **Perception (per camera, every frame):** YOLO11 detection → NvDCF tracking →
   Swin-Tiny ReID embeddings.
2. **Fusion (micro-batch cadence):** `MicroBatchFusion` clusters tracklet
   embeddings across cameras (geometry-disambiguated) into stable Global IDs.

Key source:

| File | Role |
|------|------|
| `src/main.py` | entry point, pipeline wiring, CLI |
| `src/reid/gallery.py` | per-camera gallery + live fusion wiring |
| `src/reid/micro_batch_fusion.py` | streaming cross-camera fusion engine |
| `src/eval/online_fusion.py` | fusion as an offline/near-realtime post-pass |
| `src/eval/metrics_mmp.py` | MOTA / IDF1 / Global IDF1 |
| `src/reid/geometry.py` | ground-plane geometry from MMP calibration |

Deeper notes (configs, presets, tuning, regression anchors) are in
[CLAUDE.md](CLAUDE.md).

### Runtime modes — production vs experimental

| | Blessed production path | Experimental (opt-in / off by default) |
|---|---|---|
| Detector | `yolo11n_mmp.onnx` (`nvinfer_yolov11_mmp.yml`) | SGIE-decoupled ReID, alternate detectors |
| Tracker | NvDCF **legacy DCF** (`nvdcf_accuracy_mmp_recall_all.yaml`) | NvDeepSORT, VPI DCF (`visualTrackerType:2`) |
| Cross-camera ID | Micro-batch fusion (`--micro-batch-fusion`, cooccur geometry) | trajectory geometry (`--geo-mode trajectory`), pose feet (`src/reid/pose.py`) |

The experimental geometry modes (trajectory, pose) and SGIE/NvDeepSORT paths are
**off by default** — A/B tests showed no gain on the overlapping MMP cameras
(see `report/`). Use the production path unless you are explicitly researching a
different camera topology.

Run the unit tests (no GPU/DeepStream needed):

```bash
python -m pytest tests/ -v          # or: python tests/test_fusion.py
```

---

## Layout

```text
configs/      pipeline presets, nvinfer detector + nvtracker configs, sources
models/       YOLO + Swin ReID ONNX (engines built on first run; gitignored)
dataset/      MMPTracking_short (primary) + other datasets
src/          main.py, pipeline/, reid/, eval/, dataset/
scripts/      training + benchmark utilities
Old materials/ learning milestones, daily reports, COMMANDS.md, legacy dataset docs
```

Model ONNX files are tracked via Git LFS; run `git lfs pull` after cloning.

---

## Training

```bash
python scripts/datasets/mmp_to_yolo.py        # convert MMP → YOLO format
python scripts/train/train_yolo_mmp.py     # fine-tune YOLO11n detector
python scripts/train/finetune_reid_mmp.py --train-all-nonretail   # Swin-Tiny ReID
```

Docker training services: `yolo_train`, `reid_train_mmp` (see `docker-compose.yml`).

---

The earlier learning-project history (DeepStream milestones 1–8, MTA / Wildtrack /
mtmc_4cam experiments, daily reports, full command log) is preserved under
[Old materials/](Old%20materials/).
