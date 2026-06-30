# Multi-Stream People Tracker

A production-focused **DeepStream 9.0 / pyservicemaker** pipeline for multi-camera
people tracking and cross-camera identity assignment. One person walking through a
building is given a single stable **Global ID** across every camera that sees them,
in real time, on a single consumer GPU.

The project covers two distinct multi-camera-multi-target (MTMC) settings, kept
**completely separate** because they need opposite linking strategies:

| Path | Dataset | Cameras | Cross-camera identity strategy |
|------|---------|---------|--------------------------------|
| **MMP** (production) | MMPTracking | **overlapping** field-of-view | **appearance-first** — ReID embeddings matched across cameras |
| **MTMC** (warehouse) | NVIDIA AI-City `MTMC_Tracking_2026` | **disjoint** field-of-view | **geometry-first** — back-projected ground-plane foot position |

> The MTMC warehouse crops are not visually separable (people are 8–16 m apart under
> shelf occlusion), so identity there is driven by metric calibration, not appearance.
> The MMP path is the opposite: cameras overlap, so appearance ReID wins. **Do not mix
> their configs or scorers.**

---

## Features

- **DeepStream pipeline** (`src/pipeline/`): `nvurisrcbin × N → nvstreammux → nvinfer
  (YOLO11) → nvtracker (NvDCF) → SGIE (Swin-Tiny ReID) → probes → nvosdbin → sink`.
- **Two-layer identity**: per-camera NvDCF local track IDs + a cross-camera gallery /
  buffered-MTMC layer that assigns stable Global IDs.
- **Geometry-first MTMC linker** (`scripts/eval/mtmc_global_linker.py`): constrained
  correlation clustering on ground-plane positions (warehouse W022 Global IDF1 **0.856**,
  92 % of the 0.932 oracle ceiling).
- **Evaluation engine** (`src/eval/mmp_metrics/`): MOTA / IDF1 / Global IDF1.
- **Web console** (`webui/`): React/Vite dashboard — live wall (HLS of the real pipeline
  OSD), heatmaps, ROI/zone editor with autosave, and a natural-language "Ask" view.
- **RAG Q&A layer** (`src/rag/`): FastAPI + Anthropic tool-use agent over tracking metadata.
- **Live RTSP demo** (`webui/scripts/start-live.sh`): loops local videos as RTSP or ingests
  real cameras → DeepStream → HLS → browser.

---

## Folder structure

```text
src/         production pipeline, config, ReID/gallery, MTMC linkers, metrics, RAG
configs/     pipeline / model / tracker / source / analytics-zone configs
models/      production ONNX models (Git LFS) + YOLO parser library (.so)
scripts/     eval, scoring, linker, and Docker smoke helpers
webui/       React/Vite web console (live wall, heatmaps, ROI editor, Ask)
tests/       lightweight unit/regression tests (geometry, ReID, gallery, eval, RAG)
docs/        production notes + reference papers (PDF)
report/      dated dev logs (*.md) + the formal LaTeX thesis (report/latex/)
CLAUDE.md    detailed architecture / config-preset / regression-anchor reference
```

Generated outputs (`output/`), datasets (`dataset/`), TensorRT engines (`*.engine`),
Python caches, and the archived experiment tree (`old_stuff/`) are **gitignored** — they
are not part of the submission.

---

## Setup (from a fresh clone)

Requires **Ubuntu 24.04**, an NVIDIA driver/CUDA compatible with **DeepStream 9.0**,
TensorRT, **Python 3.12**, and a CUDA GPU (developed on RTX 5060 Ti 16 GB).

```bash
git lfs pull                 # fetch the ONNX models (tracked via Git LFS)
./setup_venv.sh              # creates venv + installs deps + the DeepStream pyservicemaker wheel
source venv/bin/activate
```

> **`pyservicemaker` is NOT on PyPI.** It ships only in the DeepStream SDK and is
> installed from the SDK wheel by `setup_venv.sh`. `requirements.txt` documents this and
> lists the pip-installable dependencies; `requirements-runtime.txt` is the leaner Docker
> runtime set.

Web console (optional, no GPU needed to browse it):

```bash
cd webui && npm install && npm run dev      # dev server, then open the printed URL
```

---

## Run

**MMP production default** (recommended `reid0` preset — same IDF1 as the quality preset,
faster and leaner):

```bash
python -m src.main \
  --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
  --sources configs/sources/val_20cam_mixed.txt \
  --no-display --no-sync \
  --export-predictions output/eval/manual_run \
  --live-buffered-window 200
```

Long production-style eval (per-environment grouping):

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

**MTMC warehouse** (separate path — source-space coords, geometry linker):

```bash
python -m src.main \
  --config configs/pipelines/pipeline_mtmc_nvdcf_online_sgie_reid0.yaml \
  --sources <warehouse_sources.txt> --no-tiler \
  --export-predictions output/eval/mtmc_run
python scripts/eval/mtmc_global_linker.py  ...   # cross-camera linking
python scripts/eval/score_mtmc_idf1.py --no-rescale ...
```

---

## Evaluation & demo

- **Canonical MMP metric** — honest single-pass full-GT (every frame once, no looping):
  run `live_buffered --once`, then score with `scripts/eval/score_full_mmp_val.py`.
- **Latest verified MMP result** (24 scenes, buffered ID, `reid0`): **mean IDF1 0.798**
  (0.866 excluding the hard retail environment). Retail (0.661) is the limiter — root cause
  was detector phantom boxes (fixed by retraining on cleaned labels). See the
  *Regression Anchors* table in [CLAUDE.md](CLAUDE.md) and [CHANGE.md](CHANGE.md) for the
  full breakdown and VRAM/FPS figures.
- **MTMC scoring**: `scripts/eval/score_mtmc_idf1.py --no-rescale`.
- **Tests**: `source venv/bin/activate && python -m pytest tests/ -q`.
- **Visual demos**: heatmaps via `scripts/eval/venv_visualize.py`; the web console
  (`webui/`) shows the live wall, occupancy heatmaps, and the analytics-zone editor.

The full architecture, config-preset table, and tuning notes live in
**[CLAUDE.md](CLAUDE.md)**; the formal write-up is **[report/latex/main.pdf](report/latex/main.pdf)**.

---

## Notes for the reviewer

- **GPU + DeepStream 9.0 are required to run the pipeline.** Without them you can still
  review all source, configs, the LaTeX report (`report/latex/main.pdf`), the dated dev
  logs (`report/*.md`), and browse the web console (`cd webui && npm install && npm run dev`).
- **MMP vs MTMC are deliberately separate** — appearance-first vs geometry-first. Use each
  path's own pipeline config and scorer; do not point MMP eval at MTMC configs or vice-versa.
- **Datasets and large generated outputs are not included** (gitignored). MMPTracking /
  `MTMC_Tracking_2026` must be obtained from their sources and placed under `dataset/`.
- **Models** are fetched via `git lfs pull`. **TensorRT `.engine` files auto-build** on the
  first run for your GPU (1–3 min for YOLO11n) and are intentionally not committed.
- **Producing a clean package**: the submission is the git tree. Use
  `git archive --format=zip -o submission.zip HEAD` or a fresh `git clone` — both exclude
  the gitignored working-directory clutter (datasets, caches, local profiles).
