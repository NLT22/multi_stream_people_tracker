# Production Readiness TODO

This is the live production roadmap after the 2026-06-20 cleanup. The project is
now intentionally narrow: YOLO11 detector, NvDCF tracker, SGIE Swin ReID, and
live-buffered MTMC evaluation.

Archived research/training/prototype files live under `old_stuff/retired_20260620/`.
Do not restore them into the root path unless they become part of the production
system again.

## 0. Current Production System

Target:

- 20 cameras
- 10 FPS/cam
- mean IDF1 >= 0.8 on the current 640x360 mixed validation set
- production-style buffered/global IDs, not offline-only scoring

Active architecture:

```text
video files or RTSP
  -> DeepStream / pyservicemaker
  -> YOLO11 PGIE detector
  -> NvDCF tracker
  -> SGIE Swin-Tiny ReID on person crops
  -> PredictionExporter writes cam CSV + det_emb_chunk_*.npz
  -> src.mtmc.live_buffered groups cameras by environment
  -> IDF1/stability logs
```

Production quality preset:

```text
configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml
configs/models/nvinfer_yolov11_mmp.yml
configs/models/nvinfer_reid_swin_sgie_all.yml
configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml
```

Performance preset:

```text
configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
configs/tracker/nvdcf_accuracy_mmp_recall_sgie_reid0.yaml
```

Latest verified result, 20-cam mixed run, 600 seconds:

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

Known weakness:

- Retail remains the quality limiter.
- Real production resolution is expected to be 1920x1080, but this repo must
  keep passing the 640x360 benchmark first.

## 1. Production Commands

Cheap non-GPU wiring check:

```bash
scripts/setup/production_smoke.sh
```

Main 20-cam quality eval:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Lower-VRAM performance eval:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

RTSP loop simulation:

```bash
scripts/eval/mediamtx_loop.sh start dataset/MMPTracking_10minute/val/64pm_office_0
scripts/eval/mediamtx_loop.sh stop
```

Multi-environment RTSP cycling:

```bash
scripts/eval/mediamtx_multienv.sh start dataset/MMPTracking_10minute/val
scripts/eval/mediamtx_multienv.sh stop
```

## 2. Done

- [x] Clean root project to production path.
- [x] Archive old pipeline/tracker/model configs.
- [x] Archive training, dataset conversion, benchmark, anchor-guided, analytics,
  storage, and re-entry prototypes.
- [x] Keep old material reversible under `old_stuff/retired_20260620/`.
- [x] Set production default config to
  `configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml`.
- [x] Add `scripts/setup/production_smoke.sh`.
- [x] Fix live embedding chunk export to use uncompressed NPZ and flush final
  chunks.
- [x] Group the mixed 20-cam benchmark by environment in `src.mtmc.live_buffered`.
- [x] Validate target on the current 640x360 mixed benchmark.
- [x] Simplify Docker Compose to the production tracker service.

## 3. Next Priority

### 3.1 RTSP Production Validation

- [ ] Run a real RTSP smoke with MediaMTX and the SGIE quality preset.
- [ ] Confirm `src.main` consumes `rtsp://` sources without source-id mismatch.
- [ ] Confirm output chunks continue flushing on RTSP.
- [ ] Confirm FPS and VRAM match file-loop behavior closely.
- [ ] Record command, duration, FPS, VRAM, and observed GID behavior in
  `CHANGE.md`.

Acceptance:

```text
20 RTSP streams start
pipeline runs without deadlock
avg FPS/cam >= 10 after warmup
VRAM stable
det_emb_chunk_*.npz files keep appearing
```

### 3.2 Long-Duration Stability

- [ ] Run 2h file-loop soak.
- [ ] Run overnight RTSP/file-loop soak.
- [ ] Watch:
  - FPS stability
  - VRAM/RSS creep
  - GID count plateau
  - output chunk cadence
  - pipeline log errors
  - live-buffered clustering latency

Acceptance:

```text
no process crash
no VRAM/RSS creep trend
no unbounded GID growth
avg FPS/cam remains near target
logs are actionable and not spammy
```

### 3.3 Run Summary Tool

Build a cheap post-run summarizer.

- [ ] Add `scripts/eval/summarize_long_run.py`.
- [ ] Read:
  - `output/logs/long_stability.csv`
  - `output/logs/long_buffered.csv`
  - `output/logs/long_pipe.log`
- [ ] Report:
  - warmup-trimmed avg/min/max FPS per cam
  - avg/max VRAM
  - RSS trend
  - latest active/total GIDs
  - chunk count and last chunk time
  - error/warning counts

Acceptance:

```bash
python scripts/eval/summarize_long_run.py output/logs output/eval/long_run
```

prints one concise health report.

### 3.4 Persistence

Current production persistence is CSV/NPZ/log files only. That is fine for eval
but thin for a real system.

Recommended first step:

- [ ] Add simple append-only SQLite or hourly Parquet sink for:
  - per-detection rows
  - global assignments
  - run health metrics
  - chunk metadata

Do not reintroduce the archived analytics/storage prototype directly. Rebuild a
small production sink around the current export/log schema.

Future production step:

- [ ] TimescaleDB/Postgres for time-series metadata.
- [ ] pgvector or a separate vector DB only if ReID search becomes a product
  requirement.

### 3.5 Retail Quality Work

Retail is the weak environment in both presets.

- [ ] Audit retail predictions visually:
  - missed detections
  - ID switches
  - same-clothes confusion
  - occlusion/edge crops
- [ ] Compare quality preset vs performance preset per camera.
- [ ] Check whether retail failure is detector recall, tracker fragmentation, or
  ReID confusion.
- [ ] Only after diagnosing, test one focused lever at a time:
  - detection threshold
  - tracker confidence / shadow age
  - SGIE crop quality gate
  - retail-specific window-chunk setting

Avoid broad retraining until the failure mode is proven.

## 4. Production Hardening

### 4.1 Config Guardrails

- [ ] Add a config validator command.
- [ ] Ensure required production files exist.
- [ ] Ensure model paths resolve.
- [ ] Ensure SGIE config is present for production presets.
- [ ] Ensure tracker `outputReidTensor: 0` when SGIE is used.
- [ ] Ensure source count matches expected env map in long eval.

### 4.2 Docker

- [ ] Run `scripts/setup/docker_smoke_test.sh --build`.
- [ ] Run `docker compose run --rm tracker`.
- [ ] Confirm generated TensorRT engines stay local and are not committed.
- [ ] Document GPU/driver/TensorRT version used for verified runs.

### 4.3 Logging

- [ ] Standardize run directory naming:
  `output/runs/YYYYMMDD_HHMMSS_<preset>`.
- [ ] Write one `run_manifest.json` containing:
  - git commit
  - pipeline config
  - source list
  - env map
  - duration
  - GPU name
  - model files
  - key thresholds
- [ ] Store logs/eval output under that run directory.

## 5. Future Scale-Out

Do not build this until the single-host system is stable.

Possible future architecture:

```text
DeepStream perception
  -> message bus
  -> MTMC service
  -> Timescale/Postgres + vector store
  -> dashboard/API
```

Candidate bus:

- Redis Streams for simple single-host service split.
- Kafka only if multi-host or high-volume replay is required.

Candidate storage:

- SQLite/Parquet for eval.
- TimescaleDB/Postgres for production metadata.
- pgvector/Milvus only if embedding search is needed.

## 6. Analytics And UI

Analytics/UI are product features, not core tracker readiness.

Build only after:

- RTSP validation passes.
- overnight stability passes.
- run summaries are reliable.
- retail quality is understood.

Future features:

- named zone editor
- per-zone occupancy
- entry/exit counts
- route summaries
- Grafana or Streamlit dashboard

Keep zones coarse; fine-grained ground-plane analytics will be noisy.

## 7. Open Questions

- What is the tolerated Global ID latency: near-live, 10 seconds, 30 seconds, or
  longer?
- Is single-host 20cam the final deployment, or will production need multi-host?
- Will real deployment input be all RTSP at 1920x1080?
- Is retail IDF1 a hard product requirement, or is mean IDF1 the acceptance gate?
- Should production default be quality preset or performance preset after RTSP
  validation?
