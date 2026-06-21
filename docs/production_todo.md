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
- [x] Add exact MMPTracking zip-source detector conversion, training, and eval
  helpers.
- [x] Evaluate the current production YOLO11 detector on exact-source validation
  frames.
- [x] Add exact MMPTracking zip-source ReID crop conversion and deployed-ONNX
  retrieval eval.
- [x] Add exact MMPTracking zip-source ReID training helper and run an initial
  controlled training baseline.

## 3. Next Priority

### 3.0 Exact MMPTracking Dataset Path

Detector training/eval now reads the official MMPTracking zip files directly:

```bash
./venv/bin/python scripts/datasets/mmp_exact_to_yolo.py \
  --output-dir dataset/mmp_exact_yolo \
  --sample-rate 10 \
  --clean

./venv/bin/python scripts/eval/eval_yolo_mmp_exact.py \
  --data dataset/mmp_exact_yolo/dataset.yaml \
  --weights models/yolov11/yolo11n_mmp.onnx \
  --imgsz 640 --batch 32 --device 0 \
  --project output/eval_exact \
  --name yolo11n_mmp_exact_sr10_baseline
```

Current exact-source detector baseline:

```text
images:    61,949
instances: 422,950
precision: 0.9653
recall:    0.8929
mAP50:     0.9571
mAP50-95:  0.7565
```

One-epoch smoke training from generic `yolo11n.pt` was worse:

```text
precision: 0.952
recall:    0.830
mAP50:     0.927
mAP50-95:  0.617
```

Do not promote the one-epoch checkpoint. For detector improvement, either recover
the original trainable `.pt` checkpoint behind `models/yolov11/yolo11n_mmp.onnx`
or run a longer controlled fine-tune from `yolo11n.pt` and only export if it
beats the exact-source baseline.

Remaining exact-dataset gap:

- [ ] Add exact-source end-to-end IDF1 evaluation by feeding official zip frames
  to DeepStream, either as generated videos/RTSP streams or an image-sequence
  source path.
- [ ] Keep the existing 10-minute video benchmark passing while exact-source
  end-to-end eval is added.

ReID crop/eval also reads the official zip files directly:

```bash
./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
  --output-dir dataset/mmp_exact_reid_eval \
  --splits val \
  --sample-rate 100 \
  --max-crops-per-scene 1000 \
  --clean

./venv/bin/python scripts/eval/eval_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_eval \
  --split val \
  --weights models/reid/swin_tiny_mmp_reid_all.onnx \
  --batch 64 \
  --max-crops-per-scene 200
```

Current deployed ReID ONNX on balanced exact-source val crops:

```text
cross-camera top1: 0.5504
cross-camera mAP:  0.4263

env mean top1:
  cafe_shop:       0.7675
  industry_safety: 0.5050
  lobby:           0.8038
  office:          0.7617
  retail:          0.2644
```

This confirms retail is an embedding-quality/generalization problem, not only a
tracking or clustering problem. Exact-source ReID training is not productionized
yet because the repo currently has the deployed ONNX, not the original trainable
Swin checkpoint.

Initial exact-source ReID training baseline:

```bash
./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
  --output-dir dataset/mmp_exact_reid_trainrun \
  --splits train val \
  --sample-rate 100 \
  --max-crops-per-scene 1000 \
  --clean

./venv/bin/python scripts/train/finetune_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_trainrun \
  --output output/reid_mmp_exact_trainrun_e10 \
  --epochs 10 \
  --pk-p 16 --pk-k 4 \
  --accum-steps 2 \
  --batches-per-epoch 120 \
  --workers 4 \
  --early-stop 0
```

Crop cache:

```text
train: 43,728 crops, 308 identities, 44 scene zips
val:   23,571 crops, 168 identities, 24 scene zips
```

The 10-epoch model exported successfully, but did not beat the deployed ReID
model on balanced exact-source val crops:

```text
output/reid_mmp_exact_trainrun_e10/swin_tiny_mmp_exact_reid.onnx
  cross-camera top1: 0.3675
  cross-camera mAP:  0.2064

models/reid/swin_tiny_mmp_reid_all.onnx
  cross-camera top1: 0.5504
  cross-camera mAP:  0.4263
```

Verdict:

- Do not promote the new 10-epoch exact-trained ONNX.
- Keep production on `models/reid/swin_tiny_mmp_reid_all.onnx`.
- For real ReID improvement, recover the original trainable Swin/ReID
  checkpoint or run a longer controlled fine-tune with a stronger validation
  gate before any DeepStream promotion.
- ONNX Runtime eval currently falls back to CPU because this venv cannot load
  `libcudnn.so.9`; that slows eval but does not affect DeepStream/TensorRT
  production inference.

Important correction:

- ReID training must use the manually regrouped identity labels when training
  from the old 10-minute crop cache.
- The label files in `reid_labels/*.json` map old extracted-cache scene-track
  IDs, for example `63am_cafe_shop_0|0 -> P0`; they do not directly match most
  raw person IDs in the official zip JSON labels.
- The existing regrouped cache is:

```text
dataset/reid_cache_ssd/MMPTracking_10minute_reid_cache_labeled
```

Regrouped training smoke:

```bash
./venv/bin/python scripts/train/finetune_reid_mmp_exact.py \
  --crop-root dataset/reid_cache_ssd/MMPTracking_10minute_reid_cache_labeled \
  --output output/reid_mmp_regrouped_e10 \
  --epochs 10 \
  --pk-p 16 --pk-k 4 \
  --accum-steps 2 \
  --batches-per-epoch 120 \
  --workers 4 \
  --max-crops-per-pid 2000 \
  --early-stop 0
```

Regrouped train cache:

```text
train: 140,000 sampled crops, 70 manually regrouped identities
val:   220,930 sampled crops, 112 scene-local validation identities
```

Balanced old-cache validation, 50 crops per scene-camera:

```text
new regrouped e10:
  cross-camera top1: 0.4188
  cross-camera mAP:  0.2931

deployed production ONNX:
  cross-camera top1: 0.6919
  cross-camera mAP:  0.6159
```

Verdict:

- The regrouping correction is real and required.
- The new regrouped 10-epoch run is still rejected for production promotion.
- The likely blocker is still missing the original trainable production ReID
  checkpoint; this run starts from ImageNet Swin-Tiny.
- Keep production on `models/reid/swin_tiny_mmp_reid_all.onnx`.

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

## 8. ReID Manual Label Workflow

The existing manual labels in `reid_labels/*.json` were created for the old
`MMPTracking_10minute_reid_cache` scene-track IDs. Use the crop-cache workflow
for regrouping; do not apply those labels directly to raw official zip
`person_id` values.

Prepare the local SSD cache path expected by the retired label app:

```bash
ln -sfn reid_cache_ssd/MMPTracking_10minute_reid_cache \
  dataset/MMPTracking_10minute_reid_cache
```

Create or refresh the auto proposal:

```bash
./venv/bin/python old_stuff/retired_20260620/scripts/datasets/consolidate_reid_identities.py \
  --cache-root dataset/MMPTracking_10minute_reid_cache \
  --split train \
  --reid-onnx models/reid/swin_tiny_mmp_reid_all.onnx \
  --threshold 0.45 \
  --make-montages
```

Run the manual label UI:

```bash
./venv/bin/python old_stuff/retired_20260620/scripts/datasets/reid_label_app.py
```

Then open `http://localhost:8000`, save labels to `reid_labels/`, and apply
them:

```bash
./venv/bin/python old_stuff/retired_20260620/scripts/datasets/apply_reid_labels.py \
  --labels-dir reid_labels \
  --cache-root dataset/reid_cache_ssd/MMPTracking_10minute_reid_cache \
  --out-dir dataset/reid_cache_ssd/MMPTracking_10minute_reid_cache_labeled
```

ONNXRuntime GPU is the expected host-side runtime for ReID eval/proposal tools.
`requirements.txt` uses `onnxruntime-gpu`, and the ReID eval/proposal scripts
preload venv CUDA/cuDNN libraries before creating sessions.

### 8.1 Original MMPTracking Manual Labels

For the official MMPTracking source tree, use the exact-source label tools. This
does not use the extracted 10-minute videos or the old 10-minute crop cache.

Build exact-source crops from the official image/label zips:

```bash
./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
  --mmp-root dataset/MMPTracking \
  --output-dir dataset/mmp_exact_reid_original \
  --splits train \
  --sample-rate 20 \
  --clean
```

Run the exact-source manual label UI:

```bash
./venv/bin/python scripts/datasets/reid_label_app_exact.py \
  --crop-root dataset/mmp_exact_reid_original \
  --split train \
  --out-dir reid_labels_exact
```

Open `http://localhost:8000`. The cards are keyed by exact-source `pid_key`,
for example `63am/cafe_shop_0/1`, meaning `time/scene/raw_pid`.

Existing 10-minute manual ReID labels can be reused for the original
MMPTracking crop manifest. Convert them through scene-local rank mapping:

```bash
./venv/bin/python scripts/datasets/convert_10min_reid_labels_to_exact.py \
  --labels-dir reid_labels \
  --exact-crop-root dataset/mmp_exact_reid_original \
  --out-dir reid_labels_exact_from_10min \
  --strict
```

Verified on the current exact train manifest:

```text
total=308 matched=308 missing=0
```

Use `reid_labels_exact_from_10min` directly, or open it with the exact label UI
for manual review before applying.

Apply labels into a trainable grouped cache:

```bash
./venv/bin/python scripts/datasets/apply_reid_labels_exact.py \
  --labels-dir reid_labels_exact_from_10min \
  --crop-root dataset/mmp_exact_reid_original \
  --out-dir dataset/mmp_exact_reid_original_labeled \
  --splits train
```

Train on the grouped exact-source cache:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/train/finetune_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_original_labeled \
  --output output/reid_mmp_exact_original_labeled \
  --epochs 80 \
  --pk-p 16 --pk-k 4 \
  --accum-steps 2 \
  --batches-per-epoch 400 \
  --workers 4 \
  --early-stop 12
```
