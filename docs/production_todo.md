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

Latest verified result — canonical = honest SINGLE-PASS full-GT (every frame once, no loop,
no GT trimming; score with `scripts/eval/score_longrun_idf1.py` after `live_buffered --once`):

```text
performance preset (DEFAULT — reid0):
  mean IDF1: 0.8109  (cafe 0.833 lobby 0.895 office 0.861 industry 0.805 retail 0.660)
  ~10.6 FPS/cam, ~9.4 GB

quality preset (reidType:2):
  mean IDF1: 0.8132  (cafe 0.834 lobby 0.895 office 0.877 industry 0.806 retail 0.655)
  ~9.5 FPS/cam, ~12.7 GB
```

The two presets tie on IDF1 (global IDs come from the SGIE embeddings; NvDCF internal ReID only
aids local continuity, which buffered clustering is robust to) — reid0 is the default (more headroom).
The older 600s-looped numbers (0.8344 / 0.8098) were processed-segment (optimistic GT trimming);
untrimmed-looped is ~0.758 (over-penalized). See CHANGE.md 2026-06-21/22.

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
- [x] Add exact-source manual ReID labels and test full-env/no-retail retraining.
- [x] Add YOLO11x crop verification for exact-source ReID training data and
  test retraining on verified crops.
- [x] Verify the IDF1 target honestly (single-pass full-GT): reid0 0.811 / quality 0.813;
  make reid0 the documented default. (2026-06-21/22)
- [x] Build run-summary tool, config validator, run-dir + run_manifest.json (3.3/4.1/4.3).
- [x] RTSP smoke validation (3.1) + fix mediamtx_loop.sh (transcode mpeg4→h264, TCP transport).
- [x] Bounded 25-min soak: FPS/VRAM stable, GID plateau, no leak (3.2 partial; overnight pending).
- [x] Diagnose retail failure mode (3.5): local identity (switches/frags), not cross-cam.

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

Exact-source manual relabel experiment, 2026-06-21:

```text
labels: reid_labels_exact/

official crop cache:
  train: 416,011 crops, 308 scene-local pids
  val:   211,391 crops, 168 scene-local pids

cleaned full-env train:
  relabeled train:    378,969 kept crops, 70 manual identities
  filtered:           37,042 small/edge crops
  sampled by trainer: 164,818 crops
  best val_gap:       0.550 at epoch 3

cleaned no-retail train:
  relabeled train:    259,717 kept crops, 56 manual identities
  excluded retail:    155,601 crops
  filtered:           693 small/edge crops
  sampled by trainer: 129,818 crops
  best val_gap:       0.440 at epoch 15
```

Important correction:

```text
An earlier local run accidentally scoped manual identities by time prefix and
created 72 full-env / 57 no-retail identities. That was superseded. The correct
identity key is env::manual_person, so each environment has exactly 14 manual
identities.
```

Original-val retrieval comparison, 50 crops per scene-camera:

```text
deployed production ONNX:
  cross-camera top1: 0.5317
  cross-camera mAP:  0.4027

full-env exact relabel envmerge e20:
  cross-camera top1: 0.3317
  cross-camera mAP:  0.1848

no-retail exact relabel envmerge e20:
  cross-camera top1: 0.3037
  cross-camera mAP:  0.1815
```

Verdict:

- Do not promote either exact-relabel ONNX.
- Keep production on `models/reid/swin_tiny_mmp_reid_all.onnx`.
- Keep `reid_labels_exact/` as the manual exact-source label set.
- More training from ImageNet Swin-Tiny is unlikely to be the shortest path.
  Recover the trainable checkpoint behind the deployed ONNX or add a
  distillation/fine-tune path from deployed embeddings first.

YOLO11x crop verification experiment, 2026-06-21:

```text
script: scripts/datasets/filter_reid_crops_yolo.py
weights: yolo11x.pt
gate: person class, conf >= 0.15, imgsz 320

full env:
  kept:              323,026
  rejected:          55,943
  identities kept:   70
  retail rejected:   41,007

no retail:
  kept:              244,784
  rejected:          14,933
  identities kept:   56
```

Retraining/eval on original validation crops:

```text
deployed production ONNX:
  top1: 0.5317
  mAP:  0.4027

full-env exact relabel, geometry-clean only:
  top1: 0.3317
  mAP:  0.1848

full-env exact relabel, YOLO11x verified:
  top1: 0.3257
  mAP:  0.1976

no-retail exact relabel, YOLO11x verified:
  top1: 0.2988
  mAP:  0.1774
```

Verdict:

- YOLO11x verification is a useful crop-cleaning/audit tool.
- It confirms retail has the largest crop-quality problem.
- It does not solve the ReID model-quality gap by itself.
- Do not promote the YOLO11x-trained ONNX models.

### 3.1 RTSP Production Validation

- [x] Run a real RTSP smoke with MediaMTX (5x office_0, reid0). 2026-06-22.
- [x] Confirm `src.main` consumes `rtsp://` sources without source-id mismatch. (clean)
- [x] Confirm output chunks continue flushing on RTSP. (10 chunks)
- [x] Confirm FPS and VRAM sane (15 FPS/cam at native rate, VRAM 3.1 GB / 5 cams).
- [x] Record command, duration, FPS, VRAM in `CHANGE.md` (2026-06-22).
- [ ] Scale the RTSP smoke to the full 20 streams (multienv) and confirm >=10 FPS/cam.

Fix applied: `mediamtx_loop.sh` now transcodes non-h264 (the MMP mp4s are mpeg4) to h264 and
forces `-rtsp_transport tcp` — stream-copy + UDP made the publishers 404 / "Broken pipe".

Acceptance:

```text
20 RTSP streams start
pipeline runs without deadlock
avg FPS/cam >= 10 after warmup
VRAM stable
det_emb_chunk_*.npz files keep appearing
```

### 3.2 Long-Duration Stability

- [x] Bounded 25-min file-loop soak (reid0) — all health checks pass (see below). 2026-06-22.
- [ ] Run 2h file-loop soak.
- [ ] Run overnight RTSP/file-loop soak.
- [x] Watch (validated over 25 min via `summarize_long_run.py`):
  - FPS stability    — avg 10.93 (min 10.40, max 11.60), no degradation
  - VRAM/RSS creep   — VRAM avg 9.45 GB stable; RSS creep +117 MB/24min (watch overnight)
  - GID count plateau — active 8 / total 10 (creep +2), no leak
  - output chunk cadence — steady (81 chunks)
  - pipeline log errors  — 0
  - live-buffered clustering latency — 156 ms avg / 313 ms max

Acceptance:

```text
no process crash
no VRAM/RSS creep trend
no unbounded GID growth
avg FPS/cam remains near target
logs are actionable and not spammy
```

### 3.3 Run Summary Tool

Build a cheap post-run summarizer. **DONE 2026-06-22.**

- [x] Add `scripts/eval/summarize_long_run.py`.
- [x] Read `long_stability.csv`, `long_buffered.csv`, `long_pipe.log`.
- [x] Report warmup-trimmed FPS/cam, VRAM, RSS trend, active/total GIDs, clustering latency,
  chunk count + last-chunk time, error/warning counts.

Acceptance:

```bash
python scripts/eval/summarize_long_run.py output/logs output/eval/long_run
```

prints one concise health report.

### 3.4 Persistence

Current production persistence is CSV/NPZ/log files only. That is fine for eval
but thin for a real system.

Recommended first step:

- [x] Add simple append-only SQLite sink (`scripts/eval/persist_run.py`, 2026-06-22) for
  per-detection rows, global assignments, run health metrics, chunk metadata + run provenance.
  Idempotent per run_id (`--replace` to overwrite); built around the current export/log schema,
  NOT the archived analytics/storage prototype.

  `python scripts/eval/persist_run.py --run-dir output/runs/<ts>_<preset> --db output/runs.sqlite`

Future production step:

- [ ] TimescaleDB/Postgres for time-series metadata.
- [ ] pgvector or a separate vector DB only if ReID search becomes a product
  requirement.

### 3.5 Retail Quality Work

Retail is the weak environment in both presets.

- [x] Quantitative per-stage diagnosis via `scripts/eval/diagnose_retail.py` (2026-06-22).
- [x] Check whether retail failure is detector recall, tracker fragmentation, or ReID confusion.
  FINDING: retail recall only mildly low (0.822 vs 0.896); the dominant loss is LOCAL identity —
  local IDF1 0.522 vs 0.835, switches 643 vs 212, frags 1574 vs 1185 — while the cross-camera
  step adds ~0 extra loss (local→global gap ≈ 0 in every env). I.e. weak retail ReID embeddings
  (retrieval top1 0.264) show up as within-camera ID swaps, not a cross-cam-specific failure.
- [ ] Visual audit (missed dets / same-clothes / occlusion) to confirm the switch sources.
- [ ] Only after diagnosing, test one focused lever at a time:
  - detection threshold
  - tracker confidence / shadow age
  - SGIE crop quality gate
  - retail-specific window-chunk setting

Avoid broad retraining until the failure mode is proven.

## 4. Production Hardening

### 4.1 Config Guardrails

**DONE 2026-06-22:** `scripts/setup/validate_config.py` (run as a preflight by run_long_eval.sh).

- [x] Add a config validator command.
- [x] Ensure required production files exist.
- [x] Ensure model paths resolve.
- [x] Ensure SGIE config is present for production presets.
- [x] Ensure tracker `outputReidTensor: 0` when SGIE is used.
- [x] Ensure source count matches expected env map in long eval.

### 4.2 Docker

- [ ] Run `scripts/setup/docker_smoke_test.sh --build`.
- [ ] Run `docker compose run --rm tracker`.
- [ ] Confirm generated TensorRT engines stay local and are not committed.
- [ ] Document GPU/driver/TensorRT version used for verified runs.

### 4.3 Logging

**DONE 2026-06-22:** `run_long_eval.sh USE_RUN_DIR=1` + `scripts/eval/write_run_manifest.py`.

- [x] Standardize run directory naming: `output/runs/YYYYMMDD_HHMMSS_<preset>`.
- [x] Write one `run_manifest.json` (git commit, pipeline config, sources, env map, duration,
  GPU name, model files, key thresholds).
- [x] Store logs/eval output under that run directory.

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

Do not reuse the old `reid_labels/` files for this exact-source path. Those
labels were made against the 10-minute extracted crop-cache ID space. The
official MMPTracking person IDs reset per scene, so exact-source regrouping must
be reviewed/saved again in `reid_labels_exact/`.

Apply labels into a trainable grouped cache:

```bash
./venv/bin/python scripts/datasets/apply_reid_labels_exact.py \
  --labels-dir reid_labels_exact \
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
