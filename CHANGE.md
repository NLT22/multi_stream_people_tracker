# CHANGE.md

Recovered handoff notes after the SSD/GitHub recovery on 2026-06-20.

Use this file as the short memory for the next agent: what was restored, what was
tested, what was rejected, and what can be undone.

## Production Cleanup on 2026-06-20

The root project was reduced to the real production path:

- YOLO11 MMP detector
- NvDCF tracker
- SGIE Swin-Tiny ReID
- long eval / MediaMTX helpers
- Docker tracker service

Archived to `old_stuff/retired_20260620/`:

- old pipeline and tracker ablations
- NvDeepSORT configs
- old detector/ReID nvinfer configs
- dataset conversion, training, benchmark, and anchor-guided scripts
- old/rejected ONNX models
- stale setup scripts that depended on archived training files

Deleted local generated artifacts:

- `output/`
- Python `__pycache__/`
- TensorRT `.engine` files
- generated `configs/models/*.runtime_*.yml`

Production defaults were updated in:

- `src/config/runtime.py`
- `src/config/args.py`
- `src/pipeline/runner.py`
- `README.md`
- `configs/README.md`
- `models/README.md`
- `docker-compose.yml`
- `scripts/setup/docker_smoke_test.sh`
- `scripts/eval/mediamtx_loop.sh`
- `scripts/eval/mediamtx_multienv.sh`

Undo:

```bash
git revert <cleanup-commit>
```

or selectively restore one archived file with `git mv old_stuff/retired_20260620/<path> <path>`.

## Production Tightening After Cleanup

Additional root cleanup:

- removed accidental import of the old incremental MTMC simulator from
  `src.mtmc`
- inlined the only helper `src.mtmc.live_buffered` needed from the simulator
- archived retired source files:
  - `src/mtmc/incremental_mtmc.py`
  - `src/mtmc/run_incremental.py`
  - `src/mtmc/tracklet.py`
  - `src/eval/offline_anchor.py`
  - `docs/Note.md`
- added `scripts/setup/production_smoke.sh` as a cheap non-GPU production wiring
  check

Archived additional non-production prototypes:

- `src/eval/detect_eval_mmp.py`
- `src/eval/reid_reentry_merge.py`
- `src/analytics/`
- `src/storage/`
- `tests/test_reentry_merge.py`

## Current Best Production Path

Target:

- 20 cameras
- 10 FPS/cam
- mean IDF1 >= 0.8 on the current 640x360 MMPTracking_10minute validation set
- production-style buffered/global IDs, not offline-only scoring

Best known preset:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 180 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Files that define this path:

- `configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml`
- `configs/models/nvinfer_reid_swin_sgie_all.yml`
- `configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml`
- `src/eval/export.py`

Architecture:

- YOLO11 PGIE detector.
- NvDCF tracker keeps `reidType: 2` for tracker stability.
- Tracker uses `outputReidTensor: 0`.
- A secondary nvinfer SGIE runs `swin_tiny_mmp_reid_all.onnx` on person crops.
- SGIE uses stretch preprocessing (`maintain-aspect-ratio: 0`) and ImageNet norm.
- Export path reads the clean SGIE tensor metadata for live buffered clustering.

Why this is the best current path:

- In-tracker DeepStream ReID quality capped around IDF1 ~0.68 on office_0.
- Decoupled SGIE ReID raised office_0 live buffered IDF1 to ~0.875.
- Offline clean-crop ceiling on office_0 was ~0.92, so SGIE is close to ceiling.

## Throughput Fix Restored

Restored in `src/eval/export.py`:

- live `det_emb_chunk_*.npz` writes use `np.savez(...)`, not
  `np.savez_compressed(...)`
- `close()` flushes the final partial live embedding chunk when
  `emb_flush_frames > 0`

Reason:

Compression was happening inside the DeepStream metadata probe path. At 20 cams,
that CPU compression stalls the video pipeline. Uncompressed NPZ keeps the same
array keys and values for live buffered eval, but removes hot-path compression.

Regression test:

```bash
python -m pytest tests/test_export.py -v
```

## Best Known Target Closure Result

Updated SSD verification on 2026-06-20:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Throughput, post-warmup:

```text
elapsed >= 90s:
  avg FPS/cam: 9.99
  min FPS/cam: 9.80
  max FPS/cam: 10.20
  avg VRAM:    ~12.7 GB
```

Important recovered-code fix:

- `run_long_eval.sh` now passes `ENV_MAP` into `src.mtmc.live_buffered`.
- `src.mtmc.live_buffered` clusters each environment group independently.
- `src.mtmc.live_buffered` writes per-detection `_eval_assign.csv` assignments.
- default live-buffered chunks are `retail:4,default:1`.

Why this matters:

The recovered GitHub version clustered all 20 mixed validation cameras as one
world and ended with only about 8 active IDs total. That is wrong for the mixed
benchmark because the 20 cameras are five independent environments. Grouped
buffering restores one MTMC state per environment.

Processed-segment IDF1 from the 600s run, using grouped per-detection
assignments and GT filtered to the frames actually processed:

```text
64pm_cafe_shop_0        0.8846
64pm_lobby_0            0.9130
64pm_office_0           0.8949
64pm_industry_safety_0  0.8597
64pm_retail_0           0.6200
MEAN                    0.8344
```

Caveat:

Do not score the 600s looped run against the entire untrimmed GT. Some scenes
have GT beyond the processed frames, and retail has loop-tail predictions beyond
its GT. Untrimmed scoring undercounts the run:

```text
grouped per-detection, untrimmed GT mean: 0.7760
raw online gallery, untrimmed GT mean:    0.2138
```

This means the target is met for the processed 20-cam 10-minute segment, but
retail remains the weak environment and should be the next quality focus.

Optional performance preset tested on 2026-06-20:

- `configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml`
- `configs/tracker/nvdcf_accuracy_mmp_recall_sgie_reid0.yaml`

This keeps SGIE ReID for exported/global embeddings, but disables NvDCF internal
tracker ReID (`reidType:0`). It matches the anchor-guided idea more closely:
tracking supplies detections/local tracks, while dense ReID drives the global
assignment stage.

600s processed-segment result:

```text
reidType:2 + SGIE quality preset:
  avg FPS/cam: 9.99
  avg VRAM:    ~12.7 GB
  mean IDF1:   0.8344

reidType:0 + SGIE performance preset:
  avg FPS/cam: 10.60
  avg VRAM:    ~9.34 GB
  mean IDF1:   0.8098
```

Per-scene `reidType:0 + SGIE`:

```text
64pm_cafe_shop_0        0.8715
64pm_lobby_0            0.9024
64pm_office_0           0.8776
64pm_industry_safety_0  0.8478
64pm_retail_0           0.5498
MEAN                    0.8098
```

Verdict:

- Keep `pipeline_mmp_nvdcf_online_sgie.yaml` as the quality default for now.
- Use `pipeline_mmp_nvdcf_online_sgie_reid0.yaml` when production needs lower
  VRAM or more FPS headroom.
- Retail remains the quality limiter in both paths.

## Exact MMPTracking Detector Training/Eval on 2026-06-20

Added source-clean YOLO detector tooling that reads the official MMPTracking zip
dataset directly:

- `scripts/datasets/mmp_exact_to_yolo.py`
- `scripts/train/train_yolo_mmp_exact.py`
- `scripts/eval/eval_yolo_mmp_exact.py`

This path reads:

```text
dataset/MMPTracking/MMPTracking_training/train/{images,labels}/...
dataset/MMPTracking/MMPTracking_validation/validation/{images,labels}/...
```

It does not read the old extracted video/CSV cache. The generated YOLO folder is
only a training format derived from the official zip files and is ignored by git.

Conversion command used:

```bash
./venv/bin/python scripts/datasets/mmp_exact_to_yolo.py \
  --output-dir dataset/mmp_exact_yolo \
  --sample-rate 10 \
  --clean
```

Output:

```text
scenes:       68 official scene zips
train images: 121,534
val images:   61,949
total images: 183,483
total boxes:  1,254,629
val boxes:    422,950
```

Detector baseline on exact-source val:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/eval/eval_yolo_mmp_exact.py \
  --data dataset/mmp_exact_yolo/dataset.yaml \
  --weights models/yolov11/yolo11n_mmp.onnx \
  --imgsz 640 --batch 32 --device 0 \
  --project output/eval_exact \
  --name yolo11n_mmp_exact_sr10_baseline
```

Result:

```text
images:    61,949
instances: 422,950
precision: 0.9653
recall:    0.8929
mAP50:     0.9571
mAP50-95:  0.7565
```

One-epoch smoke training from generic `yolo11n.pt`:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/train/train_yolo_mmp_exact.py \
  --data dataset/mmp_exact_yolo/dataset.yaml \
  --weights yolo11n.pt \
  --epochs 1 --batch 32 --imgsz 640 --device 0 --workers 4 \
  --project output/train_exact \
  --name yolo11n_mmp_exact_sr10_e1 \
  --patience 0 --no-export
```

Result after the built-in epoch-end validation:

```text
images:    61,949
instances: 422,950
precision: 0.952
recall:    0.830
mAP50:     0.927
mAP50-95:  0.617
```

Verdict:

- Do not promote `output/train_exact/yolo11n_mmp_exact_sr10_e1/weights/best.pt`.
- Current production `models/yolov11/yolo11n_mmp.onnx` remains much better on
  exact-source val.
- The training script now avoids an extra duplicate full validation unless
  `--final-val` is explicitly passed.
- Exact-source detector mAP is done; exact-source end-to-end IDF1 still needs a
  DeepStream image-sequence/RTSP conversion path or exact-video generation from
  the official zip frames.

## Exact MMPTracking ReID Eval on 2026-06-21

Added exact-source ReID crop/eval tooling:

- `scripts/datasets/mmp_exact_to_reid.py`
- `scripts/eval/eval_reid_mmp_exact.py`

This path reads the official MMPTracking zip image/label files directly and
builds identity crops from GT boxes. It does not use the old extracted
MMPTracking_10minute videos/CSVs.

Bounded exact-val crop build:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
  --output-dir dataset/mmp_exact_reid_eval \
  --splits val \
  --sample-rate 100 \
  --max-crops-per-scene 1000 \
  --clean
```

Output:

```text
val crops: 23,571
val pids:  168
```

Balanced deployed-ONNX ReID eval:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/eval/eval_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_eval \
  --split val \
  --weights models/reid/swin_tiny_mmp_reid_all.onnx \
  --batch 64 \
  --max-crops-per-scene 200
```

Metric definition:

- cross-camera top1: nearest crop from a different camera has same scene-local
  pid
- cross-camera mAP: AP over crops from different cameras
- this is embedding retrieval quality only, not end-to-end MTMC IDF1

Result on 4,800 balanced exact-val crops:

```text
cross_camera_top1: 0.5504
cross_camera_mAP:  0.4263

env mean top1:
  cafe_shop:       0.7675
  industry_safety: 0.5050
  lobby:           0.8038
  office:          0.7617
  retail:          0.2644
```

Interpretation:

- The deployed Swin ReID model is acceptable on lobby/office/cafe but weak on
  industry and very weak on retail.
- This aligns with the end-to-end IDF1 bottleneck: retail is not just a tracker
  issue; ReID appearance generalization is poor there.
- Current repo only has the deployed ONNX, not the original trainable Swin
  checkpoint, so do not claim exact-source ReID training is ready yet.
- Full exact-val ReID eval is slow in this venv because ONNX Runtime falls back
  to CPU: CUDA provider cannot load `libcudnn.so.9`. Production DeepStream still
  uses TensorRT, so this is an eval-environment limitation.

Single-pass product validation before the SSD recovery:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_short_shakeout.sh configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Active 20-cam FPS window, elapsed 90..542s:

```text
avg FPS/cam: 9.92
min FPS/cam: 9.60
max FPS/cam: 10.40
```

All post-warmup samples:

```text
avg FPS/cam: 10.12
min FPS/cam: 8.40
max FPS/cam: 14.20
```

Buffered IDF1 from the same single-pass artifacts:

```text
64pm_cafe_shop_0        0.8171
64pm_lobby_0            0.8858
64pm_office_0           0.8702
64pm_industry_safety_0  0.8045
64pm_retail_0           0.6633
MEAN                    0.8082
```

Note:

Some late FPS samples dip then jump because validation videos do not all have the
same duration; once some streams hit EOS, the active stream count changes.

## SSD Recovery Smoke Test

After recloning/recovering on the 1.5TB SSD path:

```bash
bash scripts/eval/run_long_eval.sh 180 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Result:

- completed successfully
- 20 prediction CSVs written to `output/eval/long_run`
- 7 `det_emb_chunk_*.npz` files written
- SGIE model loaded successfully
- short 3-minute FPS samples averaged about 9.4 FPS/cam before restoring the
  uncompressed live chunk fix

This smoke test proves the recovered SGIE pipeline runs, but it is not a full
target closure run.

## Rejected Experiments

Do not repeat these unless there is a specific new hypothesis.

```text
NvDCF sub_batches: "5:5:5:5" on SGIE preset
  result: ~6.27 FPS/cam
  verdict: rejected

SGIE interval=1
  result: ~8.85 FPS/cam
  verdict: rejected; interval semantics did not buy useful speed here

Tracker reidType:0 / perf tracker path
  result: ~8.57 FPS/cam in tested preset
  verdict: rejected for current target

No-tiler/no-OSD experiment
  result: ~7.96 FPS/cam
  verdict: rejected; not the bottleneck

Export-only/no-gallery experiment
  result: ~8.20 FPS/cam
  verdict: rejected; gallery was not the main limiter

SGIE batch-size 128
  result: ~9.80 FPS/cam and higher OOM risk
  verdict: rejected

LIVE_BUFFERED_WINDOW=400
  result: ~9.73 FPS/cam
  verdict: rejected for current target

Async chunk writer
  result: ~9.83 FPS/cam
  verdict: rejected; uncompressed chunks were simpler and faster

Detector interval=1
  result: ~14.7 FPS/cam but mean IDF1 ~0.6919
  verdict: rejected; speed improved but quality failed
```

## Sidecar Prototype Finding

Prototype idea:

- keep DeepStream fast by capturing detections/crops without SGIE in the graph
- run ReID outside DeepStream as a sidecar worker
- feed sidecar embeddings into the same live buffered evaluator

Observed before SSD recovery:

```text
DeepStream capture path: ~17.8 FPS/cam
Office sidecar IDF1:     0.8738
```

Interpretation:

- Very promising future architecture.
- It proves SGIE-quality ReID can be moved outside the DeepStream graph.
- Current Python sidecar embedding every detection is too slow for full 20-cam
  production.
- To become production, it needs a bounded crop queue, sampling policy, and a
  TensorRT/ONNX GPU worker instead of per-detection Python overhead.

Recovered sidecar files may be missing from the GitHub-restored tree:

- `configs/pipelines/pipeline_mmp_nvdcf_sidecar_capture.yaml`
- `scripts/eval/build_reid_chunks_sidecar.py`

## Files Restored To GitHub During Recovery

These were recreated after the broken SSD copy:

- `configs/models/nvinfer_reid_swin_sgie_all.yml`
- `configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml`
- `configs/tracker/nvdcf_accuracy_mmp_recall_sgie.yaml`
- `scripts/eval/run_long_eval.sh`

Also restored/cleaned:

- root GUI/cache junk was removed after `chown`
- `.gitignore` was updated to ignore local cache/profile/log folders

## What To Verify Next

Run tests:

```bash
python -m pytest tests/test_export.py -v
```

Run a full target check after the uncompressed chunk restore:

```bash
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
```

Then evaluate buffered IDF1 from `output/eval/long_run` artifacts.

## Undo Notes

To undo this handoff file only:

```bash
git rm CHANGE.md
```

To undo the throughput fix:

```bash
git checkout -- src/eval/export.py tests/test_export.py
```

To revert the SGIE production path, remove the SGIE config files listed above
and switch `PIPECFG` back to the non-SGIE online pipeline.

## Exact MMPTracking ReID Training Baseline on 2026-06-21

Added:

- `scripts/train/finetune_reid_mmp_exact.py`

Purpose:

- train Swin-Tiny ReID from the official MMPTracking zip-derived crop cache
- avoid the older extracted `MMPTracking_10minute` cache path
- export an ONNX candidate with the same normalized 256-d feature interface used
  by the SGIE ReID path

Crop cache command:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
  --output-dir dataset/mmp_exact_reid_trainrun \
  --splits train val \
  --sample-rate 100 \
  --max-crops-per-scene 1000 \
  --clean
```

Crop cache result:

```text
train: 43,728 crops, 308 identities, 44 scene zips
val:   23,571 crops, 168 identities, 24 scene zips
```

Training command:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/train/finetune_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_trainrun \
  --output output/reid_mmp_exact_trainrun_e10 \
  --epochs 10 \
  --pk-p 16 --pk-k 4 \
  --accum-steps 2 \
  --batches-per-epoch 120 \
  --workers 4 \
  --early-stop 0
```

Training output:

```text
output/reid_mmp_exact_trainrun_e10/best.pth
output/reid_mmp_exact_trainrun_e10/last.pth
output/reid_mmp_exact_trainrun_e10/swin_tiny_mmp_exact_reid_weights.pth
output/reid_mmp_exact_trainrun_e10/swin_tiny_mmp_exact_reid.onnx
```

Balanced exact-source val eval:

```bash
PYTHONUNBUFFERED=1 ./venv/bin/python scripts/eval/eval_reid_mmp_exact.py \
  --crop-root dataset/mmp_exact_reid_trainrun \
  --split val \
  --weights output/reid_mmp_exact_trainrun_e10/swin_tiny_mmp_exact_reid.onnx \
  --batch 64 \
  --max-crops-per-scene 200
```

Result:

```text
new exact-trained e10:
  cross-camera top1: 0.3675
  cross-camera mAP:  0.2064

deployed production ONNX baseline:
  cross-camera top1: 0.5504
  cross-camera mAP:  0.4263
```

Verdict:

- rejected for production promotion
- keep `models/reid/swin_tiny_mmp_reid_all.onnx` as the SGIE production ReID
  model
- the exact trainer is useful infrastructure, but this initial run starts from
  ImageNet Swin-Tiny because no original trainable ReID checkpoint is present in
  the repo
- next serious ReID attempt should recover the original ReID `.pth` checkpoint
  or run longer training with a stricter exact-val gate before DeepStream IDF1
  validation

Known eval environment issue:

- ONNX Runtime falls back to CPU because `libcudnn.so.9` is missing from the
  current venv runtime path
- this only slows the Python retrieval eval; it does not change the DeepStream
  TensorRT production path
