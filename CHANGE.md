# CHANGE.md

Recovered handoff notes after the SSD/GitHub recovery on 2026-06-20.

Use this file as the short memory for the next agent: what was restored, what was
tested, what was rejected, and what can be undone.

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
