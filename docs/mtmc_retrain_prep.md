# MTMC_Tracking_2026 — retrain prep (YOLO + ReID)

Prep notes for retraining the detector and ReID model on **MTMC_Tracking_2026**
(NVIDIA AI-City warehouse multi-camera people-tracking dataset) and running the
pipeline on it for visualization. CPU-only prep — no GPU needed until the actual
`train` step.

## 1. Dataset analysis

| | |
|---|---|
| Splits | `train` (20 warehouses), `val` (3), `test` (5, **GT withheld**) |
| Per warehouse | `videos/Camera_XXXX.mp4` (~19–20 cams), `ground_truth.json`, `calibration.json`, `map.png` |
| Video | **1920×1080, 30 fps, 300 s = 9000 frames** |
| Total | 353 videos |
| Object types | **Person** (target), PalletTruck, Forklift (ignored) |
| Density | **~55 people/frame**, **60 distinct IDs/warehouse**, ~1.4M person boxes/warehouse |
| Box sizes | median height 62 px, but **~29 % are tiny** (<24 px h) — far cameras |

**Ground truth schema** (`ground_truth.json`, ~280 MB/warehouse — stream it, don't load):
```
{ "<frame>": [ { "object type": "Person",
                 "object id": 9634,                      # GLOBAL, cross-camera (per warehouse)
                 "3d location": [...], "3d bounding box ...": [...],
                 "2d bounding box visible": { "Camera_0000": [x1,y1,x2,y2], ... } }, ... ] }
```
Key win vs MMPTracking: **`object id` is a true cross-camera identity** (MMP's `person_id`
was scene-local — see `[[mmp_person_id_scene_local]]`). We only namespace it per warehouse
(`<warehouse>_<objid>`) so IDs don't collide across warehouses.

`calibration.json` is full 3D (per-camera `intrinsicMatrix` 3×3, `extrinsicMatrix` 3×4,
`cameraMatrix` 3×4 projection, `scaleFactor`, `translationToGlobalCoordinates`) + `map.png`
(1920×1080 top-down) — richer than MMP's homography. Used for BEV visualization (§5).

## 2. Convert → YOLO + ReID (one decode pass)

`scripts/datasets/mtmc_prepare.py` streams the GT (ijson, memory-safe) and decodes each
camera video **once**, emitting both datasets:

```bash
# smoke (1 warehouse, 2 cams, 4 frames) — verify before the big run
python scripts/datasets/mtmc_prepare.py --split val --warehouses Warehouse_020 \
    --max-cams 2 --max-frames 4 --stride 30 --out-suffix _smoke

# full prep (1 fps): run per split
python scripts/datasets/mtmc_prepare.py --split train --stride 30
python scripts/datasets/mtmc_prepare.py --split val   --stride 30
```

Outputs:
- `dataset/mtmc_yolo/{images,labels}/{train,val}/...` + `dataset.yaml` (class 0 = person, normalized xywh)
- `dataset/mtmc_reid_cache/{train,val}/<pid>/...jpg` + `{train,val}/manifest.csv`
  (columns `scene,pid,cam_id,frame,rel_path` — **drop-in for the ReID trainer's `CachedReidDataset`**)
  + `pid_map_<split>.csv` (string↔int pid).

Filters (tune via flags): YOLO `--min-h 12 --min-w 6`; ReID `--reid-min-h 64 --reid-min-w 24`
(ReID needs more pixels). `--keep-empty` to also write person-free negatives for the detector.

### Scale / disk / time estimates (stride 30 = 1 fps)
| | images/crops | disk | note |
|---|---|---|---|
| YOLO train | ~114k images | ~30–45 GB | 1080p JPEG q90 |
| YOLO val | ~17k images | ~5–7 GB | |
| ReID train | ~400k crops, ≤1200 IDs | ~4 GB | after `reid-min` filter |

**Time:** decode-bound (~380 train videos × 9000 frames, CPU). Budget several hours; run
in background / overnight. To cut cost: raise `--stride` (60 = 0.5 fps halves it), or limit
cameras with `--max-cams`. Decode currently reads videos sequentially (reliable); a future
ffmpeg `select` fast-path could speed it up. **NVDEC GPU decode is avoided** while the GPU
is busy with other training.

## 3. Retrain YOLO

```bash
python scripts/train/train_yolo_mtmc.py \
    --data dataset/mtmc_yolo/dataset.yaml \
    --epochs 30 --batch 16 --imgsz 640 --device 0 \
    --weights yolo11n.pt --onnx-dest models/yolov11/yolo11n_mtmc.onnx
```
Exports `models/yolov11/yolo11n_mtmc.onnx` (won't clobber the MMP model). Note: warehouse
people are small/dense — consider `--imgsz 960` for the tiny-box population if recall is low.

## 4. Retrain ReID

The trainer reads the crop cache directly (separate `train/`+`val/` manifests):
```bash
python scripts/train/finetune_reid.py \
    --crop-cache-root dataset/mtmc_reid_cache \
    --epochs 40 --pk-p 24 --pk-k 4 \
    --output output/reid_mtmc
```
Exports `output/reid_mtmc/swin_tiny_*_reid.onnx` → drop into the SGIE / nvtracker ReID config.
(Cross-camera IDs make this a much cleaner ReID signal than MMP.)

## 5. Deploy + run pipeline on MTMC (visualization)

1. Point nvinfer at `yolo11n_mtmc.onnx`; point SGIE at the new ReID ONNX.
2. Build a sources file from a warehouse's `videos/Camera_*.mp4`.
3. **Geometry/BEV adapter (TODO):** `src/reid/geometry.py` currently parses MMP calibration.
   MTMC needs a small loader that uses the per-camera `cameraMatrix` (3×4) to project the
   foot point to the ground plane (z=0), plus `scaleFactor` + `translationToGlobalCoordinates`
   + `map.png` for the BEV canvas. Until then, per-camera heatmaps/tracking work without
   calibration; BEV needs this adapter.

## Caveats
- **`test` split has no `ground_truth.json`** — only `train`/`val` are usable for training/eval.
- Warehouse scenes are denser and lower-resolution-per-person than MMP; expect to retune
  detector `imgsz` and the small-box filters.
- Identity count is modest (~60/warehouse); the variety comes from 20 warehouses × many cameras.

## Results (2026-06-27) — detector + ReID retrained

**Dataset fix (important).** The first conversion used `--min-h 30`, which silently dropped
all small / behind-shelf people (0 boxes <24 px) — the detector would never learn the hard
warehouse cases. Re-converted at **`--min-h 8 --min-w 4`** → `dataset/mtmc_yolo` (71.7k train /
10.2k val, all 20+3 warehouses); **8.8 % of boxes are now <24 px** (small/occluded restored).
Note: MTMC GT is `2d bounding box visible` (real visible boxes, NOT MMP-style amodal projection),
so verifier label-cleaning is the WRONG tool here — it would drop ~33 % real-but-hard people
(COCO-domain-gap + occlusion). Trained directly on the size-filtered visible-box set.

**Detector — `yolo11n_mtmc.onnx`** (YOLO11n, imgsz **960**, batch 16, 30 ep, 6.6 h, best ep22):
- val **mAP50 0.694**, mAP50-95 0.501, precision 0.913, recall 0.615.
- Recall by GT box height (val, conf 0.25 / IoU 0.5): large ≥96 px **0.94**, med 48–96 **0.66**,
  small 24–48 **0.25**, tiny <24 px **0.08**. So large/medium people are solid; **small/tiny
  remain the limit** — yolo11n@960 can't resolve sub-48 px warehouse people. Keeping them was
  right (the model now tries + the gap is measured); to actually lift small-box recall: imgsz
  1280, a bigger model (yolo11s/m), or tiled/SAHI inference.

**ReID — `models/reid/swin_tiny_mtmc_reid.onnx`** (Swin-Tiny, PK 24×4, early-stopped, val_gap 0.764):
cross-camera retrieval on MTMC val (57 ids, 16 cams) **top-1 0.929 / mAP 0.884**, vs the deployed
MMP ReID **0.609 / 0.287** on the same data — a large gain, confirming MTMC's true cross-camera
`object id` is a much cleaner ReID signal than MMP's scene-local `person_id`.

**Artifacts:** `models/yolov11/yolo11n_mtmc.onnx`, `models/reid/swin_tiny_mtmc_reid.onnx` (LFS).
The `yolo11n_mtmc_overfit` model (tiny overfit experiment) is left untouched. Next: §5 deploy +
BEV geometry adapter, and (if small-box recall matters) a higher-res / larger detector pass.
