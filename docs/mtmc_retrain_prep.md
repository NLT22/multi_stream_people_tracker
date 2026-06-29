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
The `yolo11n_mtmc_overfit` model (tiny overfit experiment) is left untouched.

## Results (2026-06-27) — deploy + eval on Warehouse_022

Deployed the retrained models through the DeepStream pipeline, **kept fully separate from the
MMP production presets** (own pipeline/detector/ReID/tracker configs — see below). Ran 4 cams
of val Warehouse_022 (trim 60 s = frames 0–1799) and scored cross-camera Global IDF1 against
`ground_truth.json`.

**MTMC-only configs (do not touch MMP):**
- `configs/pipelines/pipeline_mtmc_nvdcf_online_sgie_reid0.yaml`
- `configs/models/nvinfer_yolov11_mtmc.yml` (YOLO11n @960, topk 100)
- `configs/models/nvinfer_reid_swin_sgie_mtmc.yml` (`swin_tiny_mtmc_reid.onnx`)
- `configs/tracker/nvdcf_accuracy_mtmc_sgie_reid0.yaml` (`maxTargetsPerStream=100`, vs MMP 40)

**Run:** `--no-tiler` (pretiler) export so pred bboxes are in exact source 1920×1080 space.
Without it, the post-tiler export wrote tile-space (~1280×720) coords that don't align with GT.
Score with `score_mtmc_idf1.py --no-rescale` (pred already in GT space; the max-coord rescale
heuristic over-scales because warehouse people rarely reach the frame edges).

**Detection is solid:** frame-level cam0 recall **0.88** at IoU≥0.5 (no frame offset). The deployed
detector finds the people.

**Bottleneck = cross-camera + temporal ID linking** (not detection). Per-camera IDF1 0.44–0.58
with ~2× more pred IDs than GT people → identities fragment over time. The buffered MTMC consumer
(`live_buffered --once --groups w022:0-3 --num-people 22`) consolidates this; longer temporal
memory (`--anchor-window`) is the main lever:

| Config | Global IDF1 |
|--------|-------------|
| online gallery (raw export `global_id`) | 0.287 |
| buffered, anchor-window 15 (MMP default) | 0.340 |
| buffered, anchor-window 60 | 0.392 |
| **buffered, anchor-window 125 (best)** | **0.406** |
| buffered, anchor-window 200 | 0.398 |

MTMC's disjoint cameras with long re-appearance gaps need much longer anchor memory than MMP's
overlapping-FOV default (15). But appearance linking **plateaus at ~0.41** for a concrete reason:
a post-hoc gid-merge (`scripts/eval/mtmc_merge_gids.py`) only ever *lowered* IDF1 — every merge
threshold collapsed distinct people, because deployed warehouse crops (small, occluded) yield
embeddings whose per-gid centroids all sit within ~0.2 cosine distance. The deployed ReID can't
simultaneously keep different warehouse people apart and re-link one person's fragments.

### Position-first linker (geometry hand-off) — the real lever

This is exactly the disjoint-AICity case where **position beats appearance**. The warehouses ship
metric `calibration.json` (per-camera intrinsic K + extrinsic [R|t]). New adapter
`src/mtmc/mtmc_calib.py` back-projects each detection's foot point to the ground plane (z=0) →
world (x, y); validated on W022 GT `3d location` to **~0.15 world units**. People are **8–16 world
units apart** while per-frame motion is **~0.05** — position is ~50× more discriminative than the
deployed appearance embeddings.

`scripts/eval/mtmc_position_linker.py` ignores tracker/ReID IDs entirely: per frame it clusters
detections in world space (two cameras seeing one person → one instance → cross-camera link for
free), then tracks instances across frames by nearest active global id (Hungarian, gated), with a
`max-age` so a person re-appearing keeps their id.

| Linker | Global IDF1 | pred IDs (GT=17) |
|--------|-------------|------------------|
| appearance, online gallery | 0.287 | 65 |
| appearance, buffered (best, aw=125) | 0.406 | 21 |
| position-first, per-frame (cr=1.5, gate=2.5, no extrapolation) | 0.739 | 26 |
| **position, tracklet-level (`mtmc_tracklet_linker.py`, spatial 1.5 + velocity-gated temporal)** | **0.747** | 29 |
| *oracle — perfect IDs on the same detections* | *0.926* | *17* |

`scripts/eval/mtmc_tracklet_linker.py` works on whole (cam, local_track_id) tracklets (the NvDCF
id is stable within a camera). Two edge types + union-find: **spatial** (cross-camera tracklets
that co-occur in time and whose world positions coincide < 1.5u — reliable) and **temporal**
(a tracklet whose velocity-extrapolated exit lands on another's start — a hand-off or id-switch;
velocity gating is what stops a *different* person who later walks through the same spot from
being merged). The per-camera clustering constraint (one detection per camera per instance) was
the key swap fix: warehouse people cross within **0.2–1.3 world units**, so radius-only clustering
merged crossing people.

### The 0.9 ceiling is now the detector, not the linker

The **oracle** (assign every detection its IoU-matched GT id) scores only **0.926**, because the
deployed detector never produces **~14 % of GT boxes** (`recall ceiling 0.87`; YOLO11n@960 tiny-box
recall is 0.08). Verified this is a *detector* limit, not a threshold/tracker one:
- lowering `pre-cluster-threshold` 0.25→0.10 changed the export by **0 boxes** (export is post-tracker),
- a high-recall tracker (`nvdcf_mtmc_highrecall.yaml`: minDetConf 0.10, probationAge 1,
  minTrackerConf 0.20) added **+172 boxes (<1 %)** and moved the oracle only 0.9255→0.9301.

So to exceed IDF1 0.9 on W022 BOTH are required: (a) the linker reaching ~97 % of oracle (now 81 %),
and (b) lifting the detector recall ceiling from 0.87 toward ~0.95. Appearance fusion does NOT help
(deployed crops sit within ~0.2 cosine; gid-merge only lowered IDF1). **All MTMC-only — none of this
touches the MMP path.**

### @1280 detector retrain (2026-06-29) — resolution is NOT the lever

Retrained `yolo11n_mtmc_1280.onnx` (imgsz 1280, 30 ep, ~11 h; val mAP50 0.72 / recall 0.645 vs @960's
0.694 / 0.615). Deployed via separate `configs/models/nvinfer_yolov11_mtmc_1280.yml` +
`pipeline_mtmc_nvdcf_online_sgie_reid0_1280.yaml` (the @960 + MMP paths untouched). On W022:
**oracle ceiling moved only 0.926 → 0.932** (recall 0.861 → 0.872); linked IDF1 0.747 → 0.728 (noise).

Why so little? The missed GT boxes split ~50/50:
| GT height | % missed | nature |
|-----------|----------|--------|
| 0–32 px   | 99 %     | too tiny — resolution-limited |
| 32–64 px  | 43 %     | small — partly resolution |
| 64–128 px | 10 % (955 boxes) | medium people **occluded behind shelves/others** |
| ≥128 px   | ≤4 %     | detected fine |

Half the misses are occlusion (medium boxes), which **no detector resolution can recover** — they
need cross-camera detection *propagation* (reproject a confident world position back INTO the camera
that missed the person). The binding constraint has now flipped: oracle 0.932 > 0.9, so the **linker**
(at ~0.75, 78–81 % of oracle) is what blocks 0.9, not the detector.

### Global constrained linker (2026-06-29) — 0.747 → 0.856

`scripts/eval/mtmc_global_linker.py` replaces the greedy union-find with **constrained correlation
clustering**. Two ingredients union-find lacks:
- **MUST-NOT-LINK constraints** (hard): tracklets sharing a camera and overlapping in time are
  provably different people; so are cross-camera co-present tracklets whose world positions are far
  apart (> `conflict-thr`). No cluster may ever contain such a pair — this eliminates the swap errors
  that the per-frame/union-find linkers made (e.g. gid 4 = two different people).
- **AGGREGATE affinity**: positive evidence summed over all member pairs (cross-camera spatial
  coincidence + velocity-consistent temporal hand-offs), so merges are driven by total support, not
  a single lucky edge.

Best on W022 @1280 (`spatial-thr 1.5, conflict-thr 4.0, temporal-gap 400, pred-thr 3.5, min-merge 0.5`):

| Linker | Global IDF1 | TP | FP | pred IDs (GT=17) |
|--------|-------------|----|----|------------------|
| appearance buffered | 0.406 | – | – | 21 |
| position per-frame | 0.739 | 13.0k | 4.4k | 26 |
| position tracklet union-find | 0.747 | 14.2k | – | 29 |
| **global constrained clustering** | **0.856** | **16.4k** | **1.5k** | **25** |
| *oracle (perfect IDs)* | *0.932* | *17.9k* | *0* | *17* |

**0.856 = 92 % of the oracle ceiling** — full progression on W022 is **0.287 → 0.856** (3.0×). The
remaining gap to 0.9 is two irreducible-by-geometry pieces: (a) ~8 % of GT boxes never detected
(occlusion, caps the oracle at 0.932 — needs cross-camera detection reprojection), and (b) a handful
of tracklets separated by long out-of-view gaps that position+motion can't safely bridge and the
deployed appearance is too weak to disambiguate.
**All MTMC-only — none of this touches the MMP path.**

### Occlusion reprojection (2026-06-29) — built and tested; 0.9 is not reachable on this data

`src/mtmc/mtmc_calib.py` now also does `world_to_pixel` (validated: synthesised boxes from a correct
world position hit IoU≥0.5 on ~91 % of cases). `scripts/eval/mtmc_occlusion_fill.py` reprojects a
person localised by one camera into a camera that missed them, to recover occlusion FNs:
- **bracket mode** (fill only within-camera gaps bracketed by real detections — safe): +0.001 IDF1
  (0.856 → 0.857); only ~360 boxes are brief dropouts, most misses are sustained.
- **global-span mode** (fill a gid's whole active span in every camera that ever saw it): **HURTS**,
  0.857 → 0.787 — synthesises ~4 000 boxes but adds ~3 700 FP for ~280 TP.

Why it can't reach 0.9 — a hard *data* ceiling, measured on the 2 544 missed GT boxes:
- **61 % are detected in NO camera at that frame** (occluded in *every* view at once → no world
  position exists to reproject from),
- only **30 % (754)** are localised by another camera AND project into the missing camera's frustum;
  reprojecting all 754 with ZERO false positives still caps IDF1 at only **~0.878**,
- the frustum test cannot tell "occluded by a static shelf" (→ FP) from "visible" (→ TP), so any
  aggressive reprojection adds 2–3 FP per recovered TP → net loss.

**Final verdict for W022: best honest cross-camera Global IDF1 = 0.856** (global constrained linker;
bracketed occlusion fill 0.857). Exceeding 0.9 is blocked by the detector/occlusion ceiling, not the
linker — it would need joint *multi-view detection* (detect at the fused world level) or a static
occupancy/occlusion map to gate reprojection. Both are research-scope. **All MTMC-only.**

### Multi-warehouse: W020 / W021 (16 cams each) + 20-cam perf (2026-06-29)

W022 ships only 4 cameras; **W020 and W021 have 16 each**. Ran both (`--no-tiler`, @960 reid0),
linked, and tuned against GT.

**Critical bug fixed:** the export `cam_N` is the source-list **order**, not the `Camera_XXXX`
number — and W020/W021 cameras are non-contiguous (no 0006/0007/0008/0014). The linker + renderer
were using the wrong per-camera calibration (and silently dropping cameras whose export index had no
calibration key), corrupting world positions. Both now take `--sources <list>` to map cam_N → the
real calibration camera. (`score_mtmc_w.py` applies the same remap for GT alignment.)

| Warehouse | broken calib | calib fixed | + tuned | oracle (perfect IDs) |
|-----------|--------------|-------------|---------|----------------------|
| W020 (16 cam) | 0.227 | 0.438 | **0.507** | 0.745 (recall 0.59) |
| W021 (16 cam) | – | – | **0.443** | – |

Tuning (`scripts/eval/mtmc_tune_linker.py` sweeps params, builds tracklets+GT once; clustering
rewritten O(n³)→sparse, 11 s vs >10 min at 1060 tracklets): the big sparse warehouses need
**`spatial_thr=9, conflict_thr=18`** — their distant cameras have larger back-projection error, so
W022's tight 1.5/4.0 thresholds wrongly *forbade* genuine cross-camera pairs (must-not-link). W022
keeps its own 1.5/4.0 optimum (compact, low error). The dominant cap on W020 is **detection recall
0.59** — ~41 % of people are never detected (large warehouse, small/distant), so even perfect IDs
cap at 0.745. Demos (tiled cross-camera + top-down on `map.png`) re-rendered for all three warehouses
under `output/demo/mtmc_warehouse_0XX/` via `mtmc_bev_demo.py --mode {tiled,bev,split,camera}`.

**20-camera VRAM/FPS** (measured, `mtmc_val_20cam.txt` = 16×W020 + 4×W022, @960 reid0,
maxTargetsPerStream=100, steady-state, `--no-display --no-sync`):

| Pipeline | maxTargets | 20-cam VRAM | FPS/cam |
|----------|-----------|-------------|---------|
| MTMC @960 reid0 | 100 | **~6.3 GB** | **~11–12** |
| (MMP @960 reid0, for reference) | 40 | ~3.5 GB | ~15 |

MTMC is heavier than MMP at the same 20 cameras — 1080p sources, `maxTargetsPerStream=100` (vs 40),
and more people per frame — but fits comfortably in 16 GB.
