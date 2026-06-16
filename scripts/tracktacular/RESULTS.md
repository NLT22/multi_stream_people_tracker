# 3-Way Method Comparison — TrackTacular (lifting_BEV) vs Anchor-guided vs Current

**Scope/status:** completed on the industry environment (the hardest scene,
"industry first" per plan). The 4-env expansion (cafe/lobby/office/retail) was
**intentionally stopped** after the throughput benchmark showed TrackTacular
cannot meet 20-cam/10-FPS regardless of accuracy — so the remaining ~12h of
training was not spent. The integration is complete and reproducible
(`run_env.sh <env> <scene> <ncam>`) if the per-env IDF1 numbers are wanted later.

Scene: **63am_industry_safety_0** (the hardest scene; "industry first" per scope).
Split: TrackTacular standard protocol = first 90% train / **last 10% test** (293
multi-cam frames). All three scored on the **same topdown GT** with the same
metric (motmetrics, BEV center points, 1 m gate) via `bev_compare.py`.

## BEV-space comparison (common ground; native metric for TrackTacular)

| Method | IDF1 | MOTA | IDsw | Notes |
|---|---|---|---|---|
| **TrackTacular** (SegNet/bilinear, 30ep) | **0.448** | **0.619** | 30 | native BEV; best localization |
| **Anchor-guided** | 0.436 | -0.009 | 41 | identity ≈ TrackTacular; BEV localization noisy |
| **Current** (online) | 0.281 | -0.305 | 110 | image-space tracker; weakest in BEV |

- TrackTacular's own test metric: track/IDF1 **45.4%**, MOTA 65.9%, detect
  recall **55.7%** (misses detections -> caps IDF1).
- Current/anchor are **image-space** methods; their world positions come from
  monocular foot projection, which is geometrically unreliable (**36% of points
  land outside the scene** — grazing rays to millions of mm). Robust median +
  plausible-box filtering applied for fairness; even so BEV MOTA ~0 (identities
  consistent, but per-frame localization noisy).

## Important: the metrics measure different things

- **BEV space (above)**: TrackTacular wins — it is a native BEV localizer.
- **Image-space Global IDF1** (`metrics_mmp`, current/anchor's native metric,
  full scene): anchor-guided = **0.754**, current = 0.385. TrackTacular produces
  no per-camera image predictions, so it cannot be placed in this column without
  projecting BEV->image (not what it's built for).

So the two families optimise different objectives and neither dominates across
both metric spaces.

## Throughput (the 20-cam/10-FPS target)

Clean inference-only benchmark (`bench_fps.py`): model.forward only, warmup
excluded, no disk I/O / tracking / metrics. RTX 5060 Ti, 720x1280/cam, batch 1.

| Cameras | ms/timestep | multi-cam FPS | cam-frames/s | peak VRAM |
|---|---|---|---|---|
| 4 | 49.2 | 20.3 | 81.3 | 0.93 GB |
| 5 | 63.7 | 15.7 | 78.5 | 1.14 |
| 6 | 71.9 | 13.9 | 83.5 | 1.35 |
| 7 | 83.8 | 11.9 | 83.6 | 1.56 |
| **20** | 233.6 | **4.28** | 85.6 | 4.32 GB |

- Throughput scales ~linearly with camera count (per-camera CNN encoder
  dominates; cam-frames/s ~constant ~82).
- **20 cam = 4.3 multi-cam FPS** — ~2.3x below the 10 FPS target, and this is
  optimistic (excludes detection-decode, JDETracker association, data loading).
- The earlier "~9 FPS" was a rough end-to-end figure at 4 cam (with I/O); the
  clean model-only rate at 4 cam is 20.3 FPS.

| Method | 20-cam throughput | 20-cam/10-FPS? |
|---|---|---|
| TrackTacular (SegNet, model-only) | 4.3 FPS | **No** (~2.3x short) |
| Current (online, DeepStream) | 16.6 fps/cam | Yes (realtime) |

## Verdict vs the target (0.8 Global IDF1, all scenes, 20 cam @ 10 FPS)

- **Not met on industry by any method.** TrackTacular's first untuned run (0.448
  BEV-IDF1) ≈ anchor on identity, better on localization, but far from 0.8 and
  not realtime at 20 cam. Anchor reaches 0.75 in *image-space* Global IDF1 but is
  not a BEV localizer and the BEV target conflates two metrics.
- TrackTacular is a **first 30-epoch single-scene** model (affine ~270 mm error,
  detect recall 56%); headroom exists via more epochs, multi-scene training,
  higher BEV resolution, and a tighter grid->world fit.

## Final findings (industry; expansion stopped by decision)

1. **TrackTacular is the better BEV localizer** (BEV-IDF1 0.448 / MOTA 0.619 vs
   anchor 0.436 / ~0, current 0.281 / <0) — early-fusion in BEV beats projecting
   an image-space tracker into BEV. But its first untuned run is far from 0.8 and
   detect-recall (56%) is the cap.
2. **It is not realtime at scale**: 4.3 multi-cam FPS at 20 cam (model-only,
   optimistic) — ~2.3x short of 10 FPS. The current DeepStream pipeline (16.6
   fps/cam) remains the deployable option.
3. **Metric spaces differ**: anchor-guided still wins *image-space* Global IDF1
   (0.754 full-scene). The two families optimise different objectives; neither
   dominates both accuracy spaces, and only the current pipeline is realtime.
4. **Verdict on the target (0.8 all scenes @ 20cam/10FPS)**: not achievable with
   TrackTacular (fails FPS) nor with the current/anchor pipeline (fails 0.8 on the
   hard scenes). The target is over-constrained: BEV early-fusion buys accuracy
   at a throughput cost; the realtime ReID pipeline buys throughput at an accuracy
   cost on look-alike-heavy scenes.

## Optimisation levers (if TrackTacular accuracy is pursued later, ignoring FPS)
1. Train on **all industry scenes** (more data; needs multi-sequence dataset).
2. More epochs + higher input/BEV resolution (recall is the bottleneck).
3. Refine the grid->world affine (reduce the 270 mm GT error).
4. Try `liftnet` (depth-splat) / longer schedule.

## Repro
```
bash scripts/tracktacular/apply_integration.sh
python scripts/tracktacular/mmp_to_worldtrack.py --scene 63am_industry_safety_0 \
    --out dataset/worldtrack/mmp_industry_safety_0 --frame-step 2
cd reference/TrackTacular/WorldTrack
python world_track.py fit  -c configs/t_fit.yml -c configs/d_mmp_industry.yml -c configs/m_segnet.yml
python world_track.py test -c lightning_logs/version_0/config.yaml --ckpt <best.ckpt>
python scripts/tracktacular/bev_compare.py --gt <ver>/mota_gt.txt --tt-pred <ver>/mota_pred.txt \
    --current-dir output/eval/clean_63am_industry_safety_0 \
    --anchor-dir  output/eval/anchor_63am_industry_safety_0
```
