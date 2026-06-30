# CHANGE.md — FPS vs IDF1 trade-off, and the 1080p question (RESOLVED)

> **Resolved 2026-06-24:** the open question below — *"can 20 cams hit ≥10 FPS at real 1080p?"* —
> was measured on `MTMC_Tracking_2026` (1920×1080, 20 cams) and **the target was met: ~11.9 FPS/cam
> at 1080p, faster than the 640×360 MMP run** (decode is not the dominant term it was feared to be).
> VRAM/FPS is driven by `maxTargetsPerStream`, not resolution (full table in CLAUDE.md / production_todo §0).
> The lever analysis below is kept because it still explains *why* per-object GPU work — not
> resolution — sets the accuracy-preserving operating point.

The pipeline runs at **~11 FPS/cam** (20 cams, 640×360, mean IDF1 ~0.81). Every FPS lever tried
either does nothing or trades away accuracy:

| Lever | FPS | IDF1 |
|---|---|---|
| Detector INT8 | ~0 change | held |
| ReID (Swin) INT8 | ~0 change | held (−0.37 GB VRAM) |
| SGIE `interval` >0 | ~0 change | held |
| Python probe opts (gallery hoist / `.cpu()` batch) | ~0 change | held |
| nvstreammux 640×360 (vs 1080p) | ~0 change | collapses 0.80→0.71 |
| **Detector `interval` 2** | **+72% (→18 FPS)** | **collapses 0.80→0.65** |

**Why:** the bottleneck is **GPU per-object work at FIXED output sizes** — the detector always
resizes to 640×640, the SGIE always resizes crops to 256×128, plus the tracker — none of which
scale with mux/surface resolution or shrink with quantization. That's why INT8, Python opts, SGIE
interval, AND mux-shrink all did nothing; only cutting the *number* of detections (detector
interval) raises FPS, and that destroys recall/IDF1. Accuracy-preserving operating point is
detector-every-frame at ~11 FPS. (Full lever matrix + mechanics in agent memory.)

**Why this matters now:** production cameras are **1920×1080**. The MMP test can't reveal the
production cost because its source videos are 360p (decode runs pre-mux at the *source* resolution,
so our mux experiments never paid the 1080p decode bill). At native 1080p sources the **decode of
20 streams** (resolution-proportional) is a new heavy term on top of the fixed per-object work, so
**20-cam @ 10 FPS is very likely unreachable at 1080p on the single RTX 5060 Ti — and is unmeasured.**

**Outcome:**
1. ✅ Measured at real resolution on **MTMC_Tracking_2026** (1920×1080, 20 cams): **~11.9 FPS/cam,
   target met** — the feared 1080p-decode bill did not dominate.
2. Remaining structural levers (only if a future deployment needs more headroom): move SGIE ReID out
   of the per-frame graph (sidecar prototype hit ~17.8 FPS), downscale decoder input, a lighter
   detector, or scale across more GPUs. A Nsight Systems trace would pin the dominant GPU stage.
