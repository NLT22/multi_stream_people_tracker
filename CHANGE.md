# CHANGE.md — current problem: we cannot improve FPS without losing IDF1

The pipeline is stuck at **~11 FPS/cam** (20 cams, 640×360, mean IDF1 ~0.81). Every lever tried
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

**Next steps:**
1. Measure at real resolution — run on **MTMC_Tracking_2026** (1920×1080, ~20 cams) for the true FPS ceiling.
2. If unreachable: the only real levers left are structural — move SGIE ReID out of the per-frame
   graph (sidecar prototype hit ~17.8 FPS), downscale decoder input, a lighter detector, or scale
   across more GPUs. (A Nsight Systems trace would pin which GPU stage dominates.)
