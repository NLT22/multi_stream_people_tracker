# DeepStream Learning Notes

Concept explanations for everything you encounter in the milestones.
Read the relevant section before starting a milestone, then return here
when something is confusing.

---

## 1. NVMM Memory — Zero-Copy GPU Buffers

After decoding, video frames live on the GPU in NVMM (NVIDIA Video Memory Manager).
All DeepStream elements (muxer, nvinfer, tracker, OSD) read and write NVMM directly
— no CPU round-trips.

**You'll see this in GStreamer caps:**
```
video/x-raw(memory:NVMM), format=NV12, width=1920, height=1080
```

**Why it matters:** Copying a 1080p frame GPU→CPU→GPU takes ~5ms.
At 30 FPS × 11 streams = 1650ms/s of pure copy overhead if not avoided.
NVMM eliminates this entirely.

---

## 2. nvurisrcbin — The Universal Source (Milestone 1+)

**Problem:** Raw GStreamer needs a different chain per format:
```
filesrc → qtdemux → h264parse → nvv4l2decoder   (H.264)
filesrc → qtdemux → h265parse → nvv4l2decoder   (H.265)
rtspsrc → rtph264depay → h264parse → ...        (RTSP)
```

**`nvurisrcbin` replaces all of this:**
- Accepts `file://`, `rtsp://`, `http://` URIs
- Auto-detects container and codec
- Always outputs NVMM NV12 (zero-copy into muxer)

**Always convert local paths to URIs:**
```python
uri = "file://" + os.path.abspath("/home/user/video.mp4")
```

---

## 3. nvstreammux — Why Even 1 Stream Needs It (Milestone 1+)

`nvurisrcbin` outputs raw decoded frames. But `nvinfer`, `nvtracker`, and `nvosdbin`
all require **batched buffers with NvDsBatchMeta attached**.
The muxer is the ONLY element that creates NvDsBatchMeta.

**What nvstreammux does:**
1. Collects one frame from each source
2. Packages them into a batch buffer
3. Attaches NvDsBatchMeta with per-source FrameMetadata
4. Passes the batch downstream

**`batch-size` must equal the number of active sources.**

---

## 4. `sink_%u` — GStreamer Request Pads (Milestone 1+)

nvstreammux creates sink pads on demand using a template:

```python
# CORRECT — GStreamer creates sink_0, sink_1, sink_2, ... automatically
pipeline.link(("source_0", "mux"), ("", "sink_%u"))
pipeline.link(("source_1", "mux"), ("", "sink_%u"))

# WRONG — will raise an error
pipeline.link(("source_0", "mux"), ("", "sink_0"))
```

The `%u` is a printf template. Each call with `"sink_%u"` gets the next integer.

---

## 5. nvinfer — TensorRT Inference (Milestone 3+)

nvinfer reads batch buffers, runs TensorRT inference, and writes
`NvDsObjectMeta` (bounding box, class_id, confidence) into the metadata.

**Config file rules (YAML format):**
- Top-level section MUST be `property:` — not `model:`, not `settings:`
- All paths inside the config are **relative to the config file's directory**
  (not the working directory where you run Python)

```yaml
property:
  gpu-id: 0
  onnx-file: /absolute/path/to/model.onnx        # absolute = always safe
  labelfile-path: ../labels/trafficcamnet.txt     # relative to THIS config file
  model-engine-file: ../../engine_cache/model.engine  # writable location
  batch-size: 4
  network-mode: 2   # FP16
  network-type: 0   # 0=detector
```

**First run:** builds TensorRT engine from ONNX (~1 min on RTX 3050Ti).
**Subsequent runs:** loads cached `.engine` file (< 5s).

---

## 6. nvosdbin — Rendering Bounding Boxes (Milestone 3+)

nvosdbin reads `NvDsObjectMeta` from nvinfer and renders:
- Default bounding rectangles with class labels
- Any custom text/rects added by probe callbacks (via `DisplayMeta`)

**Must come AFTER nvinfer in the pipeline.** Without it, detections are
invisible (metadata exists but nothing draws it).

```
nvinfer → nvosdbin → tiler → sink
          ↑
          also reads DisplayMeta added by probes
```

---

## 7. Custom Probes — The Correct API (Milestone 4+)

```python
import pyservicemaker as psm

class MyProbe(psm.BatchMetadataOperator):
    def handle_metadata(self, batch_meta):   # ← NOT "execute"
        for frame_meta in batch_meta.frame_items:
            ...

# Attach — custom probes MUST be wrapped in psm.Probe()
pipeline.attach("tracker", psm.Probe("my_probe", MyProbe()))

# Built-in probes use string name with NO dict argument
pipeline.attach("pgie", "measure_fps_probe", "fps")
```

**Two common mistakes:**
1. Using `execute` instead of `handle_metadata` → probe never fires
2. Passing a dict to built-in probe → `TypeError: incompatible arguments`

---

## 8. Iterator vs List — The Most Common Bug (Milestone 4+)

```python
# WRONG — object_items is an ITERATOR, not a list
n = len(frame_meta.object_items)   # TypeError: no len()

# CORRECT — iterate to count
n = sum(1 for _ in frame_meta.object_items)

# ALSO CORRECT — convert once if you need multiple passes
objects = list(frame_meta.object_items)
n = len(objects)
for obj in objects:   # can iterate again
    ...
```

Same rule applies to `batch_meta.frame_items`.

**Why iterators?** Avoids allocating Python lists for every frame at 30 FPS × 11 streams.

---

## 9. nvtracker — Persistent IDs (Milestone 4+)

`nvinfer` detects objects per-frame with no memory across frames.
`nvtracker` links detections across frames and assigns a stable `object_id`.

```
nvinfer: frame 1 → "person at (100,200)"
         frame 2 → "person at (105,203)"    ← same person? nvinfer doesn't know
nvtracker:         "person at (100,200) in frame 1 = ID #42"
                   "person at (105,203) in frame 2 = ID #42"  ← same person!
```

**Without tracker:** `obj_meta.object_id` is always 0.
**With tracker:** `obj_meta.object_id` persists until the person leaves the scene.

**IDs are per-stream.** Person #42 in cam0 ≠ Person #42 in cam1.
Cross-camera matching requires Re-ID (Milestone 8).

---

## 10. Tracker Algorithms — Choose Your Complexity

| Tracker | Algorithm | GPU | Handles Occlusion | Start With |
|---------|-----------|-----|-------------------|-----------|
| `iou.yaml` | Bounding box overlap | No | No | ✅ Understand the concept |
| `nvdcf_perf.yaml` | Correlation filter (HOG+Color) | Yes | Partial | ✅ Daily use |
| `nvdcf_accuracy.yaml` | Same, stronger features | Yes | Better | After perf |
| NvDeepSORT (M8) | ReID + association | Yes | Yes (cross-cam) | Advanced |

**IoU mental model:** "A box at (100,200) in frame N probably overlaps the
closest box in frame N+1. If overlap > 30%, same object."

**NvDCF mental model:** "I learned what this person looks like (color + edge features).
I keep tracking them via correlation filter even if the detector misses them for 5 frames."

---

## 11. nvmultistreamtiler — NxN Grid (Milestone 2+)

Without the tiler, the sink only renders the last stream in the batch.
The tiler composites all streams into a single grid canvas.

**Grid formula:**
```python
cols = math.ceil(math.sqrt(n))   # e.g. n=7 → cols=3
rows = math.ceil(n / cols)       # e.g. n=7 → rows=3  (9 cells, 2 empty)
total_width  = tile_w * cols
total_height = tile_h * rows
```

Examples: n=4 → 2×2, n=7 → 3×3, n=11 → 4×3

---

## 12. FP16 on RTX 3050Ti (Milestone 3+)

```yaml
# In nvinfer config:
network-mode: 2   # 0=FP32, 1=INT8, 2=FP16
```

FP16 cuts model memory by ~50% vs FP32 with < 1% accuracy loss on most detectors.

**VRAM budget for 11 streams at FP16:**
- TrafficCamNet model: ~60 MB
- 11 × 1080p NV12 batch buffers: ~360 MB
- Tracker: ~100 MB
- Total: ~520 MB — well within 4 GB

**If VRAM is tight:**
1. Add `interval: 2` in nvinfer config (inference every 3rd frame)
2. Reduce `tile_w`/`tile_h` to 640×360
3. Lower `batch-size` and use fewer streams

---

## 13. DeepStream 9.0 Install Paths

```
Base:        /opt/nvidia/deepstream/deepstream-9.0/
Models:      /opt/nvidia/deepstream/deepstream-9.0/samples/models/
Tracker lib: /opt/nvidia/deepstream/deepstream-9.0/lib/libnvds_nvmultiobjecttracker.so
PSM wheel:   /opt/nvidia/deepstream/deepstream-9.0/service-maker/python/pyservicemaker*.whl
```

> DeepStream 9.0 uses `deepstream-9.0/` NOT `deepstream/`.
> Older tutorials may show the wrong path.

---

## 14. TrafficCamNet Class IDs

| class_id | Label | Filter in probe |
|----------|-------|----------------|
| 0 | Vehicle | `obj.class_id == 0` |
| 1 | Bicycle | `obj.class_id == 1` |
| **2** | **Person** | `obj.class_id == 2` ← our target |
| 3 | RoadSign | `obj.class_id == 3` |

```python
PERSON_CLASS_ID = 2
for obj_meta in frame_meta.object_items:
    if obj_meta.class_id != PERSON_CLASS_ID:
        continue
    # this is a person detection
```

---

## 15. Probe Attachment Points

```
mux → pgie → tracker → [probe here] → osd → tiler → sink
                ↑           ↑
         measure_fps_probe  PersonLabelProbe
         (built-in)         (custom, before OSD)
```

- **`measure_fps_probe`** → attach to `"pgie"` (processing element), NEVER to sink
- **Custom label probes** → attach to `"tracker"` so OSD renders the text
- **Metadata read-only probes** → can attach to `"tracker"` or `"pgie"`
