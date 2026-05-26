# DeepStream Learning Notes

Reference explanations for concepts you will encounter in each milestone.
Read the relevant section before starting a milestone, then return here
when you see something unexpected.

---

## 1. NVMM Memory — Zero-Copy GPU Buffers

**What it is:** After decoding, video frames live on the GPU in a special
memory region called NVMM (NVIDIA Video Memory Manager). All DeepStream
elements (muxer, nvinfer, tracker, OSD) read from and write to NVMM
directly — no CPU copies.

**Why it matters:** Copying a 1080p NV12 frame from GPU to CPU and back
takes ~5ms. At 30 FPS with 4 streams, that's 600ms/s of pure copy overhead.
NVMM eliminates this entirely.

**You'll see it in GStreamer caps strings:**
```
video/x-raw(memory:NVMM), format=NV12, width=1280, height=720
```

---

## 2. nvurisrcbin — The Universal Source

**Problem it solves:** In raw GStreamer you need:
```
filesrc → qtdemux → h264parse → nvv4l2decoder
```
And you must change this chain for H.265, RTSP, MKV, etc.

**nvurisrcbin does all of that automatically:**
- Accepts `file://`, `rtsp://`, `http://` URIs
- Auto-detects container and codec
- Creates the correct parser+decoder internally
- Always outputs NVMM NV12 (GPU memory, zero-copy into muxer)

**Rule:** Always convert local paths before passing to nvurisrcbin:
```python
uri = "file://" + os.path.abspath("/home/user/video.mp4")
```

---

## 3. nvstreammux — Why Even 1 Stream Needs It

**The problem:** `nvurisrcbin` outputs raw decoded frames. But `nvinfer`,
`nvtracker`, and OSD all need **batched buffers with NvDsBatchMeta attached**.
The muxer is the only element that creates NvDsBatchMeta.

**What nvstreammux does:**
1. Collects one frame from each source
2. Packages them into a single "batch buffer"
3. Attaches NvDsBatchMeta with per-source FrameMetadata
4. Passes the batch downstream

**batch-size must equal the number of sources.** If batch-size=4 but you
have 3 sources, the muxer waits forever for source 4 that never comes.

---

## 4. `sink_%u` Pad Template — GStreamer Request Pads

nvstreammux has "request pads" — sink pads that don't exist until you ask
for them. GStreamer creates them on demand using a template name.

```python
# CORRECT — GStreamer creates sink_0, sink_1, sink_2, ... automatically
pipeline.link(("source_0", "mux"), ("", "sink_%u"))
pipeline.link(("source_1", "mux"), ("", "sink_%u"))

# WRONG — bypasses the request pad mechanism, will raise an error
pipeline.link(("source_0", "mux"), ("", "sink_0"))
```

The `%u` is a printf-style template. GStreamer replaces it with the next
available integer each time you call link with the template.

---

## 5. nvinfer Config Sections

nvinfer config files must have a section named `property:` (YAML) or
`[property]` (INI). Using any other name silently fails or errors.

**YAML format (preferred):**
```yaml
property:
  gpu-id: 0
  onnx-file: /path/to/model.onnx
  batch-size: 4
  network-mode: 2   # FP16

class-attrs-all:
  pre-cluster-threshold: 0.2
```

**INI format (also supported):**
```ini
[property]
gpu-id=0
onnx-file=/path/to/model.onnx
batch-size=4
network-mode=2
```

**Common mistake:** Using `model:` instead of `property:` → config silently
ignored, model never loads, pipeline errors with "failed to create element".

---

## 6. Iterator vs List — The Most Common Beginner Bug

```python
# WRONG — frame_meta.object_items is an ITERATOR, not a list
num_objects = len(frame_meta.object_items)   # ← raises: TypeError: object has no len()

# CORRECT — iterate to count
num_objects = sum(1 for _ in frame_meta.object_items)

# ALSO CORRECT — if you need multiple passes, convert once
objects = list(frame_meta.object_items)
num_objects = len(objects)  # ← now this works
for obj in objects:         # ← can iterate again
    ...
```

**Why iterators?** DeepStream metadata lives in C structs. The Python binding
exposes them as iterators to avoid allocating a full Python list for every
frame on every buffer. At 30 FPS × 4 streams = 120 buffers/sec, avoiding
list allocation matters.

**Same rule applies to:**
- `batch_meta.frame_items`
- `frame_meta.object_items`
- `frame_meta.tensor_items`

---

## 7. Tracker Algorithms — Choose Your Complexity

| Tracker | Algorithm | GPU? | Handles occlusion? | Good for learning? |
|---------|-----------|------|--------------------|--------------------|
| `iou.yaml` | Bounding box overlap | No | No | ✅ Start here |
| `nvdcf_perf.yaml` | Correlation filter (HOG+Color) | Yes | Partial | ✅ Graduate here |
| `nvdcf_accuracy.yaml` | Same, stronger features | Yes | Better | After NvDCF perf |
| NvDeepSORT (Milestone 9) | ReID + association | Yes | Yes (cross-cam) | Advanced |

**IOU tracker mental model:** "A bounding box that was at (100,200) in frame N
is probably the object closest to (100,200) in frame N+1. If overlap >30%, same object."

**NvDCF mental model:** "I learned what this person looks like (color histogram,
HOG gradients). Even if the detector misses them for 5 frames, I keep tracking
using a correlation filter on the appearance model."

---

## 8. FP16 on RTX 3050Ti

The RTX 3050Ti has 4GB GDDR6 VRAM. FP16 (half precision) cuts model memory
by ~50% versus FP32 with less than 1% accuracy loss for most detectors.

```yaml
# In nvinfer config:
network-mode: 2   # 0=FP32, 1=INT8, 2=FP16
```

**VRAM budget for 4 streams at 1280×720 FP16:**
- TrafficCamNet model: ~60MB
- 4 batch buffers (input/output): ~100MB
- Tracker: ~50MB
- Total: ~210MB — well within 4GB

**If you run out of VRAM:**
1. Reduce batch-size
2. Reduce tile resolution (tile_width/tile_height)
3. Use `interval: 2` in nvinfer to skip frames

---

## 9. `measure_fps_probe` — Attach to nvinfer, Not sink

```python
# CORRECT
pipeline.attach("pgie", "measure_fps_probe", "fps", {"print-fps-interval": 5})

# WRONG — raises RuntimeError: Probe failure
pipeline.attach("sink", "measure_fps_probe", "fps", {"print-fps-interval": 5})
```

The built-in `measure_fps_probe` can only attach to processing elements
(nvinfer, nvosdbin, etc.). Sink elements don't expose the required pad type.

---

## 10. async=0 for Live Sources and Tee Splits

When using RTSP cameras or a tee element to split the pipeline:

```python
# Required for live sources — prevents PAUSED state deadlock
pipeline.add(get_sink_element(), "sink", {
    "sync":  0,   # don't wait for frame timestamps
    "qos":   0,   # don't drop frames based on QoS
    "async": 0,   # CRITICAL — prevents state transition hang
})
```

**Symptom if missing:** Pipeline shows "Setting to PLAYING..." but never
actually plays. Window stays black. `pipeline.wait()` blocks forever.

**File sources:** Don't need async=0 (they're not live). Using it is harmless
but sync=1 will give smoother playback.

---

## 11. First Run — TensorRT Engine Build

On the first run of any milestone with nvinfer, you will see:
```
[NvDsInferContextImpl] Building network engine for TensorRT...
```

This takes **1–3 minutes** on an RTX 3050Ti. It builds a GPU-optimized
engine from the ONNX model and caches it as a `.engine` file.

The engine file name encodes the build parameters:
```
resnet18_trafficcamnet_pruned_b4_gpu0_fp16.engine
                                    ^  ^   ^
                               batch=4  gpu=0  FP16
```

If you change `batch-size`, `gpu-id`, or `network-mode`, a NEW engine is
built (old one is not deleted, just ignored). The new build takes another 1-3 min.

**Subsequent runs:** Load the cached `.engine` file in < 5 seconds.

---

## 12. TrafficCamNet Class IDs

TrafficCamNet (the default model) detects 4 classes. The mapping matters
for filtering in your probe callbacks:

| class_id | Label |
|----------|-------|
| 0 | Vehicle |
| 1 | Bicycle |
| 2 | **Person** ← our target |
| 3 | RoadSign |

```python
# In any probe:
PERSON_CLASS_ID = 2
for obj_meta in frame_meta.object_items:
    if obj_meta.class_id != PERSON_CLASS_ID:
        continue   # skip vehicles, bicycles, signs
    # ... process this person
```
