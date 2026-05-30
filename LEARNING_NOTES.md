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
  labelfile-path: ../labels/coco_labels.txt        # relative to THIS config file
  model-engine-file: ../../models/<model_dir>/model.onnx_b4_gpu0_fp16.engine
  batch-size: 4
  network-mode: 2   # FP16
  network-type: 0   # 0=detector
```

**First run:** builds TensorRT engine from ONNX (~1 min on RTX 3050Ti).
**Subsequent runs:** loads the `.engine` file from the model directory (< 5s).

In this project, engines are written next to their source models:

```text
models/yolov11/yolo11n.onnx_b4_gpu0_fp16.engine
models/yolov8/yolov8n.onnx_b4_gpu0_fp16.engine
models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx_b4_gpu0_fp16.engine
models/peoplenet/resnet34_peoplenet.onnx_b4_gpu0_fp16.engine
models/reid/resnet50_market1501.etlt_b16_gpu0_fp16.engine
```

Do not commit `.engine` files. They are specific to the GPU, driver, CUDA,
TensorRT, and DeepStream versions that generated them.

---

## 6. nvosdbin — Rendering Bounding Boxes (Milestone 3+)

nvosdbin reads `NvDsObjectMeta` from nvinfer and renders:
- Default bounding rectangles with class labels
- Any custom text/rects added by probe callbacks (via `DisplayMeta`)

**Must come AFTER tiler AND after any probe that adds text.**

```
mux → nvinfer → tracker → tiler → [probe adds text] → nvosdbin → sink
                             ↑                              ↑
                      scales metadata coords         draws everything
```

If nvosdbin is placed before the tiler, all streams' boxes collapse onto
one stream's surface using un-scaled coordinates.

---

## 7. Custom Probes — The Correct API (Milestone 4+)

```python
import pyservicemaker as psm

class MyProbe(psm.BatchMetadataOperator):
    def handle_metadata(self, batch_meta):   # ← NOT "execute"
        for frame_meta in batch_meta.frame_items:
            ...

# Attach — custom probes MUST be wrapped in psm.Probe()
pipeline.attach("tiler", psm.Probe("my_probe", MyProbe()))

# Built-in probes use string name with NO dict argument
pipeline.attach("pgie", "measure_fps_probe", "fps")
```

**Two common mistakes:**
1. Using `execute` instead of `handle_metadata` → probe never fires
2. Passing a dict to built-in probe → `TypeError: incompatible arguments`

---

## 7b. OSD / DisplayMeta — Adding Text Overlays (Milestone 4+)

The OSD API lives in the `pyservicemaker.osd` submodule — import it separately.

```python
from pyservicemaker import osd

class MyProbe(psm.BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            # Acquire DisplayMeta from the batch (one per frame)
            display_meta = batch_meta.acquire_display_meta()

            for obj_meta in frame_meta.object_items:
                text = osd.Text()
                text.display_text = f"ID #{obj_meta.object_id}".encode()  # must be bytes
                text.x_offset = int(obj_meta.rect_params.left)
                text.y_offset = max(0, int(obj_meta.rect_params.top) - 20)
                text.font.name = osd.FontFamily.Serif  # Serif is the only available font
                text.font.size = 14
                text.font.color = osd.Color(0.0, 1.0, 0.0, 1.0)  # RGBA
                display_meta.add_text(text)

            # Must append to frame_meta — without this, text is silently discarded
            frame_meta.append(display_meta)
```

**Key points:**
- `display_text` expects **bytes** — use `"my label".encode()` or `b"my label"`
- Position uses `x_offset` / `y_offset`, not `x` / `y`
- Only `osd.FontFamily.Serif` is available in DeepStream 9.0 pyservicemaker
- `frame_meta.append(display_meta)` is required to register the overlay with the pipeline

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
- Detector engine: usually tens to low hundreds of MB, depending on model
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

## 14. Person Class IDs

The current default detector is YOLO11n COCO:

| Model | Config | Person class_id |
|-------|--------|-----------------|
| YOLO11n COCO | `configs/models/nvinfer_yolov11_people.yml` | 0 |
| YOLOv8n COCO | `configs/models/nvinfer_yolov8_people.yml` | 0 |
| TrafficCamNet | `configs/models/nvinfer_trafficcamnet.yml` | 2 |
| PeopleNet | `configs/models/nvinfer_peoplenet.yml` | 0 |

Milestones 04-08 call `infer_person_class_id()` so the probes follow the
selected label file instead of hard-coding a single detector's person index.

```python
person_class_id = infer_person_class_id(nvinfer_config)
for obj_meta in frame_meta.object_items:
    if obj_meta.class_id != person_class_id:
        continue
    # this is a person detection
```

---

## 15. Correct Pipeline Order: Tiler BEFORE OSD

**Wrong (all boxes appear in one tile):**
```
mux → pgie → tracker → osd → tiler → sink
```

**Correct:**
```
mux → pgie → tracker → tiler → [probe] → osd → sink
                          ↑        ↑        ↑
                   composites   custom   draws on
                   N streams    labels   tiled canvas
                   scales coords
```

**Why order matters:** The tiler composites all N streams into one canvas
and scales each frame's metadata coordinates to the tile positions.
If OSD runs before the tiler, it gets un-scaled coordinates and all frames'
metadata is rendered onto one stream's surface.

**Probe attachment points:**
- **`measure_fps_probe`** → attach to `"pgie"`, NEVER to sink
- **Custom label probes** → attach to `"tiler"` (after tiler scales coords)
- **Metadata read-only probes** → attach to `"tiler"` for tiled coords, or `"tracker"` for original frame coords (useful for analytics, not for drawing)

---

## 16. Cross-Camera ReID Stabilization (Milestone 8)

Milestone 8 has two identity layers:

```text
NvDeepSORT local ID:
  stable inside one camera while the tracker keeps the same object_id

Python Global ID:
  links different camera-local IDs into one cross-camera identity
```

The current M8 Hungarian pipeline:

```text
  YOLO11 detector
  -> NvDeepSORT + Swin-Tiny tracker/ReID embedding
  -> SourceIdCollectorProbe
  -> Tiler
  -> CrossCameraGalleryProbe
       -> tracklet embedding
       -> gallery prototypes
       -> Hungarian one-to-one assignment
       -> ID stickiness / ambiguity rejection
       -> online Global ID merge
  -> OSD
```

### Why Each Method Exists

| Method | Symptom | Fix idea |
|--------|---------|----------|
| Tracklet embedding | Similarity changes wildly frame by frame. | Average recent embeddings per `(camera, local_track_id)`. |
| Gallery prototypes | Same person looks different across front/back/side cameras. | Keep several appearance vectors per Global ID. |
| Hungarian assignment | Two people in the same camera both pick one Global ID. | Solve one-to-one assignment per stream/frame. |
| Duplicate guard | A known Global ID appears twice in one stream. | Release the weaker duplicate back to assignment. |
| ID stickiness | Label bounces between two close IDs, e.g. `G14` and `G8`. | Require extra margin before switching away from previous ID. |
| Ambiguity rejection | Top-1 and top-2 scores are nearly tied. | Reject uncertain matches instead of choosing randomly. |
| Online Global ID merge | Opposite camera creates `G19` even though the person is already `G4`. | Merge stable duplicate IDs after enough tracklet evidence. |
| Bounded candidate search | Long videos create many temporary IDs and the pipeline lags. | Limit gallery candidates and run merge only every N batches. |

### Important Counters

`active_gids` is the number of Global IDs still in the gallery.

`total_gids_ever_assigned` is a historical counter. It only increases, even
when `G19` is merged into `G4`. Use `active_gids` to judge whether the gallery
is actually growing too large.

### Practical Tuning Order

Start from the default settings, then tune in this order:

1. If labels bounce between two IDs:
   increase `--id-switch-margin` or `--match-ambiguity-margin`.

2. If one person splits into two stable IDs across cameras:
   lower `--global-merge-threshold` slightly or reduce
   `--global-merge-min-embeddings`.

3. If false merges happen:
   increase `--global-merge-threshold`, increase
   `--global-merge-margin`, or disable merge for A/B testing.

4. If the pipeline slows down on long videos:
   lower `--gallery-max-age`, `--assignment-max-candidates`, or
   `--global-merge-max-candidates`; increase `--global-merge-interval`.

Useful commands:

```bash
# Inspect matching and merge decisions.
python -m src.main --debug-similarity

# Lower CPU load on long videos.
python -m src.main \
  --gallery-max-age 300 \
  --assignment-max-candidates 40 \
  --global-merge-interval 30 \
  --global-merge-max-candidates 20

# Compare with merge disabled.
python -m src.main --disable-global-merge
```
