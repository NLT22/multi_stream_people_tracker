# Multi-Stream People Tracker — Learning Project

A step-by-step learning skeleton for building a DeepStream 9.0 / pyservicemaker
multi-stream people tracking pipeline on an RTX 3050Ti (4 GB VRAM).

**Goal:** Understand each DeepStream component by building the pipeline
one stage at a time, from "play a video" to "track N people across M cameras
with bounding boxes and tiled display."

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Ubuntu | 24.04 | `lsb_release -r` |
| NVIDIA Driver | 590+ | `nvidia-smi` |
| CUDA | 13.1 | `nvcc --version` |
| DeepStream | 9.0 | `deepstream-app --version` |
| Python | 3.12 | `python3 --version` |
| TensorRT | 10.14 | bundled with DeepStream |

---

## Quick Start

```bash
# 1. Navigate to the project
cd multi_stream_people_tracker

# 2. Set up virtual environment (one time)
chmod +x setup_venv.sh
./setup_venv.sh

# 3. Activate venv (every new terminal)
source venv/bin/activate

# 4. Edit your video source (replace the placeholder path)
nano configs/sources/video_files.txt

# 5. Run Milestone 1
python milestones/01_single_video_display.py --input /path/to/video.mp4
```

---

## Learning Path — Milestones

Work through these in order. Each builds on the previous.

| # | Script | What You Learn | Run Command |
|---|--------|----------------|-------------|
| 1 | `01_single_video_display.py` | Pipeline basics, nvurisrcbin, nvstreammux, sink | `python milestones/01_single_video_display.py --input /path/to/video.mp4` |
| 2 | `02_multi_video_input.py` | N sources, batch-size, nvmultistreamtiler | `python milestones/02_multi_video_input.py` |
| 3 | `03_batching_streammux.py` | Batch buffers, batched-push-timeout, live-source | `python milestones/03_batching_streammux.py` |
| 4 | `04_people_detection.py` | nvinfer, TensorRT engine, FP16, FPS probe | `python milestones/04_people_detection.py` |
| 5 | `05_object_tracking.py` | nvtracker, persistent IDs, IoU vs NvDCF | `python milestones/05_object_tracking.py` |
| 6 | `06_osd_visualization.py` | nvosdbin, DisplayMeta, Text/Color/Font API | `python milestones/06_osd_visualization.py` |
| 7 | `07_tiled_display.py` | Full pipeline, NxN tiling, tile grid math | `python milestones/07_tiled_display.py` |
| 8 | `08_metadata_extraction.py` | BatchMeta traversal, iterators, JSON export | `python milestones/08_metadata_extraction.py` |
| 9 | `09_reid_extension_stub.py` | Re-ID concept, NvDeepSORT, cross-camera (stub) | `python milestones/09_reid_extension_stub.py` |

Each script is self-contained with a `--sources` argument (default:
`configs/sources/video_files.txt`) and its own TODO exercises.

---

## Switching Source Mode

Edit `configs/pipeline.yaml`:

```yaml
# Use local video files (start here)
source_mode: video_files

# Use a folder of videos
# source_mode: folder_input

# Use RTSP cameras (future)
# source_mode: rtsp_cameras
```

Then edit the corresponding config file in `configs/sources/`.

---

## Switching Detection Model

Edit the `detection.config_file` in `configs/pipeline.yaml`:

```yaml
detection:
  # Default: TrafficCamNet (bundled with DeepStream, no download needed)
  config_file: configs/models/nvinfer_trafficcamnet.yml

  # Future: YOLOv8 people-only (requires model export, see config for instructions)
  # config_file: configs/models/nvinfer_yolov8_people.yml
```

**TrafficCamNet class IDs:** 0=Vehicle, 1=Bicycle, **2=Person**, 3=RoadSign

---

## Switching Tracker

Edit `tracker.config_file` in `configs/pipeline.yaml`:

```yaml
tracker:
  # Recommended progression:
  config_file: configs/tracker/iou.yaml            # 1st: simplest, learn the concept
  # config_file: configs/tracker/nvdcf_perf.yaml   # 2nd: GPU visual tracker
  # config_file: configs/tracker/nvdcf_accuracy.yaml  # 3rd: more stable IDs
```

---

## Project Structure

```
multi_stream_people_tracker/
├── configs/
│   ├── pipeline.yaml              ← master control panel
│   ├── sources/
│   │   ├── video_files.txt        ← your video paths (edit this first!)
│   │   ├── folder_input.yaml
│   │   └── rtsp_cameras.txt
│   ├── models/
│   │   ├── nvinfer_trafficcamnet.yml   ← default model config
│   │   └── nvinfer_yolov8_people.yml  ← future model stub
│   ├── tracker/
│   │   ├── iou.yaml               ← simplest tracker
│   │   ├── nvdcf_perf.yaml        ← GPU tracker (fast)
│   │   └── nvdcf_accuracy.yaml    ← GPU tracker (accurate)
│   └── labels/
│       └── trafficcamnet_labels.txt
├── milestones/                    ← standalone learning scripts (01–09)
├── src/
│   ├── config/loader.py           ← PipelineConfig dataclass
│   ├── pipeline/
│   │   ├── builder.py             ← full pipeline assembler
│   │   ├── sources.py             ← URI loaders (txt/folder/rtsp)
│   │   └── probes.py              ← metadata probe classes
│   └── utils/platform_utils.py   ← sink type, GPU info
├── LEARNING_NOTES.md              ← concept explanations (read this!)
├── README.md                      ← this file
├── requirements.txt
└── setup_venv.sh
```

---

## Troubleshooting

### Black screen / no video

- **Cause:** Display not available or sink type wrong.
- **Check:** `echo $DISPLAY` — must be set (e.g. `:0`)
- **Fix:** Run in a local desktop session, or set `DISPLAY=:0` before running.

### `ModuleNotFoundError: No module named 'pyservicemaker'`

- **Cause:** Running without the venv, or venv missing the wheel.
- **Fix:** `source venv/bin/activate` then check `setup_venv.sh` ran correctly.

### `[NvDsInferContextImpl] Building network engine...` takes forever

- **Normal!** First run builds a TensorRT engine. Takes 1–3 min on 3050Ti.
- **After first run:** Subsequent runs load the cached `.engine` file in < 5s.

### `TypeError: object has no len()`

- **Cause:** Calling `len()` on `frame_meta.object_items` (it's an iterator).
- **Fix:** See `LEARNING_NOTES.md` § Iterator vs List.

### `RuntimeError: Probe failure`

- **Cause:** Attaching `measure_fps_probe` to a sink element.
- **Fix:** Attach to `"pgie"` or `"nvosdbin"` instead. See `LEARNING_NOTES.md`.

### Pipeline stuck at "Setting to PLAYING"

- **Cause:** Live source or tee split without `async=0` on sink.
- **Fix:** Add `"async": 0` to sink properties. See `LEARNING_NOTES.md`.

### VRAM out of memory with 4+ streams

- Reduce `batch_size` in `pipeline.yaml`
- Reduce `tile_width`/`tile_height` in `pipeline.yaml`
- Add `interval: 2` to the nvinfer config (run inference every 3rd frame)
- Ensure `network-mode: 2` (FP16) in the nvinfer config

---

## Key Paths (DeepStream 9.0)

```
TrafficCamNet ONNX: /opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/
Tracker library:    /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
Sample configs:     /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/
pyservicemaker WHL: /opt/nvidia/deepstream/deepstream/service-maker/python/
```
