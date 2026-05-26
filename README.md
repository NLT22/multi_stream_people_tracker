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

# 4. Edit video sources (replace placeholder paths)
nano configs/sources/video_files.txt

# 5. Run Milestone 1 — single video playback
python milestones/01_single_video_display.py --input /path/to/video.mp4

# 6. Run Milestone 3 — people detection with bounding boxes
python milestones/03_people_detection.py
```

---

## Learning Path — 8 Milestones

Each milestone has **visual output on screen**. Work through them in order.

| # | Script | Visual Output | DeepStream Element(s) Added |
|---|--------|--------------|----------------------------|
| 1 | `01_single_video_display.py` | Video plays | `nvurisrcbin` → `nvstreammux` → sink |
| 2 | `02_multi_video_tiled.py` | NxN grid of videos | + `nvmultistreamtiler` |
| 3 | `03_people_detection.py` | **Bounding boxes on all detections** | + `nvinfer` + `nvosdbin` |
| 4 | `04_people_tracking.py` | **"Person #42" labels per person** | + `nvtracker` + probe |
| 5 | `05_multi_stream_tracking.py` | **NxN grid + "Cam0 #42" labels** | Scale M4 to N streams |
| 6 | `06_batching_deep_dive.py` | Full visual + batch logs in console | `nvstreammux` internals |
| 7 | `07_metadata_extraction.py` | Full visual + per-camera stats | metadata traversal |
| 8 | `08_reid_stub.py` | Concept guide (no runnable code) | ReID / NvDeepSORT |

Run commands (all use `--sources configs/sources/video_files.txt` by default):

```bash
python milestones/01_single_video_display.py --input /path/to/video.mp4
python milestones/02_multi_video_tiled.py
python milestones/03_people_detection.py
python milestones/04_people_tracking.py
python milestones/04_people_tracking.py --tracker-config configs/tracker/iou.yaml
python milestones/05_multi_stream_tracking.py
python milestones/05_multi_stream_tracking.py --tile-w 640 --tile-h 360
python milestones/06_batching_deep_dive.py
python milestones/07_metadata_extraction.py
python milestones/07_metadata_extraction.py --save-json
python milestones/08_reid_stub.py   # prints concept guide
```

---

## Switching Source Mode

Edit `configs/pipeline.yaml`:

```yaml
# Use local video files (start here)
source_mode: video_files

# Use a folder of videos
# source_mode: folder_input

# Use RTSP cameras
# source_mode: rtsp_cameras
```

Then edit the corresponding config file in `configs/sources/`.

---

## Switching Detection Model

Edit `detection.config_file` in `configs/pipeline.yaml`:

```yaml
detection:
  # Default: TrafficCamNet (bundled with DeepStream, no download needed)
  config_file: configs/models/nvinfer_trafficcamnet.yml

  # Future: YOLOv8 people-only (requires model export, see stub config)
  # config_file: configs/models/nvinfer_yolov8_people.yml
```

**TrafficCamNet class IDs:** 0=Vehicle, 1=Bicycle, **2=Person**, 3=RoadSign

---

## Switching Tracker

Edit `tracker.config_file` in `configs/pipeline.yaml`:

```yaml
tracker:
  config_file: configs/tracker/iou.yaml            # start here: simplest
  # config_file: configs/tracker/nvdcf_perf.yaml   # GPU visual tracker (recommended)
  # config_file: configs/tracker/nvdcf_accuracy.yaml  # slower, most stable IDs
```

**Recommended progression:** `iou.yaml` → `nvdcf_perf.yaml` → `nvdcf_accuracy.yaml`

---

## Project Structure

```
multi_stream_people_tracker/
├── configs/
│   ├── pipeline.yaml                  ← master control panel (source/model/tracker)
│   ├── sources/
│   │   ├── video_files.txt            ← edit this first — add your video paths
│   │   ├── folder_input.yaml
│   │   └── rtsp_cameras.txt
│   ├── models/
│   │   ├── nvinfer_trafficcamnet.yml  ← default detector (FP16, class 2 = person)
│   │   └── nvinfer_yolov8_people.yml  ← future YOLOv8 stub
│   ├── tracker/
│   │   ├── iou.yaml                   ← start here
│   │   ├── nvdcf_perf.yaml            ← recommended
│   │   └── nvdcf_accuracy.yaml
│   └── labels/
│       ├── trafficcamnet_labels.txt
│       └── people_only_labels.txt
├── engine_cache/                      ← TensorRT engines saved here (auto-created)
├── milestones/                        ← 8 standalone learning scripts
│   ├── 01_single_video_display.py
│   ├── 02_multi_video_tiled.py
│   ├── 03_people_detection.py         ← bounding boxes visible from here
│   ├── 04_people_tracking.py          ← tracking IDs visible from here
│   ├── 05_multi_stream_tracking.py    ← full N-stream pipeline
│   ├── 06_batching_deep_dive.py
│   ├── 07_metadata_extraction.py
│   └── 08_reid_stub.py
├── src/
│   ├── config/loader.py               ← PipelineConfig dataclass
│   ├── pipeline/
│   │   ├── builder.py                 ← full pipeline assembler (M5+)
│   │   ├── sources.py                 ← URI loaders (txt/folder/rtsp)
│   │   └── probes.py                  ← reusable probe classes
│   └── utils/platform_utils.py        ← sink type, GPU info
├── LEARNING_NOTES.md                  ← concept explanations (read this!)
├── README.md
├── requirements.txt
└── setup_venv.sh
```

---

## Troubleshooting

### Black screen / no video

- **Check:** `echo $DISPLAY` — must be set (e.g. `:0`)
- **Fix:** Run in a local desktop session, or `export DISPLAY=:0`

### `ModuleNotFoundError: No module named 'pyservicemaker'`

- **Fix:** `source venv/bin/activate` then verify `setup_venv.sh` ran successfully.

### Engine build takes a long time on first run

- **Normal!** First run builds a TensorRT engine (~1–3 min on RTX 3050Ti).
- Engine is saved to `engine_cache/` and reused on all subsequent runs (< 5s).

### `TypeError: object has no len()`

- **Cause:** `frame_meta.object_items` is an iterator, not a list.
- **Fix:** Use `sum(1 for _ in frame_meta.object_items)` to count.
- See `LEARNING_NOTES.md` § Iterator vs List.

### `Exception: Unsupported object type for Pipeline.attach`

- **Cause:** Passing a probe instance directly instead of wrapping in `psm.Probe`.
- **Wrong:** `pipeline.attach("tracker", MyProbe(), "name", {})`
- **Right:** `pipeline.attach("tracker", psm.Probe("name", MyProbe()))`

### `TypeError: attach(): incompatible function arguments`

- **Cause:** Passing a dict as 4th argument to `pipeline.attach` for built-in probes.
- **Wrong:** `pipeline.attach("pgie", "measure_fps_probe", "fps", {"interval": 5})`
- **Right:** `pipeline.attach("pgie", "measure_fps_probe", "fps")`

### Pipeline stuck at "Setting to PLAYING" / black window

- **Cause:** Live source or tee split without `async=0` on sink.
- **Fix:** Add `"async": 0` to sink properties.

### VRAM issues with many streams

```bash
# Lower tile resolution
python milestones/05_multi_stream_tracking.py --tile-w 640 --tile-h 360

# Skip inference frames (edit nvinfer config)
# interval: 2  → run inference every 3rd frame

# Ensure FP16 in nvinfer config
# network-mode: 2
```

---

## Key Paths (DeepStream 9.0)

```
TrafficCamNet ONNX: /opt/nvidia/deepstream/deepstream-9.0/samples/models/Primary_Detector/
Tracker library:    /opt/nvidia/deepstream/deepstream-9.0/lib/libnvds_nvmultiobjecttracker.so
Sample configs:     /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app/
pyservicemaker WHL: /opt/nvidia/deepstream/deepstream-9.0/service-maker/python/
```

> **Note:** DeepStream 9.0 installs under `deepstream-9.0/`, not `deepstream/`.
> All paths in nvinfer config files are **relative to the config file's directory**,
> not the working directory.
