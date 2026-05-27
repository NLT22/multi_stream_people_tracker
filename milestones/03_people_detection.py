"""
=============================================================================
MILESTONE 3 — People Detection with Bounding Boxes
=============================================================================

WHAT YOU LEARN:
  - nvinfer: runs TensorRT inference on each frame
  - nvosdbin: renders bounding boxes from nvinfer metadata onto the video
  - Why OSD comes AFTER nvinfer in the pipeline
  - TrafficCamNet class IDs (2 = Person)
  - FP16 precision for RTX 3050Ti

PIPELINE TOPOLOGY:
  [sources] → [mux] → [nvinfer] → [tiler] → [nvosdbin] → [sink]
                           ↑           ↑           ↑
                     detects objects  composites  draws boxes
                                     N streams    on tiled canvas

  CRITICAL ORDER: tiler BEFORE nvosdbin.
    The tiler composites all N streams into one canvas and scales each
    frame's metadata coordinates to the tile positions.
    nvosdbin then draws on the single tiled canvas using those coordinates.
    If OSD runs before tiler, all streams' boxes collapse onto one tile.

FIRST RUN — ENGINE BUILD:
  nvinfer builds a TensorRT engine from the ONNX on first run (~1 min).
  Saved to engine_cache/. Subsequent runs load it in seconds.

RUN:
  python milestones/03_people_detection.py
  python milestones/03_people_detection.py --sources configs/sources/video_files.txt

EXPECTED: Video with colored bounding boxes on every detected object.
  - Green/default boxes = nvosdbin's auto-drawn detection boxes
  - Labels show class name from labelfile (Vehicle, Bicycle, Person, RoadSign)

TODO EXERCISES:
  1. Change pre-cluster-threshold in nvinfer config from 0.2 → 0.5
     Fewer boxes appear. Change back to 0.1 to see all (noisy) detections.
  2. Change network-mode in the config: 0=FP32, 2=FP16. Compare VRAM usage
     with `nvidia-smi` in another terminal.
  3. Change `interval` in the config to 2 — inference runs every 3rd frame.
     Notice it becomes slightly choppy but faster.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def run(sources_txt: str, nvinfer_config: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    print(f"[M3] {n} source(s)  config={nvinfer_config}")
    print("[M3] First run builds TensorRT engine (~1 min). Subsequent runs are fast.")

    pipeline = psm.Pipeline("m3-detection")

    # MUXER
    pipeline.add("nvstreammux", "mux", {
        "batch-size": n, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": 0,
    })

    # SOURCES
    for i, uri in enumerate(uris):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": uri, "gpu-id": 0})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    # NVINFER — runs TensorRT, writes detection metadata into each frame
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": n, "gpu-id": 0,
    })
    # FPS probe: built-in, 3 string args only
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    # TILER — composites N streams into one grid canvas
    # Must come BEFORE nvosdbin so OSD draws on the composited frame.
    # The tiler scales each frame's metadata coordinates to tile positions.
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })

    # NVOSDBIN — draws boxes on the tiled canvas (after tiler scales coords)
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    # SINK
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    # LINK: mux → nvinfer → tiler → nvosdbin → sink
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[M3] Running — you should see bounding boxes on all detected objects.")
        print("[M3] FPS is printed to console periodically.")
        print("[M3] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M3] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3: People detection with OSD")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config)
