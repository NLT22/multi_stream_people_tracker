"""
=============================================================================
MILESTONE 4 — People Tracking with Labeled IDs
=============================================================================

WHAT YOU LEARN:
  - nvtracker: assigns persistent object_id integers across frames
  - The difference between detection (per-frame) and tracking (across frames)
  - How to write a probe that adds custom text labels above bounding boxes
  - IoU vs NvDCF tracker: algorithm tradeoffs
  - Why object_id=0 without nvtracker, and persistent ID with it

PIPELINE TOPOLOGY:
  [sources] → [mux] → [nvinfer] → [nvtracker] → [tiler]
                                                     │
                                            [PersonLabelProbe]
                                                     ↓
                                               [nvosdbin] → [sink]

  nvinfer:          detects persons, writes NvDsObjectMeta (class_id, rect)
  nvtracker:        links detections across frames, assigns persistent object_id
  tiler:            composites N streams → one canvas, SCALES metadata coords
  PersonLabelProbe: adds "Person #42" text using TILED coordinates
  nvosdbin:         draws boxes + our text on the tiled canvas

  WHY probe attaches to "tiler" (not "tracker"):
    After the tiler runs, metadata coordinates are in tiled canvas space.
    The probe draws text at those coordinates so labels appear at the right
    position on screen. Attaching before the tiler would use original
    1920×1080 coordinates, which are wrong after downscaling.

WITHOUT TRACKER: object_id is always 0. Every frame "forgets" who was who.
WITH TRACKER:    object_id=42 follows that person until they leave the frame.

TRACKER OPTIONS (change --tracker-config):
  iou.yaml           → simplest: match by box overlap only. IDs break on occlusion.
  nvdcf_perf.yaml    → GPU visual tracker. More stable IDs. Start here.
  nvdcf_accuracy.yaml → stronger features, slower, most stable IDs.

RUN:
  python milestones/04_people_tracking.py
  python milestones/04_people_tracking.py --tracker-config configs/tracker/iou.yaml

EXPECTED: Video with bounding boxes + green "Person #N" labels that persist
  across frames. IDs should NOT change unless the person leaves the scene.

TODO EXERCISES:
  1. Run with iou.yaml. Walk through the video. Watch IDs change when
     two people cross or one is briefly occluded.
  2. Switch to nvdcf_perf.yaml. Same scenario — IDs are more stable.
  3. Change label color to red for persons with ID > 5.
  4. Add confidence score to the label: "Person #42 (87%)"
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2  # TrafficCamNet: 0=Vehicle 1=Bicycle 2=Person 3=RoadSign


class PersonLabelProbe(psm.BatchMetadataOperator):
    """
    Adds "Person #<ID>" text above each tracked person's bounding box.

    Runs BEFORE nvosdbin so the text is included in the render pass.
    The tracking ID (object_id) comes from nvtracker.
    """

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:   # ITERATOR
            display_meta = psm.DisplayMeta(frame_meta)

            for obj_meta in frame_meta.object_items:  # ITERATOR
                if obj_meta.class_id != PERSON_CLASS_ID:
                    continue

                label = f"Person #{obj_meta.object_id}"
                # TODO Exercise 4: add confidence
                # label = f"Person #{obj_meta.object_id} ({obj_meta.confidence:.0%})"

                box = obj_meta.rect_params

                # TODO Exercise 3: color by ID range
                color = psm.Color(0.0, 1.0, 0.0, 1.0)  # green
                # color = psm.Color(1.0, 0.0, 0.0, 1.0) if obj_meta.object_id > 5 \
                #         else psm.Color(0.0, 1.0, 0.0, 1.0)

                display_meta.add_text(psm.Text(
                    label,
                    x=int(box.left),
                    y=max(0, int(box.top) - 22),
                    font=psm.Font(psm.FontFamily.Sans, 14),
                    color=color,
                ))


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def run(sources_txt: str, nvinfer_config: str, tracker_config: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    print(f"[M4] {n} source(s)  tracker={tracker_config}")

    pipeline = psm.Pipeline("m4-tracking")

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

    # NVINFER
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": n, "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    # NVTRACKER
    # ll-lib-file: shared library implementing the tracking algorithm
    # ll-config-file: algorithm parameters (see configs/tracker/)
    # tracker-width/height: resolution at which tracking runs
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream-9.0/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
    })

    # TILER — must come BEFORE probe and OSD
    # Composites all N streams into one canvas; scales metadata to tile coords.
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })

    # PERSON LABEL PROBE — attaches to tiler output (tiled canvas coordinates)
    pipeline.attach("tiler", psm.Probe("label_probe", PersonLabelProbe()))

    # NVOSDBIN — draws boxes + custom text on the tiled canvas
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    # SINK
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    # LINK: mux → nvinfer → nvtracker → tiler → osd → sink
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[M4] Running — you should see 'Person #N' labels on tracked people.")
        print("[M4] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M4] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 4: People tracking with IDs")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml",
                        help="Try iou.yaml first, then nvdcf_perf.yaml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config)
