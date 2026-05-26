"""
=============================================================================
MILESTONE 6 — OSD Visualization (Bounding Boxes + Labels)
=============================================================================

WHAT YOU LEARN:
  - How nvosdbin renders visual overlays on video
  - How to use DisplayMeta to add custom text/rectangles before OSD renders
  - The Text, Font, Color, FontFamily API from pyservicemaker
  - Why the probe attaches BEFORE nvosdbin (not after)
  - How to show tracking IDs as visible labels on screen

PIPELINE TOPOLOGY:
  [sources] → [mux] → [nvinfer] → [nvtracker]
                                       │
                              [PersonOSDProbe]  ← probe runs here
                                       ↓
                                  [nvosdbin]    ← renders DisplayMeta
                                       ↓
                                   [tiler]
                                       ↓
                                    [sink]       ← you see boxes + labels

WHAT nvosdbin DOES:
  1. Reads NvDsObjectMeta (from nvinfer) → draws default bounding boxes
  2. Reads NvDsDisplayMeta (from our probe) → draws our custom text on top
  3. Outputs the composited frame as NVMM buffer for the next element

WHY PROBE BEFORE OSD (not after):
  The probe adds text to DisplayMeta. nvosdbin READS DisplayMeta to render.
  If we attach the probe after OSD, the text is added too late to be drawn.

COLOR SYSTEM:
  psm.Color(r, g, b, a) — all values 0.0 to 1.0
  Green text: Color(0.0, 1.0, 0.0, 1.0)
  Red text:   Color(1.0, 0.0, 0.0, 1.0)
  Yellow:     Color(1.0, 1.0, 0.0, 1.0)
  Blue:       Color(0.0, 0.0, 1.0, 1.0)

RUN:
  source venv/bin/activate
  python milestones/06_osd_visualization.py --sources configs/sources/video_files.txt

EXPECTED RESULT:
  Video with green "Person #42" labels floating above each detected person.
  IDs persist across frames (from nvtracker in Milestone 5).

TODO EXERCISES:
  1. Change the label color to yellow for persons with ID > 10.
  2. Add confidence score to the label: "Person #42 (87%)"
  3. Change font size from 14 to 20 — is it more readable?
  4. Add a custom colored rectangle using display_meta.add_rect()
  5. Try hiding non-person detections (vehicles, bicycles) by skipping them.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2   # TrafficCamNet class_id for "Person"


class PersonOSDProbe(psm.BatchMetadataOperator):
    """
    Draw "Person #<tracking_id>" labels above each detected person.

    Runs BEFORE nvosdbin so the text gets included in the render pass.
    """

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:  # ITERATOR
            # DisplayMeta is attached to this frame for our custom overlays
            display_meta = psm.DisplayMeta(frame_meta)

            for obj_meta in frame_meta.object_items:  # ITERATOR
                if obj_meta.class_id != PERSON_CLASS_ID:
                    # TODO Exercise 5: decide what to do with non-person classes
                    # Option A: skip entirely (only show persons)
                    # Option B: show with a different color
                    continue

                # ── Label text ────────────────────────────────────────────
                label = f"Person #{obj_meta.object_id}"

                # TODO Exercise 2: Add confidence score
                # label = f"Person #{obj_meta.object_id} ({obj_meta.confidence:.0%})"

                # ── Position: above the top-left corner of the bounding box ──
                box = obj_meta.rect_params
                x = int(box.left)
                y = max(0, int(box.top) - 22)  # 22px above the box

                # ── Choose color ──────────────────────────────────────────
                color = psm.Color(0.0, 1.0, 0.0, 1.0)   # green

                # TODO Exercise 1: color by tracking ID
                # color = psm.Color(1.0, 1.0, 0.0, 1.0) if obj_meta.object_id > 10 \
                #         else psm.Color(0.0, 1.0, 0.0, 1.0)

                # ── Add text to DisplayMeta ───────────────────────────────
                display_meta.add_text(
                    psm.Text(
                        label,
                        x=x,
                        y=y,
                        font=psm.Font(psm.FontFamily.Sans, 14),
                        # TODO Exercise 3: change font size to 20
                        color=color,
                    )
                )

                # TODO Exercise 4: add a custom rectangle
                # display_meta.add_rect(psm.Rect(
                #     left=int(box.left), top=int(box.top),
                #     width=int(box.width), height=int(box.height),
                #     border_color=psm.Color(0.0, 1.0, 0.0, 1.0),
                #     border_width=2,
                # ))


def run(sources_txt: str, nvinfer_config: str, tracker_config: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)

    print(f"[M6] Sources: {num_sources}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m6-osd")

    # MUXER
    pipeline.add("nvstreammux", "mux", {
        "batch-size": num_sources,
        "batched-push-timeout": 40000,
        "width": 1920, "height": 1080,
        "gpu-id": 0,
    })

    # SOURCES
    for i, uri in enumerate(uris):
        src_name = f"source_{i}"
        pipeline.add("nvurisrcbin", src_name, {"uri": uri, "gpu-id": 0})
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

    # NVINFER
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": num_sources,
        "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe",
                    {"print-fps-interval": 5})

    # NVTRACKER
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384,
        "gpu-id": 0,
    })

    # OSD PROBE — must attach BEFORE nvosdbin
    # Custom probes must be wrapped: psm.Probe("name", instance)
    pipeline.attach("tracker", psm.Probe("osd_probe", PersonOSDProbe()))

    # NVOSDBIN — renders bounding boxes (from nvinfer) + custom text (from probe)
    # process-mode: 1 = GPU rendering (uses NVMM directly, fastest)
    # process-mode: 0 = CPU rendering (debug only, very slow)
    pipeline.add("nvosdbin", "osd", {
        "gpu-id": 0,
        "process-mode": 1,
    })

    # TILER + SINK
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows,
        "gpu-id": 0,
    })
    pipeline.add(get_sink_element(), "sink", {"sync": 1})

    # ── Link ─────────────────────────────────────────────────────────────────
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")   # osd_probe already attached to tracker
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M6] Pipeline running. You should see 'Person #N' labels on screen.")
        print("[M6] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M6] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 6: OSD visualization")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config)
