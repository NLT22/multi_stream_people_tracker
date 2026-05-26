"""
=============================================================================
MILESTONE 5 — Object Tracking with nvtracker
=============================================================================

WHAT YOU LEARN:
  - WHY tracking is separate from detection
  - How nvtracker assigns persistent object_id integers across frames
  - The difference between NvDCF and IoU tracker algorithms
  - How to switch tracker config files
  - Tracking quality: ID stability under occlusion

PIPELINE TOPOLOGY:
  [sources] → [mux] → [nvinfer/pgie] → [nvtracker] → [tiler] → [sink]

WHY TRACKING IS SEPARATE:
  nvinfer: "In frame 1, there's a person at (100,200). In frame 2, there's
            a person at (110,205)." — but nvinfer doesn't know they're the SAME person.
  nvtracker: associates detections across frames using motion prediction
             and visual appearance, then assigns a stable object_id.

  Without tracker: object_id is always 0 for all objects every frame.
  With tracker:    object_id=42 persists as long as that person is visible.

NvDCF vs IoU (start simple, then graduate):
  IoU tracker:   "If box1 and box2 overlap by >30%, same object."
                 Fast, CPU-only. IDs break on occlusion or fast movement.
                 → Start here to understand tracking concept.

  NvDCF tracker: Uses a correlation filter on visual patches (HOG + ColorNames).
                 Handles brief occlusion. More stable IDs. GPU-accelerated.
                 → Graduate to this after understanding IoU behavior.

RUN:
  source venv/bin/activate
  python milestones/05_object_tracking.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Try --tracker-config configs/tracker/iou.yaml — simplest tracker.
     Notice how IDs change when people cross or occlude each other.
  2. Switch to configs/tracker/nvdcf_perf.yaml — observe more stable IDs.
  3. Switch to configs/tracker/nvdcf_accuracy.yaml — compare stability vs speed.
  4. Cover a person's face in the video with your hand in another window.
     Does the ID survive? IoU fails; NvDCF may survive.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


class TrackingIDProbe(psm.BatchMetadataOperator):
    """
    Learning probe: observe how object_id changes over time.
    Compare IoU tracker (IDs break often) vs NvDCF (IDs stay stable).
    """

    PERSON_CLASS_ID = 2

    def __init__(self):
        super().__init__()
        self._active_ids: set[int] = set()
        self._frame_count = 0

    def execute(self, batch_meta):
        self._frame_count += 1
        if self._frame_count % 30 != 0:
            return

        for frame_meta in batch_meta.frame_items:
            current_ids = set()
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id == self.PERSON_CLASS_ID:
                    current_ids.add(obj_meta.object_id)

            # Detect new IDs (newly appeared or re-identified)
            new_ids = current_ids - self._active_ids
            lost_ids = self._active_ids - current_ids

            if new_ids or lost_ids or current_ids:
                print(
                    f"[M5] src={frame_meta.source_id}  "
                    f"tracking: {sorted(current_ids)}"
                    + (f"  +new={sorted(new_ids)}" if new_ids else "")
                    + (f"  -lost={sorted(lost_ids)}" if lost_ids else "")
                )
            self._active_ids = current_ids.copy()


def run(sources_txt: str, nvinfer_config: str, tracker_config: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)

    print(f"[M5] Sources        : {num_sources}")
    print(f"[M5] Tracker config : {tracker_config}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m5-tracking")

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
    # ll-lib-file:    The tracker algorithm library (NvDCF / IOU / DeepSORT)
    # ll-config-file: Algorithm-specific parameters (see configs/tracker/)
    # tracker-width/height: resolution at which tracking happens
    #   → Should match nvinfer input resolution for best accuracy
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width":  640,   # matches TrafficCamNet input width
        "tracker-height": 384,   # matches TrafficCamNet input height
        "gpu-id": 0,
    })

    # TRACKING ID PROBE — observe how IDs evolve over time
    pipeline.attach("tracker", TrackingIDProbe(), "tracking_probe", {})

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
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M5] Pipeline running. Tracking IDs print every 30 frames.")
        print("[M5] Watch how IDs change (IoU) vs stay stable (NvDCF).")
        print("[M5] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M5] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 5: Object tracking")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/iou.yaml",
                        help="Start with iou.yaml, then try nvdcf_perf.yaml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config)
