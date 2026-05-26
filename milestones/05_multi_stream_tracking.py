"""
=============================================================================
MILESTONE 5 — Full Multi-Stream People Tracking Pipeline
=============================================================================

WHAT YOU LEARN:
  - Scaling the full pipeline (M3+M4) to N streams automatically
  - Dynamic tile grid computation from source count
  - How source_id distinguishes streams in metadata
  - Per-camera person count in console
  - Memory constraints on RTX 3050Ti at scale

PIPELINE TOPOLOGY (COMPLETE):
  [src_0] ──┐
  [src_1] ──┼──→ [mux] → [nvinfer] → [nvtracker]
  [src_N] ──┘                              │
                                  [PersonLabelProbe]
                                           ↓
                                      [nvosdbin]
                                           ↓
                               [nvmultistreamtiler]  ← NxN grid
                                           ↓
                                       [sink]

MEMORY NOTE FOR RTX 3050Ti (4GB VRAM):
  Wildtrack 7 cams (1920×1080) + 4 warehouse cams (1920×1080) = 11 streams
  At FP16 with TrafficCamNet:
    Model weights: ~60 MB
    Buffers (11×1080p NV12): ~330 MB
    Tracker: ~100 MB
    Total: ~490 MB — well within 4 GB
  If VRAM issues: reduce tile_h/tile_w or set interval=2 in nvinfer config.

RUN:
  python milestones/05_multi_stream_tracking.py
  python milestones/05_multi_stream_tracking.py --tile-w 640 --tile-h 360

TODO EXERCISES:
  1. Add --tile-w 640 --tile-h 360 → lower res, more streams fit on screen.
  2. Watch which camera has the most people simultaneously.
  3. Note how tracking IDs are PER-STREAM (Person #1 in cam0 ≠ Person #1 in cam1).
     Cross-camera ReID is Milestone 8.
  4. Open nvidia-smi in another terminal: watch VRAM usage scale with stream count.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2


class PersonLabelProbe(psm.BatchMetadataOperator):
    """
    Labels each tracked person with "Cam N | Person #ID" text.
    Also prints per-camera person count to console every 60 frames.
    """

    def __init__(self):
        super().__init__()
        self._frame_count = 0

    def handle_metadata(self, batch_meta):
        self._frame_count += 1
        log = self._frame_count % 60 == 0

        for frame_meta in batch_meta.frame_items:
            display_meta = psm.DisplayMeta(frame_meta)
            src = frame_meta.source_id
            person_count = 0

            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != PERSON_CLASS_ID:
                    continue
                person_count += 1

                box = obj_meta.rect_params
                # Show camera ID + tracking ID so you can tell streams apart
                label = f"Cam{src} #{obj_meta.object_id}"

                display_meta.add_text(psm.Text(
                    label,
                    x=int(box.left),
                    y=max(0, int(box.top) - 20),
                    font=psm.Font(psm.FontFamily.Sans, 12),
                    color=psm.Color(0.2, 1.0, 0.2, 1.0),
                ))

            if log and person_count > 0:
                print(f"  Cam{src:02d}: {person_count} person(s)")


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def run(sources_txt: str, nvinfer_config: str, tracker_config: str,
        tile_w: int, tile_h: int):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    total_w, total_h = tile_w * cols, tile_h * rows
    print(f"[M5] {n} stream(s) → {rows}×{cols} grid  canvas={total_w}×{total_h}")
    print(f"[M5] tracker={tracker_config}")

    pipeline = psm.Pipeline("m5-multi-stream")

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
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream-9.0/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384,
        "gpu-id": 0,
    })

    # LABEL PROBE → OSD
    pipeline.attach("tracker", psm.Probe("label_probe", PersonLabelProbe()))
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    # TILER
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": total_w, "height": total_h, "gpu-id": 0,
    })

    # SINK
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    # LINK
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

    try:
        pipeline.start()
        print(f"[M5] Running {n} streams in {rows}×{cols} grid.")
        print("[M5] Per-camera person counts print every 60 frames.")
        print("[M5] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M5] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 5: Full multi-stream people tracking")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    parser.add_argument("--tile-w", type=int, default=1280)
    parser.add_argument("--tile-h", type=int, default=720)
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h)
