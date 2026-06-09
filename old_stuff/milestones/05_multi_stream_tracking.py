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
  [src_1] ──┼──→ [mux] → [nvinfer] → [nvtracker] ──→ [tiler]
  [src_N] ──┘                              │               │
                                [SourceIdCollectorProbe]  [PersonLabelProbe]
                                  (fills id_map dict)    (draws labels)
                                                               ↓
                                                          [nvosdbin]
                                                               ↓
                                                           [sink]

  WHY TWO PROBES:
    After nvmultistreamtiler all N streams become ONE composited frame and
    source_id resets to 0. We MUST read source_id on tracker output (pre-tiler)
    and pass it via a shared dict to the label probe (post-tiler) which has
    the correct tiled-canvas coordinates for drawing text.

MEMORY NOTE FOR RTX 3050Ti (4GB VRAM):
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

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
    infer_source_id_from_tiled_box,
    set_object_label,
)
from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tracker' output (pre-tiler) where source_id is still valid.
    Legacy pre-tiler source collector kept for experiments.

    Why two probes:
      After nvmultistreamtiler all frames are composited into one canvas and
      source_id becomes 0 for every frame.  We must read source_id HERE
      (before tiler) and pass it via a shared dict to the label probe (after
      tiler) which has the correct tiled-canvas coordinates for drawing.
    """

    def __init__(self, id_map: dict, person_class_id: int):
        super().__init__()
        self._id_map = id_map  # shared with PersonLabelProbe
        self._person_class_id = person_class_id

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id == self._person_class_id:
                    self._id_map[obj_meta.object_id] = src


class PersonLabelProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tiler' output (post-tiler) where coordinates are correct.
    Labels each tracked person with "CamN #ID" text.
    Camera id is inferred from the tiled bbox position to avoid object_id
    collisions across streams.
    Also prints per-camera person count to console every 60 frames.
    """

    def __init__(self, person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int):
        super().__init__()
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        self._frame_count = 0

    def handle_metadata(self, batch_meta):
        self._frame_count += 1
        log = self._frame_count % 60 == 0

        cam_counts: dict[int, int] = {}

        for frame_meta in batch_meta.frame_items:
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue

                src = infer_source_id_from_tiled_box(
                    obj_meta.rect_params, self._tile_w, self._tile_h,
                    self._cols, self._num_sources)
                cam_counts[src] = cam_counts.get(src, 0) + 1

                label = f"Cam:{src}|Person:{obj_meta.object_id}"
                set_object_label(obj_meta, label)

        if log:
            for src, count in sorted(cam_counts.items()):
                print(f"  Cam{src:02d}: {count} person(s)")


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
    person_class_id = infer_person_class_id(nvinfer_config)
    total_w, total_h = tile_w * cols, tile_h * rows
    print(f"[M5] {n} stream(s) → {rows}×{cols} grid  canvas={total_w}×{total_h}")
    print(f"[M5] tracker={tracker_config}")
    print(f"[M5] person_class_id={person_class_id} inferred from {nvinfer_config}")

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
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384,
        "gpu-id": 0,
    })

    # TILER — composites N streams, scales metadata to tile canvas coords
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": total_w, "height": total_h, "gpu-id": 0,
    })

    # LABEL PROBE — attaches AFTER tiler (reads tiled canvas coordinates)
    pipeline.attach("tiler", psm.Probe(
        "label_probe",
        PersonLabelProbe(person_class_id, tile_w, tile_h, cols, n)))

    # OSD — draws on the tiled canvas
    pipeline.add("nvosdbin", "osd", {
        "gpu-id": 0,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size": 18,
    })

    # SINK
    pipeline.add(get_sink_element(), "sink", {"sync": 1, "qos": 0})

    # LINK: mux → nvinfer → tracker → tiler → osd → sink
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

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
                        default="configs/models/nvinfer_yolov8_people.yml",
                        help="nvinfer config. Default: YOLOv8. "
                             "Alternatives: configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml",
                        help="nvdcf_accuracy.yaml for better ID stability (tuned for people tracking)")
    parser.add_argument("--tile-w", type=int, default=1280)
    parser.add_argument("--tile-h", type=int, default=720)
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h)
