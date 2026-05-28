"""
=============================================================================
MILESTONE 6 — Batching Deep Dive (nvstreammux Internals)
=============================================================================

WHAT YOU LEARN:
  - What "batch buffers" are at the data level
  - How batched-push-timeout controls partial batch behavior
  - The difference between file source and live-source mode
  - How to verify batch composition with a metadata probe
  - This milestone has full visual output (M5 pipeline) so you can
    watch the effect of batch settings while understanding them

BATCHING CONCEPT:
  nvstreammux gathers one frame from each source and packages them
  into a single "batch buffer" carrying NvDsBatchMeta.
  nvinfer processes the ENTIRE batch in one GPU call — this is how
  DeepStream achieves high throughput across multiple cameras.

  Timeline for 3 sources (file mode, timeout=40ms):
    t=0ms:   frame from source_0 arrives → wait for others
    t=5ms:   frame from source_1 arrives → still waiting
    t=40ms:  timeout → push batch with only source_0 + source_1
    t=41ms:  source_2 frame arrives → goes into NEXT batch

  → Lower timeout: smaller batches, lower latency, more GPU calls
  → Higher timeout: fuller batches, better GPU utilization

TUNABLE CONSTANTS BELOW — change them and observe behavior.

RUN:
  python milestones/06_batching_deep_dive.py

TODO EXERCISES:
  1. Set TIMEOUT = 1000 (1ms) → small batches, watch FPS drop.
  2. Set TIMEOUT = 1000000 (1s) → large batches, visible frame stutter.
  3. Set IS_LIVE = True → muxer uses NTP sync instead of timeout.
  4. Change PRINT_EVERY to 5 → see batch info more frequently.
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
from src.utils.platform_utils import get_sink_element, get_sink_properties


# ── Change these to experiment ──────────────────────────────────────────────
TIMEOUT  = 40000  # batched-push-timeout in µs (40ms = good for files)
IS_LIVE  = False  # True = NTP sync mode (needed for RTSP)
PRINT_EVERY = 30  # print batch info every N batches


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tracker' output (pre-tiler) where source_id is still valid.
    Collects pre-tiler batch stats while source_id is still explicit.

    After nvmultistreamtiler source_id resets to 0 — must read it here.
    """

    def __init__(self, id_map: dict, batch_stats: dict, person_class_id: int):
        super().__init__()
        self._id_map = id_map
        self._batch_stats = batch_stats  # shared: batch_n → (num_frames, src_ids)
        self._person_class_id = person_class_id
        self._n = 0

    def handle_metadata(self, batch_meta):
        self._n += 1
        log = self._n % PRINT_EVERY == 0

        num_frames = 0
        src_ids = []

        for frame_meta in batch_meta.frame_items:
            num_frames += 1
            src = frame_meta.source_id
            if log:
                src_ids.append(src)
            for obj in frame_meta.object_items:
                if obj.class_id == self._person_class_id:
                    self._id_map[obj.object_id] = src

        if log:
            self._batch_stats[self._n] = (num_frames, src_ids)


class BatchInspectorProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tiler' output (post-tiler) — tiled canvas coordinates.
    Draws Person labels and prints batch stats collected by SourceIdCollectorProbe.
    """

    def __init__(self, id_map: dict, batch_stats: dict, person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int):
        super().__init__()
        self._id_map = id_map
        self._batch_stats = batch_stats
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        self._n = 0

    def handle_metadata(self, batch_meta):
        self._n += 1

        # Print stats collected pre-tiler (source_ids are correct there)
        if self._n in self._batch_stats:
            num_frames, src_ids = self._batch_stats.pop(self._n)
            print(f"[M6] batch #{self._n:06d}: {num_frames} frame(s)  sources={src_ids}")

        for frame_meta in batch_meta.frame_items:
            for obj in frame_meta.object_items:
                if obj.class_id != self._person_class_id:
                    continue
                b = obj.rect_params
                src = infer_source_id_from_tiled_box(
                    b, self._tile_w, self._tile_h, self._cols,
                    self._num_sources)
                set_object_label(obj, f"Cam:{src}|Person:{obj.object_id}|Conf:{obj.confidence:.0%}")


def compute_grid(n):
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
    person_class_id = infer_person_class_id(nvinfer_config)
    print(f"[M6] {n} source(s)  timeout={TIMEOUT}µs  live={IS_LIVE}")
    print(f"[M6] person_class_id={person_class_id} inferred from {nvinfer_config}")

    pipeline = psm.Pipeline("m6-batching")

    mux_props = {
        "batch-size": n, "batched-push-timeout": TIMEOUT,
        "width": 1920, "height": 1080, "gpu-id": 0,
    }
    if IS_LIVE:
        mux_props["live-source"] = 1
    pipeline.add("nvstreammux", "mux", mux_props)

    for i, uri in enumerate(uris):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": uri, "gpu-id": 0})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": n, "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
    })

    # Shared state between the two probes
    id_map: dict[int, int] = {}
    batch_stats: dict[int, tuple] = {}  # batch_n → (num_frames, src_ids)

    # Probe 1 on tracker (pre-tiler): source_id still valid here
    pipeline.attach("tracker", psm.Probe(
        "src_collector", SourceIdCollectorProbe(id_map, batch_stats, person_class_id)))

    # TILER first — composites streams, scales metadata to tile coords
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })

    # Probe 2 on tiler (post-tiler): tiled canvas coordinates for drawing
    pipeline.attach("tiler", psm.Probe(
        "batch_probe",
        BatchInspectorProbe(
            id_map, batch_stats, person_class_id, 1280, 720, cols, n)))
    pipeline.add("nvosdbin", "osd", {
        "gpu-id": 0,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size": 18,
    })

    sink_props = get_sink_properties(is_live=IS_LIVE)
    pipeline.add(get_sink_element(), "sink", sink_props)

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print(f"[M6] Running. Batch info prints every {PRINT_EVERY} batches.")
        print("[M6] Try changing TIMEOUT and IS_LIVE at the top of this file.")
        print("[M6] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M6] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 6: Batching deep dive")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_yolov8_people.yml",
                        help="nvinfer config. Default: YOLOv8. "
                             "Alternatives: configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config)
