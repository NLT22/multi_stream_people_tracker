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

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element, get_sink_properties


# ── Change these to experiment ──────────────────────────────────────────────
TIMEOUT  = 40000  # batched-push-timeout in µs (40ms = good for files)
IS_LIVE  = False  # True = NTP sync mode (needed for RTSP)
PRINT_EVERY = 30  # print batch info every N batches


PERSON_CLASS_ID = 2


class BatchInspectorProbe(psm.BatchMetadataOperator):
    """Logs batch composition + draws Person labels for visual context."""

    def __init__(self):
        super().__init__()
        self._n = 0

    def handle_metadata(self, batch_meta):
        self._n += 1

        # Count frames in this batch — use iterator, NOT len()
        frames = list(batch_meta.frame_items)
        num_frames = len(frames)

        if self._n % PRINT_EVERY == 0:
            src_ids = [f.source_id for f in frames]
            print(f"[M6] batch #{self._n:06d}: {num_frames} frame(s)  sources={src_ids}")

        # Also draw labels so we can see tracking visually while learning
        for frame_meta in frames:
            dm = psm.DisplayMeta(frame_meta)
            for obj in frame_meta.object_items:
                if obj.class_id != PERSON_CLASS_ID:
                    continue
                b = obj.rect_params
                dm.add_text(psm.Text(
                    f"#{obj.object_id}",
                    x=int(b.left), y=max(0, int(b.top) - 18),
                    font=psm.Font(psm.FontFamily.Sans, 12),
                    color=psm.Color(0.0, 1.0, 0.5, 1.0),
                ))


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
    print(f"[M6] {n} source(s)  timeout={TIMEOUT}µs  live={IS_LIVE}")

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
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream-9.0/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
    })

    # Batch inspector probe attaches to tracker output
    pipeline.attach("tracker", psm.Probe("batch_probe", BatchInspectorProbe()))
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })

    sink_props = get_sink_properties(is_live=IS_LIVE)
    pipeline.add(get_sink_element(), "sink", sink_props)

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

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
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config)
