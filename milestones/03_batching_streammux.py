"""
=============================================================================
MILESTONE 3 — Batching with nvstreammux
=============================================================================

WHAT YOU LEARN:
  - What "batching" means in DeepStream context
  - How batched-push-timeout affects file vs live sources
  - The difference between file source and live source behavior
  - How batch buffers are structured (NvDsBatchMeta concept)
  - Why RTSP requires live-source=1 and sync=0 on sink

FOCUS: This milestone is about UNDERSTANDING nvstreammux deeply.
The visible output is the same as M2, but we add logging to show
batch behavior and explore different timeout settings.

BATCHING EXPLAINED:
  nvstreammux collects one frame from each active source into a single
  "batch buffer". This batch buffer is what nvinfer processes in one
  GPU inference call — that's why batch_size matters for performance.

  Timeline for 3 sources (file mode, batched-push-timeout=40ms):
    t=0ms:   frame from source_0 arrives
    t=5ms:   frame from source_1 arrives
    t=40ms:  timeout! source_2 hasn't arrived → push partial batch anyway
    t=41ms:  frame from source_2 arrives (will be in NEXT batch)

  For RTSP (live) sources:
    Use live-source=1 instead. The muxer uses timestamp synchronization
    rather than a fixed timeout.

RUN:
  source venv/bin/activate
  python milestones/03_batching_streammux.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Change BATCHED_PUSH_TIMEOUT to 1000000 (1 second) and observe frame drops.
  2. Change BATCHED_PUSH_TIMEOUT to 1000 (1ms) and observe dropped frames.
  3. Set IS_LIVE_SOURCE = True (even for file sources) — what happens?
  4. Add a metadata probe to count how many sources are in each batch.
=============================================================================
"""

import argparse
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element, get_sink_properties

# ── Tunable parameters for learning ──────────────────────────────────────────

# How long (microseconds) nvstreammux waits for a full batch before pushing.
# For file sources, all frames arrive at near-zero latency so 40000µs works.
# TODO Exercise 1: Change this value and observe behavior
BATCHED_PUSH_TIMEOUT = 40000  # 40ms — good for file sources

# Set True to simulate live-source behavior (normally only for RTSP).
# With True: muxer uses NTP timestamps; sink must use sync=0.
IS_LIVE_SOURCE = False


class BatchInspectorProbe(psm.BatchMetadataOperator):
    """
    Learning probe: inspect how many frames are in each batch buffer.

    This shows you the concrete effect of batching: each call to execute()
    corresponds to ONE batch buffer passing through the pipeline.
    The number of frame_items equals the number of active sources in that batch.
    """

    def __init__(self):
        super().__init__()
        self._batch_count = 0

    def handle_metadata(self, batch_meta):
        self._batch_count += 1

        # Count frames in this batch (can be < batch_size if partial batch)
        # NOTE: frame_items is an ITERATOR — we must iterate to count
        num_frames = sum(1 for _ in batch_meta.frame_items)

        # Print every 30 batches to avoid console spam
        if self._batch_count % 30 == 0:
            print(
                f"[M3] Batch #{self._batch_count:06d}: "
                f"{num_frames} frame(s) in this batch"
            )

        # TODO Exercise 4: Print source_id for each frame to see which
        #                  streams contributed to this batch:
        for frame_meta in batch_meta.frame_items:
            print(f"  source_id={frame_meta.source_id}")


def run(sources_txt: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    print(f"[M3] Sources              : {num_sources}")
    print(f"[M3] batched-push-timeout : {BATCHED_PUSH_TIMEOUT} µs")
    print(f"[M3] live-source mode     : {IS_LIVE_SOURCE}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m3-batching")

    # ── Muxer with explicit timeout control ──────────────────────────────────
    mux_props = {
        "batch-size": num_sources,
        "batched-push-timeout": BATCHED_PUSH_TIMEOUT,
        "width":  1280,
        "height": 720,
        "gpu-id": 0,
    }
    if IS_LIVE_SOURCE:
        # live-source=1 tells the muxer to use NTP timestamp sync
        # instead of the fixed timeout. Required for RTSP cameras.
        mux_props["live-source"] = 1

    pipeline.add("nvstreammux", "mux", mux_props)

    # ── Sources ───────────────────────────────────────────────────────────────
    for i, uri in enumerate(uris):
        src_name = f"source_{i}"
        pipeline.add("nvurisrcbin", src_name, {"uri": uri, "gpu-id": 0})
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

    # ── Batch inspector probe ─────────────────────────────────────────────────
    # Attach to mux (src pad) to inspect each batch as it leaves the muxer.
    # This is WHERE the NvDsBatchMeta is first available.
    # Custom probes must be wrapped: psm.Probe("name", instance)
    pipeline.attach("mux", psm.Probe("batch_inspector", BatchInspectorProbe()))

    # ── Tiler + Sink ─────────────────────────────────────────────────────────
    import math
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows,
        "gpu-id": 0,
    })

    sink_props = get_sink_properties(is_live=IS_LIVE_SOURCE)
    pipeline.add(get_sink_element(), "sink", sink_props)

    pipeline.link("mux", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M3] Pipeline running. Batch counts print every 30 batches.")
        print("[M3] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M3] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3: Batching with nvstreammux")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    args = parser.parse_args()
    run(args.sources)
