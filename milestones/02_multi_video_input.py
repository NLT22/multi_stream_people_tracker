"""
=============================================================================
MILESTONE 2 — Multi-Video Input
=============================================================================

WHAT YOU LEARN:
  - How to add N sources to one pipeline (N nvurisrcbins)
  - How batch-size must equal the number of sources
  - Why the tiler is needed to see all N streams (without it: only 1 visible)
  - Loading source URIs from a text config file
  - Dynamic source count: the pipeline adapts to however many lines are in the txt

PIPELINE TOPOLOGY:
  [nvurisrcbin_0] ──┐
  [nvurisrcbin_1] ──┼──→ [nvstreammux] → [nvmultistreamtiler] → [nveglglessink]
  [nvurisrcbin_N] ──┘

  nvmultistreamtiler: arranges N streams in a grid (e.g. 4 streams → 2×2).
                      Without it, the sink only shows the LAST stream.

RUN:
  # First edit configs/sources/video_files.txt to have 2+ video paths
  source venv/bin/activate
  python milestones/02_multi_video_input.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Add a second video to video_files.txt → grid should expand to 1×2.
  2. Add 3 videos → should become 2×2 (one empty cell).
  3. Add 4 videos → 2×2 grid with all cells filled.
  4. What happens with 9 videos? Grid = 3×3.
  5. Comment out all but 1 video → back to single stream.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


def compute_tile_grid(n: int) -> tuple[int, int]:
    """
    Return (rows, cols) for an NxN-ish grid fitting n streams.

    This is the same formula used in PipelineBuilder and Milestone 7.
    Having it here helps you understand the logic before it moves to the lib.

    n=1 → (1,1)  n=2 → (1,2)  n=3 → (2,2)  n=4 → (2,2)  n=9 → (3,3)
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def run(sources_txt: str):
    # ── Load URIs from text config ───────────────────────────────────────────
    # TODO: load_uris_from_txt is already implemented in src/pipeline/sources.py
    #       Open that file to understand how it works.
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    rows, cols = compute_tile_grid(num_sources)

    print(f"[M2] Sources    : {num_sources}")
    print(f"[M2] Tile grid  : {rows}×{cols}")
    for i, uri in enumerate(uris):
        print(f"[M2]   [{i}] {uri}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m2-multi-video")

    # MUXER: batch-size MUST equal the number of source streams
    # If batch-size < num_sources: some streams are dropped
    # If batch-size > num_sources: DeepStream waits forever for missing streams
    pipeline.add("nvstreammux", "mux", {
        "batch-size": num_sources,  # ← dynamic, based on loaded URIs
        "width":  1280,
        "height": 720,
        "batched-push-timeout": 40000,
        "gpu-id": 0,
    })

    # SOURCES: one nvurisrcbin per URI
    for i, uri in enumerate(uris):
        src_name = f"source_{i}"
        pipeline.add("nvurisrcbin", src_name, {
            "uri": uri,
            "gpu-id": 0,
        })
        # "sink_%u" — request pad template: auto-assigns sink_0, sink_1, sink_2, ...
        # NEVER hardcode "sink_0" even for the first source
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

    # TILER: arrange N streams in a grid
    # Without this element, the sink only renders the last stream in the batch.
    # width/height here is the TOTAL canvas size (all tiles combined).
    tile_w = 1280
    tile_h = 720
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows":    rows,
        "columns": cols,
        "width":   tile_w * cols,
        "height":  tile_h * rows,
        "gpu-id":  0,
    })

    # SINK
    pipeline.add(get_sink_element(), "sink", {"sync": 1})

    # ── Link ─────────────────────────────────────────────────────────────────
    pipeline.link("mux", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M2] Pipeline running. Close the window or press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M2] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2: Multi-video input")
    parser.add_argument(
        "--sources",
        default="configs/sources/video_files.txt",
        help="Path to sources text file (default: configs/sources/video_files.txt)",
    )
    args = parser.parse_args()
    run(args.sources)
