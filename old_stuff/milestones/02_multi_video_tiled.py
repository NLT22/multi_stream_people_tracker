"""
=============================================================================
MILESTONE 2 — Multi-Video Input with Tiled Display
=============================================================================

WHAT YOU LEARN:
  - How to add N sources to one pipeline
  - Why batch-size MUST equal the number of sources
  - nvmultistreamtiler: arranges N streams in an NxN grid
  - Dynamic grid sizing: ceil(sqrt(N)) × ceil(N/cols)
  - Loading source URIs from a text config file

PIPELINE TOPOLOGY:
  [nvurisrcbin_0] ──┐
  [nvurisrcbin_1] ──┼──→ [nvstreammux] → [nvmultistreamtiler] → [sink]
  [nvurisrcbin_N] ──┘

  Without nvmultistreamtiler: the sink only shows the LAST stream.
  With nvmultistreamtiler: all N streams appear in a grid side-by-side.

GRID FORMULA:
  cols = ceil(sqrt(N))   →  N=4 → 2×2,  N=7 → 3×3,  N=11 → 4×3

RUN:
  python milestones/02_multi_video_tiled.py
  python milestones/02_multi_video_tiled.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Add/remove video paths in video_files.txt — watch the grid resize.
  2. Change tile_w/tile_h to 640×360 — fits more streams, lower resolution.
  3. Uncomment show-source to focus one stream full-screen.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


def compute_grid(n: int) -> tuple[int, int]:
    """Return (rows, cols) for an NxN-ish grid. E.g. n=7 → (3,3)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def run(sources_txt: str, tile_w: int = 1280, tile_h: int = 720):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    print(f"[M2] {n} source(s) → {rows}×{cols} grid  ({tile_w}×{tile_h} per tile)")

    pipeline = psm.Pipeline("m2-multi-video")

    # MUXER — batch-size must equal number of sources
    pipeline.add("nvstreammux", "mux", {
        "batch-size": n,
        "width": 1920, "height": 1080,
        "batched-push-timeout": 40000,
        "gpu-id": 0,
    })

    # SOURCES — one nvurisrcbin per URI, linked with "sink_%u" request pad
    for i, uri in enumerate(uris):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": uri, "gpu-id": 0})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    # TILER — NxN grid layout
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": tile_w * cols, "height": tile_h * rows,
        # TODO Exercise 3: uncomment to focus stream 0
        # "show-source": 0,
        "gpu-id": 0,
    })

    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    pipeline.link("mux", "tiler")
    pipeline.link("tiler", "sink")

    try:
        pipeline.start()
        print("[M2] Showing tiled grid. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M2] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2: Multi-video tiled display")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--tile-w", type=int, default=1280)
    parser.add_argument("--tile-h", type=int, default=720)
    args = parser.parse_args()
    run(args.sources, args.tile_w, args.tile_h)
