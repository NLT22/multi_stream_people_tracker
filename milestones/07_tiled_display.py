"""
=============================================================================
MILESTONE 7 — NxN Tiled Display
=============================================================================

WHAT YOU LEARN:
  - How nvmultistreamtiler creates the NxN grid layout
  - How grid dimensions are computed dynamically from source count
  - The difference between per-tile resolution and total canvas resolution
  - How show-source focuses one stream
  - This is the COMPLETE pipeline (Milestones 1-6 combined)

PIPELINE TOPOLOGY (COMPLETE):
  [nvurisrcbin_0] ──┐
  [nvurisrcbin_1] ──┼──→ [nvstreammux] → [nvinfer] → [nvtracker]
  [nvurisrcbin_N] ──┘                                      │
                                                  [PersonOSDProbe]
                                                            ↓
                                                       [nvosdbin]
                                                            ↓
                                               [nvmultistreamtiler]  ← NxN grid
                                                            ↓
                                                    [nveglglessink]

TILE GRID FORMULA:
  cols = ceil(sqrt(N))       e.g. N=6 → cols=3
  rows = ceil(N / cols)      e.g. N=6 → rows=2
  Result: 2×3 grid for 6 streams (some cells empty for non-perfect squares)

TOTAL CANVAS SIZE:
  total_width  = tile_width  × cols
  total_height = tile_height × rows
  For 4 streams at 1280×720: total = 2560×1440

MEMORY NOTE FOR RTX 3050Ti (4GB VRAM):
  Each stream buffer at 1280×720 NV12 ≈ 1.4MB on GPU.
  4 streams × FP16 inference + tiler ≈ stays under 4GB.
  If you go to 8 streams, reduce tile resolution to 640×360.

RUN:
  source venv/bin/activate
  python milestones/07_tiled_display.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Add a 5th video to video_files.txt → should switch to 2×3 grid (one empty).
  2. Reduce tile_width to 640 and tile_height to 360 — more streams fit on screen.
  3. Uncomment show-source=0 in the tiler to see stream 0 focused full-screen.
  4. Change the OSD label to show "Cam 0 | Person #42" using frame_meta.source_id.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2


class PersonOSDProbe(psm.BatchMetadataOperator):
    """OSD probe: show "Cam N | Person #ID" for each tracked person."""

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            display_meta = psm.DisplayMeta(frame_meta)
            source_id = frame_meta.source_id

            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != PERSON_CLASS_ID:
                    continue

                # TODO Exercise 4: show camera ID in label
                label = f"Person #{obj_meta.object_id}"
                # label = f"Cam {source_id} | Person #{obj_meta.object_id}"

                box = obj_meta.rect_params
                display_meta.add_text(
                    psm.Text(
                        label,
                        x=int(box.left),
                        y=max(0, int(box.top) - 22),
                        font=psm.Font(psm.FontFamily.Sans, 14),
                        color=psm.Color(0.0, 1.0, 0.0, 1.0),
                    )
                )


def compute_tile_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def run(sources_txt: str, nvinfer_config: str, tracker_config: str,
        tile_w: int, tile_h: int):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    rows, cols = compute_tile_grid(num_sources)
    total_w = tile_w * cols
    total_h = tile_h * rows

    print(f"[M7] Sources     : {num_sources}")
    print(f"[M7] Grid        : {rows}×{cols} (total canvas {total_w}×{total_h})")
    print(f"[M7] Tile size   : {tile_w}×{tile_h} per stream")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m7-tiled")

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

    # OSD PROBE (before nvosdbin) — custom probes must be wrapped
    pipeline.attach("tracker", psm.Probe("osd_probe", PersonOSDProbe()))

    # NVOSDBIN
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    # TILER — NxN grid layout
    tiler_props = {
        "rows":    rows,
        "columns": cols,
        "width":   total_w,
        "height":  total_h,
        "gpu-id":  0,
        # TODO Exercise 3: uncomment to focus stream 0
        # "show-source": 0,
    }
    pipeline.add("nvmultistreamtiler", "tiler", tiler_props)

    # SINK
    # For a large tiled canvas, sync=0 avoids frame drops when rendering is slow
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    # ── Link ─────────────────────────────────────────────────────────────────
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print(f"[M7] Pipeline running. Showing {rows}×{cols} grid.")
        print("[M7] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M7] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 7: NxN tiled display")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    parser.add_argument("--tile-width",  type=int, default=1280)
    parser.add_argument("--tile-height", type=int, default=720)
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_width, args.tile_height)
