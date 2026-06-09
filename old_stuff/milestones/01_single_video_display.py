"""
=============================================================================
MILESTONE 1 — Single Video Display
=============================================================================

WHAT YOU LEARN:
  - The minimal DeepStream pipeline: source → mux → sink
  - Why nvurisrcbin is used instead of filesrc + qtdemux + h264parse
  - Why nvstreammux is required even for a single stream
  - How to convert a file path to a file:// URI
  - The "sink_%u" pad template (GStreamer request pads)
  - Platform-specific sinks: nveglglessink vs nv3dsink

PIPELINE TOPOLOGY:
  [nvurisrcbin] → [nvstreammux] → [nveglglessink]

  nvurisrcbin:     opens any URI (file/RTSP/HTTP), handles codec detection,
                   decoding, and outputs raw NV12 frames in GPU memory (NVMM)

  nvstreammux:     REQUIRED — converts raw decoded buffers into "batch buffers"
                   with NvDsBatchMeta attached. All downstream elements
                   (nvinfer, nvtracker, OSD) expect this metadata structure.

  nveglglessink:   opens a display window and renders the video (x86_64)

WHY NVMM (memory:NVMM):
  After decoding, frames live on the GPU. nvstreammux, nvinfer, and OSD
  all work directly on GPU memory — zero CPU copies.
  This is the key performance advantage of the DeepStream pipeline.

RUN:
  source venv/bin/activate
  python milestones/01_single_video_display.py --input /path/to/video.mp4

EXPECTED RESULT:
  A window opens showing the video playing in real time.
  Close the window or press Ctrl+C to stop.

TODO EXERCISES:
  1. Change --input to a different video file and verify it plays.
  2. Try sync=0 (already in code) vs sync=1 — what changes?
  3. (Advanced) Try rtsp://... as --input without changing any code.
     nvurisrcbin handles it transparently.
=============================================================================
"""

import argparse
import os
import sys

import pyservicemaker as psm

from src.utils.platform_utils import get_sink_element


def path_to_uri(path: str) -> str:
    """Convert a local file path to a file:// URI for nvurisrcbin."""
    if "://" in path:
        return path  # already a URI
    return "file://" + os.path.abspath(path)


def run(video_path: str):
    uri = path_to_uri(video_path)
    print(f"[M1] Source URI : {uri}")
    print(f"[M1] Sink type  : {get_sink_element()}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m1-single-video")

    # SOURCE: nvurisrcbin
    # Handles any URI format. Internally it creates:
    #   uridecodebin → nvv4l2decoder → outputs NVMM NV12 frames
    # You do NOT need filesrc, qtdemux, h264parse, or nvv4l2decoder yourself.
    pipeline.add("nvurisrcbin", "source", {
        "uri": uri,
        "gpu-id": 0,
    })

    # MUXER: nvstreammux
    # Even for 1 stream, the muxer is needed because ALL downstream elements
    # (nvinfer, OSD, tracker) work with NvDsBatchMeta which is attached here.
    #
    # batch-size: must equal number of input streams (1 here)
    # width/height: output resolution for downstream elements
    # batched-push-timeout: how long to wait for a full batch (µs)
    #   → For files: 40000 µs works well
    #   → For RTSP: set live-source=1 instead of tuning the timeout
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1,
        "width":  1920,
        "height": 1080,
        "batched-push-timeout": 40000, # 40ms
        "gpu-id": 0,
    })

    # SINK: display window
    # sync=1 → playback at original frame rate (smooth, file sources)
    # sync=0 → as fast as possible (good for benchmarking)
    #
    # TODO Exercise 2: Change sync=1 and notice the difference in playback speed
    # nveglglessink / nv3dsink supported window properties:
    #   window-x, window-y  → top-left corner position (pixels)
    #   window-width, window-height → initial window size
    # NOTE: "title" is NOT a supported property on these sinks — window title
    #       must be set via the OS/display manager after the window appears.
    pipeline.add(get_sink_element(), "sink", {
        "sync": 1,
        "window-width": 1280,
        "window-height": 720,
    })

    # ── Link elements ────────────────────────────────────────────────────────
    #
    # IMPORTANT: nvstreammux uses request pads (created on demand).
    # Syntax: pipeline.link((upstream, downstream), (src_pad, sink_pad))
    # "sink_%u" is the GStreamer template name — DO NOT use "sink_0" or "sink_1"!
    # The %u is replaced with an auto-incremented number by GStreamer.
    #
    pipeline.link(("source", "mux"), ("", "sink_%u"))   # ← request pad template
    pipeline.link("mux", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M1] Pipeline running. Close the window or press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M1] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 1: Single video display")
    parser.add_argument(
        "--input",
        required=True,
        help="Path or URI to video file (e.g. /home/user/video.mp4 or file:///...)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input) and "://" not in args.input:
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    run(args.input)
