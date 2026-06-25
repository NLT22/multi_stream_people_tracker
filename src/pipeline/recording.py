"""
Tiling grid + annotated-video recording branch helpers.
"""

import math
from pathlib import Path

import pyservicemaker as psm


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


# H.264 encoders cap out around 4096 px per side (and choke well before that on
# a single GPU). A multi-camera tiled canvas easily exceeds this — e.g. 12 cams
# at 1280x720 tiles = 5120x2160 — which makes nvv4l2h264enc stall and freeze the
# whole pipeline. Downscale the recording to a safe long side before encoding.
RECORD_MAX_SIDE = 1920
# HLS can go larger than the MP4 recording: a 20-cam 5×4 mosaic of 640×360 cells
# is 3200×1440 — under the encoder limit — so each browser tile stays sharp
# (~640px/cell) instead of the ~384px you get when squeezed to 1920 wide.
HLS_MAX_SIDE = 3200


def _bounded_even(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale (width, height) so the longest side <= max_side, keep even dims."""
    longest = max(width, height)
    if longest <= max_side:
        scale = 1.0
    else:
        scale = max_side / longest
    w = max(2, int(round(width * scale)) & ~1)   # force even (H.264 requirement)
    h = max(2, int(round(height * scale)) & ~1)
    return w, h


def add_hls_branch(pipeline: psm.Pipeline, upstream: str,
                   hls_dir: str, bitrate: int,
                   canvas_w: int = 0, canvas_h: int = 0,
                   gpu_id: int = 0, target_duration: int = 2,
                   window: int = 8) -> str:
    """Encode the annotated (OSD) canvas to a live HLS stream a browser can play.

    Writes <hls_dir>/stream.m3u8 + rolling segNNNNN.ts. A sliding playlist
    (max-files/playlist-length) keeps disk bounded for an indefinite live run.
    Returns the playlist path. Mirrors add_recording_branch's encode chain so
    the same H.264 settings / downscale guard apply.
    """
    out = Path(hls_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline.add("queue", "hls_queue", {"leaky": 2, "max-size-buffers": 5})
    pipeline.add("nvvideoconvert", "hls_convert", {"gpu-id": gpu_id})

    enc_w, enc_h = (_bounded_even(canvas_w, canvas_h, HLS_MAX_SIDE)
                    if canvas_w and canvas_h else (0, 0))
    if enc_w and enc_h and (enc_w, enc_h) != (canvas_w, canvas_h):
        print(f"[reid] HLS downscaled {canvas_w}x{canvas_h} -> {enc_w}x{enc_h}")
        pipeline.add("capsfilter", "hls_caps", {
            "caps": f"video/x-raw(memory:NVMM), width={enc_w}, height={enc_h}",
        })
        scale_chain = ["hls_convert", "hls_caps"]
    else:
        scale_chain = ["hls_convert"]

    pipeline.add("nvv4l2h264enc", "hls_encoder", {
        "bitrate": bitrate, "insert-sps-pps": 1, "idrinterval": 15,
    })
    pipeline.add("h264parse", "hls_h264parse")
    pipeline.add("hlssink2", "hls_sink", {
        "location": str(out / "seg%05d.ts"),
        "playlist-location": str(out / "stream.m3u8"),
        "target-duration": target_duration,
        "max-files": window + 4,
        "playlist-length": window,
        # sync=0: write segments as frames arrive. On a live pipeline a sync=1
        # sink gates on the clock and can deadlock preroll → 0 fps. The RTSP
        # publisher's ffmpeg -re already paces input to real time.
        "sync": 0,
    })
    pipeline.link(upstream, "hls_queue", *scale_chain, "hls_encoder", "hls_h264parse")
    # hlssink2's video pad is request-only.
    pipeline.link(("hls_h264parse", "hls_sink"), ("", "video"))
    return str(out / "stream.m3u8")


def add_recording_branch(pipeline: psm.Pipeline, upstream: str,
                         output_path: str, bitrate: int,
                         canvas_w: int = 0, canvas_h: int = 0,
                         gpu_id: int = 0) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # leaky=2 (drop oldest) so a slow encoder can never back-pressure the tee
    # and freeze the live display branch.
    pipeline.add("queue", "record_queue",
                 {"leaky": 2, "max-size-buffers": 5})
    pipeline.add("nvvideoconvert", "record_convert", {"gpu-id": gpu_id})

    enc_w, enc_h = (_bounded_even(canvas_w, canvas_h, RECORD_MAX_SIDE)
                    if canvas_w and canvas_h else (0, 0))
    if enc_w and enc_h and (enc_w, enc_h) != (canvas_w, canvas_h):
        print(f"[reid] recording downscaled {canvas_w}x{canvas_h} -> "
              f"{enc_w}x{enc_h} (H.264 encode limit)")
        pipeline.add("capsfilter", "record_caps", {
            "caps": f"video/x-raw(memory:NVMM), width={enc_w}, height={enc_h}",
        })
        scale_chain = ["record_convert", "record_caps"]
    else:
        scale_chain = ["record_convert"]

    pipeline.add("nvv4l2h264enc", "record_encoder", {
        "bitrate": bitrate,
        "insert-sps-pps": 1,
    })
    pipeline.add("h264parse", "record_h264parse")
    # fragment-duration writes a fragmented (streamable) MP4: the file stays
    # playable even if the pipeline is stopped abruptly (e.g. Ctrl+C), because
    # there is no single trailing moov atom to finalize.
    pipeline.add("qtmux", "record_mux", {"fragment-duration": 1000})
    pipeline.add("filesink", "record_sink", {
        "location": str(out),
        "sync": 0,
        "async": 0,
    })
    pipeline.link(
        upstream,
        "record_queue",
        *scale_chain,
        "record_encoder",
        "record_h264parse",
        "record_mux",
        "record_sink",
    )
    return str(out)
