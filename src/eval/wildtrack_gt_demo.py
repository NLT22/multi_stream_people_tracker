"""
Wildtrack ground-truth demo — displays only GT boxes, no inference or tracker.

Pipeline: nvurisrcbin → nvstreammux → nvmultistreamtiler → nvosdbin → sink
Two probes draw GT boxes for the nearest annotation frame.

Wildtrack annotations are at 2 fps; video runs at ~59.94 fps.
The collector snaps each pipeline frame to the nearest annotation slot
(within ±SNAP_TOLERANCE_FRAMES) so GT boxes appear at the right time.

Run:
    python -m src.eval.wildtrack_gt_demo \\
        --wildtrack-dataset dataset/Wildtrack \\
        [--cameras 0 1 2]      # 0-based cam IDs (default: all found)
        [--minutes 5]          # limit playback duration (default: all annotated)
        [--tile-w 960] [--tile-h 540]
        [--no-display]
        [--save-video output/videos/wildtrack_gt.mp4]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pyservicemaker as psm
from pyservicemaker import osd

from src.dataset.wildtrack import WildtrackDataset, FRAMES_PER_ANN, ANN_FPS
from src.pipeline.recording import add_recording_branch, compute_grid
from src.pipeline.sources import trim_sources
from src.utils.platform_utils import get_sink_element


_GT_COLOR = osd.Color(0.0, 1.0, 0.2, 1.0)   # bright green
_GT_BORDER_WIDTH = 3
_MAX_RECTS = 16
_MAX_TEXTS  = 16


def _build_gt_index(
    gt_by_cam: dict[int, object],
) -> dict[int, dict[int, list[dict]]]:
    """Build {cam_id → {ann_frame → [box, …]}} for O(1) per-frame lookup.

    ann_frame is the nearest video frame rounded to annotation boundary
    (multiples of FRAMES_PER_ANN).
    """
    index: dict[int, dict[int, list[dict]]] = {}
    for cam_id, df in gt_by_cam.items():
        cam_idx: dict[int, list[dict]] = {}
        for _, row in df.iterrows():
            cam_idx.setdefault(int(row["frame"]), []).append({
                "person_id": int(row["person_id"]),
                "left":  float(row["left"]),
                "top":   float(row["top"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
            })
        index[cam_id] = cam_idx
    return index


def _snap_frame(frame_no: int) -> int:
    """Return the most-recent annotation video-frame for this pipeline frame.

    Uses floor so boxes always reflect the PAST annotation slot, never a future
    one.  This avoids the visual artefact of a box appearing ahead of the person
    (which happens with round() when the frame is in the first half of a slot).
    """
    ann_idx = int(frame_no / FRAMES_PER_ANN)   # floor
    return round(ann_idx * FRAMES_PER_ANN)


class GtCollectorProbe(psm.BatchMetadataOperator):
    """Pre-tiler: collects GT boxes for each source in the current batch."""

    def __init__(
        self,
        gt_index: dict[int, dict[int, list[dict]]],
        pending: dict,
    ) -> None:
        super().__init__()
        self._index = gt_index
        self._pending = pending   # cam_id → list[dict]

    def handle_metadata(self, batch_meta) -> None:
        self._pending.clear()
        for frame_meta in batch_meta.frame_items:
            cam_id   = frame_meta.source_id
            frame_no = frame_meta.frame_number
            ann_frame = _snap_frame(frame_no)
            cam_data = self._index.get(cam_id)
            if cam_data is None:
                continue
            boxes = cam_data.get(ann_frame, [])
            if boxes:
                self._pending[cam_id] = boxes


class GtDrawProbe(psm.BatchMetadataOperator):
    """Post-tiler: draws GT boxes on the tiled canvas."""

    def __init__(
        self,
        pending: dict,
        cam_ids: list[int],
        tile_w: int,
        tile_h: int,
        cols: int,
        src_w: int = 1920,
        src_h: int = 1080,
    ) -> None:
        super().__init__()
        self._pending = pending
        self._cam_ids = cam_ids
        self._tile_w  = tile_w
        self._tile_h  = tile_h
        self._cols    = max(1, cols)
        self._src_w   = src_w
        self._src_h   = src_h

    def handle_metadata(self, batch_meta) -> None:
        if not self._pending:
            return
        frame_meta = next(iter(batch_meta.frame_items), None)
        if frame_meta is None:
            return

        sx = self._tile_w / max(1, self._src_w)
        sy = self._tile_h / max(1, self._src_h)

        dm      = None
        n_rects = 0
        n_texts = 0

        def _new_dm():
            nonlocal dm, n_rects, n_texts
            if dm is not None:
                frame_meta.append(dm)
            dm = batch_meta.acquire_display_meta()
            n_rects = 0
            n_texts = 0

        _new_dm()

        for cam_id, boxes in self._pending.items():
            try:
                tile_idx = self._cam_ids.index(cam_id)
            except ValueError:
                continue
            ox = (tile_idx % self._cols) * self._tile_w
            oy = (tile_idx // self._cols) * self._tile_h

            for box in boxes:
                if n_rects >= _MAX_RECTS or n_texts >= _MAX_TEXTS:
                    _new_dm()
                _draw_box(dm, box, ox, oy, sx, sy, self._tile_w, self._tile_h)
                n_rects += 1
                n_texts += 1

        if dm is not None:
            frame_meta.append(dm)


def _draw_box(dm, box: dict,
              ox: int, oy: int, sx: float, sy: float,
              tile_w: int, tile_h: int) -> None:
    left   = ox + box["left"]   * sx
    top    = oy + box["top"]    * sy
    right  = ox + (box["left"] + box["width"])  * sx
    bottom = oy + (box["top"]  + box["height"]) * sy

    # Clip to tile boundary — annotation coords can exceed frame dimensions
    # (person partially outside frame or touching camera border).
    left   = max(float(ox),          left)
    top    = max(float(oy),          top)
    right  = min(float(ox + tile_w), right)
    bottom = min(float(oy + tile_h), bottom)
    width  = right - left
    height = bottom - top
    if width < 1.0 or height < 1.0:
        return   # fully outside tile — skip

    rect = osd.Rect()
    rect.left         = left
    rect.top          = top
    rect.width        = width
    rect.height       = height
    rect.border_width = _GT_BORDER_WIDTH
    rect.border_color = _GT_COLOR
    rect.has_bg_color = False
    dm.add_rect(rect)

    text = osd.Text()
    text.display_text = f"GT:{box['person_id']}".encode()
    text.x_offset     = max(0, int(left))
    text.y_offset     = max(0, int(top) - 18)
    text.font.name    = osd.FontFamily.Serif
    text.font.size    = 12
    text.font.color   = _GT_COLOR
    text.set_bg_color = False
    dm.add_text(text)


def run(
    dataset_path: str,
    cam_ids: list[int] | None,
    minutes: float | None,
    trim_seconds: float | None,
    trim_start: float,
    tile_w: int,
    tile_h: int,
    no_display: bool,
    save_video: str | None,
    record_bitrate: int,
    gpu_id: int,
) -> None:
    dataset = WildtrackDataset(dataset_path)
    available = dataset.get_cam_ids()
    selected = [c for c in (cam_ids or available) if c in available]
    if not selected:
        print(f"[wildtrack_gt_demo] No cameras found under {dataset_path}")
        sys.exit(1)

    # Determine how many seconds of GT annotations to load.
    # --minutes caps to annotated range; --trim-seconds further limits playback.
    ann_duration = dataset.annotated_duration_seconds
    max_seconds = minutes * 60.0 if minutes is not None else ann_duration
    if trim_seconds is not None:
        max_seconds = min(max_seconds, trim_start + trim_seconds)
    max_seconds = min(max_seconds, ann_duration)

    uris = dataset.get_video_uris(selected)
    if trim_seconds is not None:
        uris = trim_sources(uris, trim_seconds, trim_start)

    gt_by_cam = dataset.load_all_gt(cam_ids=selected, max_seconds=max_seconds)
    gt_index  = _build_gt_index(gt_by_cam)

    n = len(uris)
    rows_grid, cols = compute_grid(n)
    total_w, total_h = tile_w * cols, tile_h * rows_grid

    print(f"[wildtrack_gt_demo] cameras={selected}  grid={rows_grid}×{cols}  "
          f"canvas={total_w}×{total_h}")
    print(f"[wildtrack_gt_demo] annotated coverage: {ann_duration:.0f}s  "
          f"playing: {max_seconds:.0f}s")
    print(f"[wildtrack_gt_demo] Green boxes = ground-truth person IDs")

    pipeline = psm.Pipeline("wildtrack-gt-demo")

    pipeline.add("nvstreammux", "mux", {
        "batch-size":          n,
        "batched-push-timeout": 40000,
        "width":  1920,
        "height": 1080,
        "gpu-id": gpu_id,
    })
    for i, uri in enumerate(uris):
        pipeline.add("nvurisrcbin", f"src_{i}", {"uri": uri, "gpu-id": gpu_id})
        pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows":    rows_grid,
        "columns": cols,
        "width":   total_w,
        "height":  total_h,
        "gpu-id":  gpu_id,
    })
    pipeline.add("nvosdbin", "osd", {
        "gpu-id":       gpu_id,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size":    12,
    })
    pipeline.link("mux", "tiler", "osd")

    pending: dict = {}
    pipeline.attach("mux", psm.Probe(
        "wt_gt_collect",
        GtCollectorProbe(gt_index, pending),
    ))
    pipeline.attach("tiler", psm.Probe(
        "wt_gt_draw",
        GtDrawProbe(pending, selected, tile_w, tile_h, cols,
                    src_w=1920, src_h=1080),
    ))

    # Wildtrack ~60fps; sync=0 avoids buffer drops at playback speed.
    sink_sync = 0
    if save_video and not no_display:
        pipeline.add("tee", "output_tee")
        pipeline.add("queue", "display_queue",
                     {"leaky": 2, "max-size-buffers": 5})
        pipeline.add(get_sink_element(), "sink",
                     {"sync": sink_sync, "qos": 0, "async": 0})
        pipeline.link("osd", "output_tee", "display_queue", "sink")
        written = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[wildtrack_gt_demo] Recording to: {written}")
    elif save_video:
        written = add_recording_branch(
            pipeline, "osd", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[wildtrack_gt_demo] Recording to: {written}")
    elif no_display:
        pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
        pipeline.link("osd", "sink")
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0})
        pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[wildtrack_gt_demo] Running. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[wildtrack_gt_demo] Stopped.")
    finally:
        pipeline.stop()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Display Wildtrack ground-truth boxes only (no inference)")
    p.add_argument("--wildtrack-dataset", required=True, metavar="PATH",
                   help="Root folder of the Wildtrack dataset "
                        "(must contain cam1.mp4…cam7.mp4 and annotations_positions/)")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="0-based camera IDs to display (default: all found)")
    p.add_argument("--minutes", type=float, default=None,
                   help="Limit to this many minutes of annotated video "
                        "(max: ~3.3 min = full annotated range)")
    p.add_argument("--trim-seconds", type=float, default=None,
                   help="Hard-cut each source video after this many seconds "
                        "(pre-trims the mp4 before feeding to the pipeline, "
                        "so the pipeline exits cleanly at end-of-stream)")
    p.add_argument("--trim-start", type=float, default=0.0,
                   help="Start offset in seconds for --trim-seconds (default: 0)")
    p.add_argument("--tile-w", type=int, default=960)
    p.add_argument("--tile-h", type=int, default=540)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-video", default=None, metavar="PATH",
                   help="Save annotated video to this path")
    p.add_argument("--record-bitrate", type=int, default=8_000_000)
    args = p.parse_args()

    run(
        dataset_path=args.wildtrack_dataset,
        cam_ids=args.cameras,
        minutes=args.minutes,
        trim_seconds=args.trim_seconds,
        trim_start=args.trim_start,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        no_display=args.no_display,
        save_video=args.save_video,
        record_bitrate=args.record_bitrate,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
