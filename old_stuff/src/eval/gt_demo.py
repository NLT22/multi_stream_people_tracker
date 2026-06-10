"""
Ground-truth demo — displays only GT boxes, no inference or tracker.

Pipeline: nvurisrcbin → nvstreammux → nvosdbin → sink
A single probe on the mux draws GT bounding boxes each frame.

Run:
    python -m src.eval.gt_demo \\
        --mta-dataset dataset/mta/MTA_ext_short/test \\
        [--cameras 0 1]          # subset of cameras (default: all found)
        [--split test]           # train or test
        [--tile-w 960] [--tile-h 540]
        [--no-display]
        [--save-video output/videos/gt_demo.mp4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyservicemaker as psm
from pyservicemaker import osd

from src.dataset.mta import MtaDataset
from src.pipeline.recording import add_recording_branch, compute_grid
from src.pipeline.sources import path_to_uri, trim_sources
from src.utils.platform_utils import get_sink_element


_GT_COLOR = osd.Color(0.0, 1.0, 0.2, 1.0)   # bright green
_GT_BORDER_WIDTH = 3
_MAX_RECTS = 16
_MAX_TEXTS = 16


class GtCollectorProbe(psm.BatchMetadataOperator):
    """Pre-tiler: collects GT boxes for each source in the current batch."""

    def __init__(self, gt_index: dict[tuple[int, int], list[dict]],
                 pending: dict) -> None:
        super().__init__()
        self._index = gt_index
        self._pending = pending   # shared dict: cam_id → list[dict]

    def handle_metadata(self, batch_meta) -> None:
        self._pending.clear()
        for frame_meta in batch_meta.frame_items:
            cam_id = frame_meta.source_id
            frame_no = frame_meta.frame_number
            boxes = self._index.get((cam_id, frame_no), [])
            if boxes:
                self._pending[cam_id] = boxes


class GtDrawProbe(psm.BatchMetadataOperator):
    """Post-tiler: draws GT boxes on the tiled canvas using stored box data."""

    def __init__(self, pending: dict, cam_ids: list[int],
                 tile_w: int, tile_h: int, cols: int,
                 src_w: int = 1920, src_h: int = 1080) -> None:
        super().__init__()
        self._pending = pending
        self._cam_ids = cam_ids
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = max(1, cols)
        self._src_w = src_w
        self._src_h = src_h

    def handle_metadata(self, batch_meta) -> None:
        if not self._pending:
            return
        # Post-tiler: one frame in batch covers the whole tiled canvas.
        frame_meta = next(iter(batch_meta.frame_items), None)
        if frame_meta is None:
            return

        sx = self._tile_w / max(1, self._src_w)
        sy = self._tile_h / max(1, self._src_h)

        dm = None
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
                self._add_box(dm, box, ox, oy, sx, sy)
                n_rects += 1
                n_texts += 1

        if dm is not None:
            frame_meta.append(dm)

    def _add_box(self, dm, box: dict,
                 ox: int, oy: int, sx: float, sy: float) -> None:
        left   = ox + box["left"]   * sx
        top    = oy + box["top"]    * sy
        right  = ox + (box["left"] + box["width"])  * sx
        bottom = oy + (box["top"]  + box["height"]) * sy

        # Clip to tile boundary
        left   = max(float(ox),              left)
        top    = max(float(oy),              top)
        right  = min(float(ox + self._tile_w), right)
        bottom = min(float(oy + self._tile_h), bottom)
        if right - left < 1.0 or bottom - top < 1.0:
            return

        rect = osd.Rect()
        rect.left   = left
        rect.top    = top
        rect.width  = right - left
        rect.height = bottom - top
        rect.border_width = _GT_BORDER_WIDTH
        rect.border_color = _GT_COLOR
        rect.has_bg_color = False
        dm.add_rect(rect)

        text = osd.Text()
        text.display_text = f"GT:{box['person_id']}".encode()
        text.x_offset = max(0, int(left))
        text.y_offset = max(0, int(top) - 18)
        text.font.name = osd.FontFamily.Serif
        text.font.size = 12
        text.font.color = _GT_COLOR
        text.set_bg_color = False
        dm.add_text(text)



def _build_gt_index(gt_by_cam: dict) -> dict[tuple[int, int], list[dict]]:
    index: dict[tuple[int, int], list[dict]] = {}
    for cam_id, df in gt_by_cam.items():
        for _, row in df.iterrows():
            key = (cam_id, int(row["frame"]))
            index.setdefault(key, []).append({
                "person_id": int(row["person_id"]),
                "left": float(row["left"]),
                "top": float(row["top"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
            })
    return index


def run(mta_path: str, split: str, cam_ids: list[int] | None,
        tile_w: int, tile_h: int,
        no_display: bool, save_video: str | None,
        record_bitrate: int, gpu_id: int,
        trim_seconds: float | None = None,
        trim_start: float = 0.0) -> None:

    mta = MtaDataset(str(Path(mta_path).parent), split=Path(mta_path).name)
    available = mta.get_cam_ids()
    selected = [c for c in (cam_ids or available) if c in available]
    if not selected:
        print(f"[gt_demo] No cameras found under {mta_path}")
        sys.exit(1)

    uris = [f"file://{(Path(mta_path) / f'cam_{c}' / f'cam_{c}.mp4').resolve()}"
            for c in selected]
    if trim_seconds is not None:
        uris = trim_sources(uris, trim_seconds, trim_start)
    gt_by_cam = {c: mta.load_gt(c) for c in selected}
    gt_index = _build_gt_index(gt_by_cam)

    n = len(uris)
    rows, cols = compute_grid(n)
    total_w, total_h = tile_w * cols, tile_h * rows

    print(f"[gt_demo] cameras={selected}  grid={rows}×{cols}  canvas={total_w}×{total_h}")
    print(f"[gt_demo] Green boxes = ground-truth person IDs")

    pipeline = psm.Pipeline("gt-demo")

    pipeline.add("nvstreammux", "mux", {
        "batch-size": n,
        "batched-push-timeout": 40000,
        "width": 1920, "height": 1080,
        "gpu-id": gpu_id,
    })
    for i, uri in enumerate(uris):
        pipeline.add("nvurisrcbin", f"src_{i}", {"uri": uri, "gpu-id": gpu_id})
        pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": total_w, "height": total_h,
        "gpu-id": gpu_id,
    })
    pipeline.add("nvosdbin", "osd", {
        "gpu-id": gpu_id,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size": 12,
    })
    pipeline.link("mux", "tiler", "osd")

    # Two-probe approach:
    # 1. Pre-tiler on mux: collect GT boxes per source (source_id is exact here)
    # 2. Post-tiler on tiler: draw with tile offset + scale onto the canvas
    pending: dict = {}
    pipeline.attach("mux", psm.Probe("gt_collect", GtCollectorProbe(gt_index, pending)))
    pipeline.attach("tiler", psm.Probe("gt_draw", GtDrawProbe(
        pending, selected, tile_w, tile_h, cols, src_w=1920, src_h=1080)))

    sink_sync = 0  # MTA is 41fps; sync=1 causes buffer drops on most GPUs
    if save_video and not no_display:
        pipeline.add("tee", "output_tee")
        pipeline.add("queue", "display_queue", {"leaky": 2, "max-size-buffers": 5})
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0, "async": 0})
        pipeline.link("osd", "output_tee", "display_queue", "sink")
        written = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[gt_demo] Recording to: {written}")
    elif save_video:
        written = add_recording_branch(
            pipeline, "osd", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[gt_demo] Recording to: {written}")
    elif no_display:
        pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
        pipeline.link("osd", "sink")
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0})
        pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[gt_demo] Running. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[gt_demo] Stopped.")
    finally:
        pipeline.stop()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Display MTA ground-truth boxes only (no inference)")
    p.add_argument("--mta-dataset", required=True, metavar="PATH",
                   help="MTA split folder, e.g. dataset/mta/MTA_ext_short/test")
    p.add_argument("--split", default=None,
                   help="Override split name (default: last component of --mta-dataset)")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Camera IDs to display (default: all)")
    p.add_argument("--tile-w", type=int, default=960)
    p.add_argument("--tile-h", type=int, default=540)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-video", default=None, metavar="PATH",
                   help="Save annotated video to this path")
    p.add_argument("--record-bitrate", type=int, default=8_000_000)
    p.add_argument("--trim-seconds", type=float, default=None,
                   help="Hard-cut each source after this many seconds "
                        "(pipeline exits cleanly at end-of-stream)")
    p.add_argument("--trim-start", type=float, default=0.0,
                   help="Start offset in seconds for --trim-seconds (default: 0)")
    args = p.parse_args()

    mta_path = args.mta_dataset
    run(mta_path=mta_path,
        split=args.split or Path(mta_path).name,
        cam_ids=args.cameras,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        no_display=args.no_display,
        save_video=args.save_video,
        record_bitrate=args.record_bitrate,
        gpu_id=args.gpu_id,
        trim_seconds=args.trim_seconds,
        trim_start=args.trim_start)


if __name__ == "__main__":
    main()
