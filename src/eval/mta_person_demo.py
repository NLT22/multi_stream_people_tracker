"""
MTA single-person cross-camera demo.

Highlights specific person_ids across cameras with a distinct color so you can
visually verify cross-camera identity consistency.

Highlighted persons: bright yellow border + label
Other persons (if --show-all): dim green border, no label

Run:
    python -m src.eval.mta_person_demo \\
        --mta-dataset dataset/mta/MTA_ext_short/test \\
        --person-ids 1993 \\
        [--cameras 1 2 3 4 5]          # default: all cams where person appears
        [--show-all]                   # also draw other persons (dim)
        [--trim-seconds 60]
        [--save-video output/videos/person_1993.mp4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyservicemaker as psm
from pyservicemaker import osd

from src.dataset.mta import MtaDataset
from src.pipeline.recording import add_recording_branch, compute_grid
from src.pipeline.sources import trim_sources
from src.utils.platform_utils import get_sink_element


_HIGHLIGHT_COLOR = osd.Color(1.0, 1.0, 0.0, 1.0)   # yellow  — tracked person
_OTHER_COLOR     = osd.Color(0.0, 0.8, 0.2, 0.6)   # dim green — everyone else
_HIGHLIGHT_WIDTH = 4
_OTHER_WIDTH     = 2
_MAX_RECTS = 16
_MAX_TEXTS  = 16


def _build_index(gt_by_cam: dict, highlight_ids: set[int]) -> dict:
    """Build {cam_id → {frame → [box, …]}} with 'highlight' flag per box."""
    index: dict[int, dict[int, list[dict]]] = {}
    for cam_id, df in gt_by_cam.items():
        cam_idx: dict[int, list[dict]] = {}
        for _, row in df.iterrows():
            cam_idx.setdefault(int(row["frame"]), []).append({
                "person_id": int(row["person_id"]),
                "left":      float(row["left"]),
                "top":       float(row["top"]),
                "width":     float(row["width"]),
                "height":    float(row["height"]),
                "highlight": int(row["person_id"]) in highlight_ids,
            })
        index[cam_id] = cam_idx
    return index


class CollectorProbe(psm.BatchMetadataOperator):
    def __init__(self, index: dict, pending: dict) -> None:
        super().__init__()
        self._index = index
        self._pending = pending

    def handle_metadata(self, batch_meta) -> None:
        self._pending.clear()
        for frame_meta in batch_meta.frame_items:
            cam_id   = frame_meta.source_id
            frame_no = frame_meta.frame_number
            cam_data = self._index.get(cam_id)
            if cam_data is None:
                continue
            boxes = cam_data.get(frame_no, [])
            if boxes:
                self._pending[cam_id] = boxes


class DrawProbe(psm.BatchMetadataOperator):
    def __init__(self, pending: dict, cam_ids: list[int],
                 tile_w: int, tile_h: int, cols: int,
                 show_all: bool) -> None:
        super().__init__()
        self._pending  = pending
        self._cam_ids  = cam_ids
        self._tile_w   = tile_w
        self._tile_h   = tile_h
        self._cols     = max(1, cols)
        self._show_all = show_all

    def handle_metadata(self, batch_meta) -> None:
        if not self._pending:
            return
        frame_meta = next(iter(batch_meta.frame_items), None)
        if frame_meta is None:
            return

        sx = self._tile_w / 1920.0
        sy = self._tile_h / 1080.0

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

            # Draw highlighted persons first so they appear on top
            for highlight_pass in (True, False):
                for box in boxes:
                    if box["highlight"] != highlight_pass:
                        continue
                    if not highlight_pass and not self._show_all:
                        continue
                    if n_rects >= _MAX_RECTS or n_texts >= _MAX_TEXTS:
                        _new_dm()
                    _draw_box(dm, box, ox, oy, sx, sy,
                              self._tile_w, self._tile_h)
                    n_rects += 1
                    if highlight_pass:
                        n_texts += 1

        if dm is not None:
            frame_meta.append(dm)


def _draw_box(dm, box: dict, ox: int, oy: int,
              sx: float, sy: float,
              tile_w: int, tile_h: int) -> None:
    left   = ox + box["left"]  * sx
    top    = oy + box["top"]   * sy
    right  = ox + (box["left"] + box["width"])  * sx
    bottom = oy + (box["top"]  + box["height"]) * sy

    left   = max(float(ox),          left)
    top    = max(float(oy),          top)
    right  = min(float(ox + tile_w), right)
    bottom = min(float(oy + tile_h), bottom)
    if right - left < 1.0 or bottom - top < 1.0:
        return

    color  = _HIGHLIGHT_COLOR if box["highlight"] else _OTHER_COLOR
    bwidth = _HIGHLIGHT_WIDTH  if box["highlight"] else _OTHER_WIDTH

    rect = osd.Rect()
    rect.left         = left
    rect.top          = top
    rect.width        = right - left
    rect.height       = bottom - top
    rect.border_width = bwidth
    rect.border_color = color
    rect.has_bg_color = False
    dm.add_rect(rect)

    if box["highlight"]:
        text = osd.Text()
        text.display_text = f"ID:{box['person_id']}".encode()
        text.x_offset     = max(0, int(left))
        text.y_offset     = max(0, int(top) - 18)
        text.font.name    = osd.FontFamily.Serif
        text.font.size    = 14
        text.font.color   = color
        text.set_bg_color = False
        dm.add_text(text)


def run(mta_path: str, person_ids: list[int], cam_ids: list[int] | None,
        show_all: bool, tile_w: int, tile_h: int,
        no_display: bool, save_video: str | None,
        record_bitrate: int, gpu_id: int,
        trim_seconds: float | None, trim_start: float) -> None:

    mta = MtaDataset(str(Path(mta_path).parent), split=Path(mta_path).name)
    available = mta.get_cam_ids()
    highlight_ids = set(person_ids)

    # If no --cameras given, default to cams where ANY highlight person appears
    if cam_ids is None:
        cams_with_person: set[int] = set()
        for pid in highlight_ids:
            for c in available:
                df = mta.load_gt(c)
                if pid in df["person_id"].values:
                    cams_with_person.add(c)
        selected = sorted(cams_with_person) or available
        print(f"[mta_person_demo] person_ids={person_ids} found in cams={selected}")
    else:
        selected = [c for c in cam_ids if c in available]

    if not selected:
        print("[mta_person_demo] No cameras selected.")
        sys.exit(1)

    uris = [f"file://{(Path(mta_path) / f'cam_{c}' / f'cam_{c}.mp4').resolve()}"
            for c in selected]
    if trim_seconds is not None:
        uris = trim_sources(uris, trim_seconds, trim_start)

    gt_by_cam = {c: mta.load_gt(c) for c in selected}
    index = _build_index(gt_by_cam, highlight_ids)

    n = len(uris)
    rows_grid, cols = compute_grid(n)
    total_w, total_h = tile_w * cols, tile_h * rows_grid

    print(f"[mta_person_demo] grid={rows_grid}×{cols}  canvas={total_w}×{total_h}")
    print(f"[mta_person_demo] Yellow = person_id {person_ids}")

    pipeline = psm.Pipeline("mta-person-demo")
    pipeline.add("nvstreammux", "mux", {
        "batch-size": n, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": gpu_id,
    })
    for i, uri in enumerate(uris):
        pipeline.add("nvurisrcbin", f"src_{i}", {"uri": uri, "gpu-id": gpu_id})
        pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows_grid, "columns": cols,
        "width": total_w, "height": total_h, "gpu-id": gpu_id,
    })
    pipeline.add("nvosdbin", "osd", {
        "gpu-id": gpu_id, "process-mode": 1,
        "display-text": 1, "display-bbox": 1, "text-size": 14,
    })
    pipeline.link("mux", "tiler", "osd")

    pending: dict = {}
    pipeline.attach("mux", psm.Probe("collect", CollectorProbe(index, pending)))
    pipeline.attach("tiler", psm.Probe("draw", DrawProbe(
        pending, selected, tile_w, tile_h, cols, show_all)))

    sink_sync = 0
    if save_video and not no_display:
        pipeline.add("tee", "output_tee")
        pipeline.add("queue", "display_queue", {"leaky": 2, "max-size-buffers": 5})
        pipeline.add(get_sink_element(), "sink",
                     {"sync": sink_sync, "qos": 0, "async": 0})
        pipeline.link("osd", "output_tee", "display_queue", "sink")
        written = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[mta_person_demo] Recording to: {written}")
    elif save_video:
        written = add_recording_branch(
            pipeline, "osd", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
        print(f"[mta_person_demo] Recording to: {written}")
    elif no_display:
        pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
        pipeline.link("osd", "sink")
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0})
        pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[mta_person_demo] Running. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[mta_person_demo] Stopped.")
    finally:
        pipeline.stop()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Highlight specific MTA person_ids across cameras (no inference)")
    p.add_argument("--mta-dataset", required=True, metavar="PATH",
                   help="MTA split folder, e.g. dataset/mta/MTA_ext_short/test")
    p.add_argument("--person-ids", nargs="+", type=int, required=True,
                   help="person_id(s) to highlight in yellow")
    p.add_argument("--cameras", nargs="+", type=int, default=None,
                   help="Camera IDs to show (default: all cams where person appears)")
    p.add_argument("--show-all", action="store_true",
                   help="Also draw other persons in dim green")
    p.add_argument("--trim-seconds", type=float, default=None,
                   help="Hard-cut each source after this many seconds")
    p.add_argument("--trim-start", type=float, default=0.0,
                   help="Start offset in seconds (default: 0)")
    p.add_argument("--tile-w", type=int, default=960)
    p.add_argument("--tile-h", type=int, default=540)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-video", default=None, metavar="PATH")
    p.add_argument("--record-bitrate", type=int, default=8_000_000)
    args = p.parse_args()

    run(mta_path=args.mta_dataset,
        person_ids=args.person_ids,
        cam_ids=args.cameras,
        show_all=args.show_all,
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
