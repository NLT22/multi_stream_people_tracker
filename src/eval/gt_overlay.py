"""
Ground-truth overlay probe for MTA dataset demo.

Draws GT bounding boxes (dashed green border) and person_id labels on top of
the pipeline's own detections so you can visually compare GT vs predictions.

Attaches as a pre-tiler probe on the tracker element (same placement as
SourceIdCollectorProbe) so source_id is exact.

Usage — wired in automatically when --show-gt / --mta-dataset is passed:
    python -m src.main \\
        --mta-dataset dataset/mta/MTA_ext_short/test \\
        --show-gt
"""

from __future__ import annotations

import traceback

import pyservicemaker as psm
from pyservicemaker import osd
import pandas as pd


# NvDS hard limit per DisplayMeta allocation
_MAX_RECTS_PER_META = 16
_MAX_TEXTS_PER_META = 16

# GT box style — bright green, thin border so it doesn't swamp the pred bbox
_GT_COLOR = osd.Color(0.0, 1.0, 0.2, 1.0)   # green
_GT_BORDER_WIDTH = 2


class GtOverlayProbe(psm.BatchMetadataOperator):
    """
    Pre-tiler probe: draws ground-truth boxes for the current frame.

    gt_by_cam : dict[cam_id, pd.DataFrame]
        DataFrame with columns: frame (int), person_id (int),
        left, top, width, height (float).

    snap_frames : int or None
        When set, floors each pipeline frame_number to the most-recent
        annotation boundary (multiples of snap_frames).
        Use for sparse annotations (e.g. Wildtrack: ~30 video frames per slot).
        Leave None for dense annotations (e.g. MTA: frame-exact).
    """

    def __init__(
        self,
        gt_by_cam: dict[int, pd.DataFrame],
        snap_frames: int | None = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        super().__init__()
        self._snap = snap_frames
        self._scale_x = scale_x
        self._scale_y = scale_y
        # Index by (cam_id, frame) → list of rows for O(1) lookup per frame
        self._index: dict[tuple[int, int], list[dict]] = {}
        for cam_id, df in gt_by_cam.items():
            for _, row in df.iterrows():
                key = (cam_id, int(row["frame"]))
                self._index.setdefault(key, []).append({
                    "person_id": int(row["person_id"]),
                    "left": float(row["left"]) * self._scale_x,
                    "top": float(row["top"]) * self._scale_y,
                    "width": float(row["width"]) * self._scale_x,
                    "height": float(row["height"]) * self._scale_y,
                })

    def _resolve_frame(self, frame_no: int) -> int:
        """Return the frame key to look up, applying snap if configured.

        Uses floor (not round) so the returned key is always the most-recent
        annotation slot — boxes reflect where people WERE, never where they
        will be.  This avoids the visual artefact of boxes leading the person.
        """
        if self._snap is None:
            return frame_no
        ann_idx = int(frame_no / self._snap)   # floor division
        return ann_idx * self._snap

    def handle_metadata(self, batch_meta) -> None:
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            print("[gt_overlay ERROR]")
            traceback.print_exc()

    def _handle_metadata(self, batch_meta) -> None:
        for frame_meta in batch_meta.frame_items:
            cam_id   = frame_meta.source_id
            frame_no = frame_meta.frame_number
            key_frame = self._resolve_frame(frame_no)
            if key_frame is None:
                continue
            boxes = self._index.get((cam_id, key_frame), [])
            if not boxes:
                continue
            self._draw_boxes(batch_meta, frame_meta, boxes)

    def _draw_boxes(self, batch_meta, frame_meta, boxes: list[dict],
                    frame_w: int = 1920, frame_h: int = 1080) -> None:
        writer = _RectTextWriter(batch_meta, frame_meta)
        for box in boxes:
            # Clip to frame — annotation coords can exceed frame dimensions.
            left   = max(0.0, box["left"])
            top    = max(0.0, box["top"])
            right  = min(float(frame_w), box["left"] + box["width"])
            bottom = min(float(frame_h), box["top"]  + box["height"])
            if right - left < 1.0 or bottom - top < 1.0:
                continue   # fully outside frame

            rect = osd.Rect()
            rect.left = left
            rect.top  = top
            rect.width  = right - left
            rect.height = bottom - top
            rect.border_width = _GT_BORDER_WIDTH
            rect.border_color = _GT_COLOR
            rect.has_bg_color = False
            writer.add_rect(rect)

            text = osd.Text()
            text.display_text = f"GT:{box['person_id']}".encode()
            text.x_offset = max(0, int(left))
            text.y_offset = max(0, int(top) - 18)
            text.font.name = osd.FontFamily.Serif
            text.font.size = 14
            text.font.color = _GT_COLOR
            text.set_bg_color = False
            writer.add_text(text)

        writer.flush()


class _RectTextWriter:
    """Splits OSD primitives across multiple DisplayMeta allocations."""

    def __init__(self, batch_meta, frame_meta) -> None:
        self._batch_meta = batch_meta
        self._frame_meta = frame_meta
        self._dm = None
        self._n_rects = 0
        self._n_texts = 0

    def add_rect(self, rect) -> None:
        self._ensure(rects=1)
        self._dm.add_rect(rect)
        self._n_rects += 1

    def add_text(self, text) -> None:
        self._ensure(texts=1)
        self._dm.add_text(text)
        self._n_texts += 1

    def flush(self) -> None:
        if self._dm is not None:
            self._frame_meta.append(self._dm)
            self._dm = None
            self._n_rects = 0
            self._n_texts = 0

    def _ensure(self, rects: int = 0, texts: int = 0) -> None:
        if self._dm is None:
            self._dm = self._batch_meta.acquire_display_meta()
            return
        if (
            self._n_rects + rects > _MAX_RECTS_PER_META
            or self._n_texts + texts > _MAX_TEXTS_PER_META
        ):
            self.flush()
            self._dm = self._batch_meta.acquire_display_meta()
