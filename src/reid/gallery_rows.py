"""Metadata->row extraction and embedding-quality annotation.

Extracted from gallery.py as a mixin of CrossCameraGalleryProbe. These methods
operate on the probe's shared state (self._gs, self._track_to_gid,
self._tracklets, self._cfg, ...); the split is by concern, not ownership.
"""

from __future__ import annotations

from src.reid import quality

class GalleryRowsMixin:
    def _local_rect(self, rect_params, src: int, frame_meta) -> dict:
        """Return bbox coordinates in source-local/tile-local space."""
        left = float(rect_params.left)
        top = float(rect_params.top)
        width = float(rect_params.width)
        height = float(rect_params.height)

        if self._pretiler:
            frame_w, frame_h = self._frame_size(frame_meta)
        else:
            col = src % max(1, self._cols)
            row = src // max(1, self._cols)
            left -= col * self._tile_w
            top -= row * self._tile_h
            if self._frame_sizes is not None:
                if src in self._frame_sizes:
                    frame_w, frame_h = self._frame_sizes[src]
                elif self._frame_sizes:
                    # Use minimum valid frame width as fallback so all cameras
                    # land in the same coordinate space.  The min corresponds to
                    # the actual source resolution (e.g. 640×360) while the
                    # tile_w default (1280) creates a mixed PRED/GT space that
                    # breaks the single-scale assumption in metrics_mmp.py.
                    frame_w = min(w for w, _ in self._frame_sizes.values())
                    frame_h = min(h for _, h in self._frame_sizes.values())
                else:
                    frame_w, frame_h = float(self._tile_w), float(self._tile_h)
            else:
                frame_w, frame_h = float(self._tile_w), float(self._tile_h)
            sx = frame_w / max(1.0, float(self._tile_w))
            sy = frame_h / max(1.0, float(self._tile_h))
            left *= sx
            top *= sy
            width *= sx
            height *= sy

        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    @staticmethod
    def _frame_size(frame_meta) -> tuple[float, float]:
        """Best-effort frame size for pre-tiler quality checks."""
        width_names = ("source_frame_width", "frame_width", "source_width", "width")
        height_names = ("source_frame_height", "frame_height", "source_height", "height")
        width = next(
            (float(getattr(frame_meta, name)) for name in width_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            1920.0,
        )
        height = next(
            (float(getattr(frame_meta, name)) for name in height_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            1080.0,
        )
        return width, height

    def _annotate_embedding_quality(self, rows: list[dict]) -> None:
        for row in rows:
            ok, reason = self._embedding_quality(row, rows)
            row["embedding_quality_ok"] = ok
            row["embedding_quality_reason"] = reason

    def _embedding_quality(self, row: dict,
                           rows: list[dict]) -> tuple[bool, str]:
        # Pure logic lives in src/reid/quality.py; pass the current tuning values.
        return quality.embedding_quality(
            row, rows,
            enabled=self._cfg.enable_embedding_quality_gate,
            edge_margin_ratio=self._cfg.reid_edge_margin_ratio,
            min_height_ratio=self._cfg.reid_min_bbox_height_ratio,
            min_area_ratio=self._cfg.reid_min_bbox_area_ratio,
            min_aspect=self._cfg.reid_min_bbox_aspect_ratio,
            max_aspect=self._cfg.reid_max_bbox_aspect_ratio,
            max_overlap_iou=self._cfg.reid_max_overlap_iou_for_update,
        )

    @staticmethod
    def _rect_iou(a: dict, b: dict) -> float:
        return quality.rect_iou(a, b)
