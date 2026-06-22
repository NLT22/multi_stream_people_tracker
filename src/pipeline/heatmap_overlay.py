"""Live occupancy-heatmap overlay drawn on the tiled video (production_todo §6).

Post-tiler probe: accumulates person foot points (bbox bottom-centre) into a
decaying density grid over the tiled canvas, and each frame draws the densest
cells as filled OSD circles coloured blue→red by occupancy. Because it reads the
post-tiler object rect_params (already in canvas coordinates, same space the OSD
renders in), it needs no per-tile geometry maths and appears in the live view and
in any `--save-video` recording.

Attach to the tiler (post-tiler) alongside the gallery probe; see runner.py.
"""
from __future__ import annotations

import numpy as np
import pyservicemaker as psm
from pyservicemaker import osd

from src.reid.visualization import _DisplayMetaWriter


class HeatmapOverlayProbe(psm.BatchMetadataOperator):
    def __init__(self, canvas_w: int, canvas_h: int, *,
                 grid_w: int = 48, grid_h: int = 27, decay: float = 0.96,
                 max_circles: int = 220, radius: int = 9, max_alpha: float = 0.5):
        super().__init__()
        self._cw = max(1, int(canvas_w))
        self._ch = max(1, int(canvas_h))
        self._gw = grid_w
        self._gh = grid_h
        self._decay = decay
        self._maxc = max_circles
        self._radius = radius
        self._max_alpha = max_alpha
        self._grid = np.zeros((grid_h, grid_w), dtype=np.float64)

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            try:
                self._draw(batch_meta, frame_meta)
            except Exception:  # never let an overlay crash the pipeline
                import traceback
                traceback.print_exc()

    def _draw(self, batch_meta, frame_meta):
        self._grid *= self._decay
        for obj in frame_meta.object_items:
            rp = obj.rect_params
            fx = rp.left + rp.width / 2.0
            fy = rp.top + rp.height
            cx = int(np.clip(fx / self._cw * self._gw, 0, self._gw - 1))
            cy = int(np.clip(fy / self._ch * self._gh, 0, self._gh - 1))
            self._grid[cy, cx] += 1.0

        peak = self._grid.max()
        if peak <= 1e-6:
            return

        writer = _DisplayMetaWriter(batch_meta, frame_meta)
        flat = self._grid.ravel()
        cell_w = self._cw / self._gw
        cell_h = self._ch / self._gh
        for idx in np.argsort(-flat)[:self._maxc]:
            v = float(flat[idx] / peak)
            if v < 0.06:
                break
            r, c = divmod(int(idx), self._gw)
            circle = osd.Circle()
            circle.xc = int((c + 0.5) * cell_w)
            circle.yc = int((r + 0.5) * cell_h)
            circle.radius = self._radius
            circle.width = 1
            color = self._color(v)
            circle.color = color
            circle.has_bg_color = True
            circle.bg_color = color
            writer.add_circle(circle)
        writer.flush()

    def _color(self, v: float) -> "osd.Color":
        # blue (low) -> green (mid) -> red (high); alpha grows with density.
        color = osd.Color()
        color.red = float(min(1.0, 2.0 * v))
        color.green = float(max(0.0, 1.0 - abs(2.0 * v - 1.0)))
        color.blue = float(max(0.0, 1.0 - 2.0 * v))
        color.alpha = float(self._max_alpha * (0.25 + 0.75 * v))
        return color
