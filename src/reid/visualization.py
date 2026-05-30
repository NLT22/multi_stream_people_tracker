"""OSD helpers for visualizing tracker trajectories.

The ReID gallery owns identity assignment, while this module owns only drawing
state. Keeping it separate avoids turning gallery.py into an OSD toolbox.
"""

from __future__ import annotations

from collections import deque
import colorsys

from pyservicemaker import osd


_MAX_LINES_PER_DISPLAY_META = 16
_MAX_CIRCLES_PER_DISPLAY_META = 16


class TrajectoryVisualizer:
    """Draw recent per-camera local-track trajectories as OSD line overlays."""

    def __init__(
        self,
        tile_w: int,
        tile_h: int,
        cols: int,
        num_sources: int,
        *,
        max_points: int = 96,
        sample_interval: int = 2,
        max_segments_per_track: int = 24,
        line_width: int = 2,
        draw_points: bool = True,
        pretiler: bool = False,
    ):
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = max(1, cols)
        self._num_sources = num_sources
        self._max_points = max(2, max_points)
        self._sample_interval = max(1, sample_interval)
        self._max_segments_per_track = max(1, max_segments_per_track)
        self._line_width = max(1, line_width)
        self._draw_points = draw_points
        self._pretiler = pretiler
        self._history: dict[tuple[int, int], deque[tuple[int, float, float, int | None]]] = {}
        self._last_sample_frame: dict[tuple[int, int], int] = {}

    def update_and_draw(self, batch_meta, frame_meta, rows: list[dict], frame_count: int) -> None:
        """Update track histories from current rows and append DisplayMeta."""
        active_keys = set()
        for row in rows:
            key = row["track_key"]
            active_keys.add(key)
            if frame_count - self._last_sample_frame.get(key, -self._sample_interval) < self._sample_interval:
                continue

            x, y = self._display_point(row)
            points = self._history.setdefault(key, deque(maxlen=self._max_points))
            points.append((frame_count, x, y, row.get("gid")))
            self._last_sample_frame[key] = frame_count

        self._prune_inactive(active_keys, frame_count)
        self._draw(batch_meta, frame_meta, rows)

    def _display_point(self, row: dict) -> tuple[float, float]:
        rect = row["rect"]
        x = rect["left"] + rect["width"] * 0.5
        y = rect["top"] + rect["height"]
        if self._pretiler:
            return x, y

        src = row["src"]
        col = src % self._cols
        tile_row = src // self._cols
        return x + col * self._tile_w, y + tile_row * self._tile_h

    def _prune_inactive(self, active_keys: set[tuple[int, int]], frame_count: int) -> None:
        # Keep recent inactive histories briefly so short tracker gaps do not
        # immediately erase the visual trail, but avoid unbounded growth.
        stale = [
            key for key in self._history
            if (
                key not in active_keys
                and frame_count - self._last_sample_frame.get(key, frame_count)
                > self._max_points * self._sample_interval
            )
        ]
        for key in stale:
            self._history.pop(key, None)
            self._last_sample_frame.pop(key, None)

    def _draw(self, batch_meta, frame_meta, rows: list[dict]) -> None:
        writer = _DisplayMetaWriter(batch_meta, frame_meta)

        visible_keys = {row["track_key"] for row in rows}
        for key in visible_keys:
            points = self._history.get(key)
            if not points or len(points) < 2:
                continue

            gid = points[-1][3]
            color = self._color_for_id(gid if gid is not None else key[1])
            previous = None
            # NvDsDisplayMeta has small per-meta capacities. Draw only the
            # latest segments and let _DisplayMetaWriter split them safely.
            tail_points = list(points)[-(self._max_segments_per_track + 1):]
            for _, x, y, _ in tail_points:
                if previous is not None:
                    line = osd.Line()
                    line.x1 = int(previous[0])
                    line.y1 = int(previous[1])
                    line.x2 = int(x)
                    line.y2 = int(y)
                    line.width = self._line_width
                    line.color = color
                    writer.add_line(line)
                previous = (x, y)

            if self._draw_points:
                _, x, y, _ = points[-1]
                circle = osd.Circle()
                circle.xc = int(x)
                circle.yc = int(y)
                circle.radius = 3
                circle.width = max(1, self._line_width)
                circle.color = color
                circle.has_bg_color = True
                circle.bg_color = color
                writer.add_circle(circle)

        writer.flush()

    @staticmethod
    def _color_for_id(identity: int) -> osd.Color:
        return color_for_id(identity)


def color_for_id(identity: int) -> osd.Color:
    """Stable bright color for a tracker/global id."""
    # Golden-ratio hue spacing gives more separable colors than hashing RGB
    # directly, especially on gray warehouse floors.
    hue = (int(identity) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 1.0)
    return osd.Color(r, g, b, 1.0)


def style_object_by_id(obj_meta, identity: int | None, border_width: int = 3) -> None:
    """Color an object's bbox and label by Global ID before nvosdbin draws it."""
    if identity is None:
        color = osd.Color(0.8, 0.8, 0.8, 1.0)
    else:
        color = color_for_id(identity)

    rect_params = getattr(obj_meta, "rect_params", None)
    if rect_params is not None:
        try:
            rect_params.border_width = border_width
            rect_params.border_color = color
        except (AttributeError, TypeError):
            pass

    text_params = getattr(obj_meta, "text_params", None)
    font = getattr(text_params, "font", None) if text_params is not None else None
    if font is not None:
        try:
            font.color = color
        except (AttributeError, TypeError):
            pass


class _DisplayMetaWriter:
    """Append OSD primitives across multiple DisplayMeta objects safely."""

    def __init__(self, batch_meta, frame_meta):
        self._batch_meta = batch_meta
        self._frame_meta = frame_meta
        self._display_meta = None
        self._line_count = 0
        self._circle_count = 0

    def add_line(self, line) -> None:
        self._ensure_capacity(lines=1)
        self._display_meta.add_line(line)
        self._line_count += 1

    def add_circle(self, circle) -> None:
        self._ensure_capacity(circles=1)
        self._display_meta.add_circle(circle)
        self._circle_count += 1

    def flush(self) -> None:
        if self._display_meta is not None:
            self._frame_meta.append(self._display_meta)
            self._display_meta = None
            self._line_count = 0
            self._circle_count = 0

    def _ensure_capacity(self, lines: int = 0, circles: int = 0) -> None:
        if self._display_meta is None:
            self._display_meta = self._batch_meta.acquire_display_meta()
            return
        if (
            self._line_count + lines > _MAX_LINES_PER_DISPLAY_META
            or self._circle_count + circles > _MAX_CIRCLES_PER_DISPLAY_META
        ):
            self.flush()
            self._display_meta = self._batch_meta.acquire_display_meta()
