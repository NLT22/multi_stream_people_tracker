"""Named-zone registry + foot-point -> zone resolver  (RAG Phase A).

The zone vocabulary is reused from the existing gst-nvdsanalytics ROI configs
(`configs/analytics/nvdsanalytics_<env>.txt`) — the same files the webUI ROI
editor emits. Each `[roi-filtering-stream-N]` / `[overcrowding-stream-N]` block
defines one or more `roi-<NAME>=x1;y1;x2;y2;...` polygons for camera N.

ROI polygons are authored in the OSD/mux resolution (default 1920x1080); the
exported detections are in source resolution (default 640x360). We normalise
both to [0,1] so a foot point in pred space can be tested against a polygon in
ROI space regardless of scale.

A foot point resolves to the zone id `cam{N}:{NAME}` of every ROI (for its
camera) whose polygon contains it; if none, `cam{N}:other`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def _point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon (no external deps). poly: list of (x,y)."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


@dataclass
class ZoneRegistry:
    """Named ROI polygons per camera, normalised to [0,1]."""
    # cam_id -> list of (zone_name, normalised polygon[(x,y),...])
    zones: dict[int, list[tuple[str, list[tuple[float, float]]]]] = field(default_factory=dict)
    roi_w: float = 1920.0
    roi_h: float = 1080.0

    @classmethod
    def from_analytics(cls, path: str | Path, roi_w: float = 1920.0,
                       roi_h: float = 1080.0) -> "ZoneRegistry":
        """Parse a gst-nvdsanalytics config into a zone registry."""
        text = Path(path).read_text()
        reg = cls(roi_w=roi_w, roi_h=roi_h)
        cur_cam: int | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"\[(?:roi-filtering|overcrowding)-stream-(\d+)\]", line)
            if m:
                cur_cam = int(m.group(1))
                continue
            if line.startswith("[") and not line.startswith("[roi") and not line.startswith("[over"):
                cur_cam = None
                continue
            m = re.match(r"roi-([A-Za-z0-9_-]+)\s*=\s*(.+)", line)
            if m and cur_cam is not None:
                name = m.group(1)
                nums = [float(v) for v in re.split(r"[;,]", m.group(2).strip()) if v.strip()]
                pts = [(nums[i] / roi_w, nums[i + 1] / roi_h)
                       for i in range(0, len(nums) - 1, 2)]
                if len(pts) >= 3:
                    reg.zones.setdefault(cur_cam, []).append((name, pts))
        return reg

    def resolve(self, cam_id: int, foot_x: float, foot_y: float,
                pred_w: float = 640.0, pred_h: float = 360.0) -> str:
        """Return the zone id for a foot point (source-space pixels)."""
        nx, ny = foot_x / pred_w, foot_y / pred_h
        for name, poly in self.zones.get(cam_id, []):
            if _point_in_poly(nx, ny, poly):
                return f"cam{cam_id}:{name}"
        return f"cam{cam_id}:other"

    def zone_names(self) -> list[str]:
        out = []
        for cam, lst in sorted(self.zones.items()):
            for name, _ in lst:
                out.append(f"cam{cam}:{name}")
            out.append(f"cam{cam}:other")
        return out
