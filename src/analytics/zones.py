"""Named ground-plane zones + point-in-polygon assignment.

A Zone is a named world-plane polygon (mm) with optional tags (entry/exit/aisle…).
Authored by hand (or the zone-editor UI) into configs/zones/<scene>.json, or
auto-generated as a coarse grid when no file exists. Coarse zones are robust to
MMP's ~270 mm ground-plane projection noise (the *good* use of geometry here).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class Zone:
    name: str
    polygon: list[list[float]]            # [[x,y], ...] world mm, CCW
    tags: list[str] = field(default_factory=list)   # e.g. ["entry"], ["exit"], ["checkout"]

    def __post_init__(self):
        self._path = MplPath(np.asarray(self.polygon, dtype=float))

    def contains(self, x: float, y: float) -> bool:
        return bool(self._path.contains_point((x, y)))

    @property
    def centroid(self) -> tuple[float, float]:
        p = np.asarray(self.polygon, dtype=float)
        return float(p[:, 0].mean()), float(p[:, 1].mean())


def load_zones(path: str | Path) -> list[Zone]:
    data = json.load(open(path))
    return [Zone(z["name"], z["polygon"], z.get("tags", [])) for z in data["zones"]]


def save_zones(zones: list[Zone], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"zones": [{"name": z.name, "polygon": z.polygon, "tags": z.tags}
                         for z in zones]}, open(path, "w"), indent=2)


def assign_zone(x: float, y: float, zones: list[Zone]) -> str | None:
    """First zone whose polygon contains (x,y); None if outside all zones."""
    for z in zones:
        if z.contains(x, y):
            return z.name
    return None


def auto_grid_zones(xs: np.ndarray, ys: np.ndarray, nx: int = 3, ny: int = 3,
                    pct: float = 2.0) -> list[Zone]:
    """Coarse nx×ny grid over the robust extent of the points (pct..100-pct
    percentile, to ignore projection outliers). Named zone_<col><row> (A1,B2…)."""
    x0, x1 = np.percentile(xs, [pct, 100 - pct])
    y0, y1 = np.percentile(ys, [pct, 100 - pct])
    xe = np.linspace(x0, x1, nx + 1)
    ye = np.linspace(y0, y1, ny + 1)
    cols = "ABCDEFGH"
    zones = []
    for i in range(nx):
        for j in range(ny):
            poly = [[xe[i], ye[j]], [xe[i + 1], ye[j]],
                    [xe[i + 1], ye[j + 1]], [xe[i], ye[j + 1]]]
            zones.append(Zone(f"zone_{cols[i]}{j + 1}", poly))
    return zones
