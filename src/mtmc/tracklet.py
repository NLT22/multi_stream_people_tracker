"""Per-tracklet message — the schema perception publishes to the bus.

A Tracklet carries a small BANK of k (subsampled) per-crop ReID embeddings, NOT a
single mean — keeping per-crop discrimination (the offline strength) while staying
cheap (k≈4-8 crops, frame-skipped; the in-tracker ReID already runs ~1-in-5). Mirrors
the MDX `mdx-raw` per-tracklet record ("embedding_bank").
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class Tracklet:
    sensor_id: int                 # camera id
    tracklet_id: int               # local track id within that camera
    t_start: float                 # first frame/timestamp seen
    t_end: float                   # last frame/timestamp seen
    bank: np.ndarray               # (K, D) L2-normalized per-crop embeddings
    foot_world: np.ndarray | None = None   # (2,) world (x, y) mm, optional
    n_obs: int = 0                 # number of detections aggregated

    @property
    def key(self) -> tuple[int, int]:
        return (self.sensor_id, self.tracklet_id)

    @property
    def embedding(self) -> np.ndarray:
        """Centroid of the bank (backward-compat / coarse use)."""
        return _l2(self.bank.mean(axis=0))

    def overlaps(self, other: "Tracklet") -> bool:
        return not (self.t_end < other.t_start or other.t_end < self.t_start)
