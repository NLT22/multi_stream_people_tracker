"""Typed per-detection row passed through the gallery probe each frame.

Replaces the loose dict that ``CrossCameraGalleryProbe`` built per object. Fields
are the single source of truth for what a "row" carries through row-building,
quality annotation, identity assignment, conflict resolution, merge, draw and
export.

Transitional dict-compat: ``__getitem__``/``__setitem__``/``get`` let the existing
``row["key"]`` call sites (and the pure helpers in quality.py / fusion_bridge.py /
visualization.py, which are still unit-tested with plain dicts) keep working
unchanged while call sites migrate to attribute access incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionRow:
    # Set at construction (from DeepStream object metadata).
    src: int
    track_id: int
    track_key: tuple[int, int]
    rect: dict
    raw_embedding: list[float]
    foot_world: tuple[float, float] | None = None

    # Filled in as the row flows through the per-frame pipeline.
    embedding: list[float] = field(default_factory=list)
    tracklet_len: int = 0
    gid: int | None = None
    previous_gid: int | None = None
    had_previous_gid: bool = False
    embedding_quality_ok: bool = False
    embedding_quality_reason: str = ""
    update_gallery: bool = False
    allow_new_gid: bool = False
    identity_conflict: bool = False
    suppress_gallery_update: bool = False
    release_previous_gid: bool = False
    defer_assignment: bool = False
    gallery_updated: bool = False

    # ---- transitional dict-compat (remove once all call sites use attrs) -----
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)
