"""Typed configuration for the pipeline runner.

`runner.run()` used to take ~40 positional/keyword arguments; passing them by
position at the call site was error-prone and unreadable. This dataclass bundles
them into one object. Field names and defaults match the old `run()` signature
exactly, so behavior is unchanged.

Kept deliberately free of heavy imports (no pyservicemaker / GStreamer) so it can
be constructed and inspected without bringing up the DeepStream stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineRunConfig:
    # ── Required ────────────────────────────────────────────────────────────
    sources: list[str]
    nvinfer_config: str
    tracker_config: str
    tile_w: int
    tile_h: int
    debug_similarity: bool
    use_hungarian_assignment: bool
    enforce_unique_per_stream: bool
    save_video: str | None
    record_bitrate: int
    no_display: bool

    # ── Optional (defaults mirror the old run() signature) ──────────────────
    batch_size: int | None = None
    gpu_id: int = 0
    tracker_width: int = 640
    tracker_height: int = 384
    tracker_sub_batches: str | None = None
    # nvstreammux surface size. Inputs are scaled to this before PGIE/tracker/SGIE,
    # so it drives per-object crop/preprocess cost. Default 1920x1080 (legacy); set to
    # the source resolution (e.g. 640x360) to avoid upscaling small inputs.
    mux_width: int = 1920
    mux_height: int = 1080
    max_sources: int | None = None
    force_rebuild_engine: bool = False
    trim_seconds: float | None = None
    trim_start: float = 0.0
    pretiler: bool = False
    no_tiler: bool = False
    show_trajectories: bool = True
    trajectory_history: int = 96
    trajectory_sample_interval: int = 20
    trajectory_max_segments: int = 24
    export_predictions: str | None = None
    live_buffered_window: int = 0   # >0: flush per-det embedding chunks every N frames (live buffered MTMC)
    disable_gallery: bool = False
    osd_enabled: bool = True
    gt_by_cam: dict | None = None
    gt_snap_frames: int | None = None
    gt_scale: tuple[float, float] = (1.0, 1.0)
    no_sync: bool = False
    loop_video: bool = False
    reid_sgie_config: str | None = None
    # Sidecar ReID: path to ONNX model. When set, SGIE is skipped; a background
    # thread extracts crops from the NvBufSurface and runs inference off the
    # critical path. Mutually exclusive with reid_sgie_config.
    sidecar_reid_onnx: str | None = None
    nvdsanalytics_config: str | None = None
    heatmap_overlay: bool = False
    buffered_remap: str | None = None
    export_only: bool = False
    geometry: Any = None
    reid_config: Any = None
