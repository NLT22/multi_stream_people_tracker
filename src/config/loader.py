"""
Pipeline configuration loader.

WHY THIS EXISTS:
  All milestones and the main pipeline read from configs/pipeline.yaml.
  This module parses that YAML into a typed PipelineConfig dataclass so
  the rest of the code gets IDE autocompletion and clear field names
  instead of raw dict["key"] access everywhere.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DetectionConfig:
    enabled: bool = True
    config_file: str = "configs/models/nvinfer_yolov8_people.yml"


@dataclass
class TrackerConfig:
    enabled: bool = True
    config_file: str = "configs/tracker/nvdcf_perf.yaml"
    tracker_width: int = 640
    tracker_height: int = 384
    sub_batches: str | None = None


@dataclass
class DisplayConfig:
    osd_enabled: bool = True
    tiled_display: bool = True
    tile_width: int = 1280
    tile_height: int = 720


@dataclass
class PipelineConfig:
    # ── Source ──────────────────────────────────────────────────────────────
    # "video_files" | "folder_input" | "rtsp_cameras"
    source_mode: str = "video_files"

    # Paths to each source config file
    source_configs: dict = field(default_factory=lambda: {
        "video_files":  "configs/sources/video_files.txt",
        "folder_input": "configs/sources/folder_input.yaml",
        "rtsp_cameras": "configs/sources/rtsp_cameras.txt",
    })

    # ── Sub-configs ──────────────────────────────────────────────────────────
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # ── Performance ──────────────────────────────────────────────────────────
    batch_size: int = 4
    gpu_id: int = 0

    # ── Derived (set after loading URIs) ─────────────────────────────────────
    num_sources: int = 0  # filled in by pipeline/sources.py after URI loading

    # ── Convenience property ─────────────────────────────────────────────────
    @property
    def active_source_config(self) -> str:
        """Return the source config path for the currently selected mode."""
        return self.source_configs.get(self.source_mode, "")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """
        Load PipelineConfig from a YAML file.

        TODO (Milestone 4+): This method is used by main.py and the later
        milestones. For the early milestones (01-03) you pass URIs directly
        via --input, so this is not called yet.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with yaml_path.open() as f:
            raw = yaml.safe_load(f)

        cfg = cls()

        # ── source ────────────────────────────────────────────────────────
        cfg.source_mode = raw.get("source_mode", cfg.source_mode)
        if "source_configs" in raw:
            cfg.source_configs.update(raw["source_configs"])

        # ── detection ────────────────────────────────────────────────────
        if "detection" in raw:
            d = raw["detection"]
            cfg.detection = DetectionConfig(
                enabled=d.get("enabled", True),
                config_file=d.get("config_file", cfg.detection.config_file),
            )

        # ── tracker ───────────────────────────────────────────────────────
        if "tracker" in raw:
            t = raw["tracker"]
            cfg.tracker = TrackerConfig(
                enabled=t.get("enabled", True),
                config_file=t.get("config_file", cfg.tracker.config_file),
                tracker_width=t.get("tracker_width", cfg.tracker.tracker_width),
                tracker_height=t.get("tracker_height", cfg.tracker.tracker_height),
                sub_batches=t.get("sub_batches", cfg.tracker.sub_batches),
            )

        # ── display ───────────────────────────────────────────────────────
        if "display" in raw:
            disp = raw["display"]
            cfg.display = DisplayConfig(
                osd_enabled=disp.get("osd_enabled", True),
                tiled_display=disp.get("tiled_display", True),
                tile_width=disp.get("tile_width", 1280),
                tile_height=disp.get("tile_height", 720),
            )

        # ── performance ───────────────────────────────────────────────────
        cfg.batch_size = raw.get("batch_size", cfg.batch_size)
        cfg.gpu_id = raw.get("gpu_id", cfg.gpu_id)

        return cfg

    def summary(self) -> str:
        """Return a human-readable summary for logging at startup."""
        lines = [
            "─── Pipeline Config ───────────────────────────────",
            f"  source_mode   : {self.source_mode}",
            f"  source_config : {self.active_source_config}",
            f"  detection     : {'enabled — ' + self.detection.config_file if self.detection.enabled else 'disabled'}",
            f"  tracker       : {'enabled — ' + self.tracker.config_file if self.tracker.enabled else 'disabled'}",
            f"  osd           : {'enabled' if self.display.osd_enabled else 'disabled'}",
            f"  tiled_display : {'enabled (' + str(self.display.tile_width) + 'x' + str(self.display.tile_height) + ')' if self.display.tiled_display else 'disabled'}",
            f"  batch_size    : {self.batch_size}",
            f"  gpu_id        : {self.gpu_id}",
            "───────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
