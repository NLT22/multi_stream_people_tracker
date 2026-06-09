"""Runtime configuration: build the defaults dict from a pipeline
YAML preset + the gallery module's tuning constants.
Extracted from src/main.py (see also src/config/args.py)."""

import argparse
import sys
from pathlib import Path

import pyservicemaker as psm
import yaml

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
)
from src.pipeline.engine_prep import prepare_nvinfer_config
from src.pipeline.recording import add_recording_branch, compute_grid
from src.dataset.mta import MtaDataset
from src.dataset.mmp_tracking import MMPTrackingDataset, MMPTrackingShortDataset
from src.dataset.wildtrack import WildtrackDataset
from src.eval.export import PredictionExporter
from src.eval.gt_overlay import GtOverlayProbe
from src.pipeline.sources import resolve_sources, trim_sources
from src.reid import gallery
from src.reid.visualization import TrajectoryVisualizer
from src.utils.platform_utils import get_sink_element


DEFAULT_CONFIG_PATH = "configs/pipeline.yaml"


def _load_defaults(config_path: str) -> dict:
    """Read pipeline.yaml and turn it into CLI defaults for this app."""
    defaults = {
        "sources": ["configs/sources/video_files.txt"],
        "nvinfer_config": "configs/models/nvinfer_yolov11_people.yml",
        "reid_sgie_config": None,
        "tracker_config": "configs/tracker/nvdeepsort_reid_swin.yaml",
        "tracker_width": 640,
        "tracker_height": 384,
        "tracker_sub_batches": None,
        "tile_w": 1280,
        "tile_h": 720,
        "batch_size": None,
        "max_sources": None,
        "gpu_id": 0,
        "pretiler": False,
        "no_tiler": False,
        "save_video": None,
        "record_bitrate": 8000000,
        "no_display": False,
        "no_sync": False,
        "disable_gallery": False,
        "osd_enabled": True,
        "show_trajectories": True,
        "trajectory_history": 96,
        "trajectory_sample_interval": 20,
        "trajectory_max_segments": 24,
        # ReID/gallery tuning — mirrors gallery.py module defaults
        "similarity_threshold":              gallery.SIMILARITY_THRESHOLD,
        "gallery_max_age":                   gallery.GALLERY_MAX_AGE,
        "assignment_max_candidates":         gallery.GLOBAL_ASSIGNMENT_MAX_CANDIDATES,
        "disable_id_stickiness":             not gallery.ENABLE_ID_STICKINESS,
        "id_switch_margin":                  gallery.ID_SWITCH_MARGIN,
        "allow_ambiguous_match":             not gallery.ENABLE_AMBIGUOUS_MATCH_REJECTION,
        "match_ambiguity_margin":            gallery.MATCH_AMBIGUITY_MARGIN,
        "disable_global_merge":              not gallery.ENABLE_GLOBAL_ID_MERGE,
        "global_merge_threshold":            gallery.GLOBAL_ID_MERGE_THRESHOLD,
        "global_merge_min_embeddings":       gallery.GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS,
        "global_merge_margin":              gallery.GLOBAL_ID_MERGE_MARGIN,
        "global_merge_interval":             gallery.GLOBAL_ID_MERGE_INTERVAL,
        "global_merge_max_candidates":       gallery.GLOBAL_ID_MERGE_MAX_CANDIDATES,
        "micro_batch_fusion":                gallery.USE_MICRO_BATCH_FUSION,
        "fusion_interval":                   gallery.MICRO_BATCH_FUSION_INTERVAL,
        "fusion_threshold":                  gallery.MICRO_BATCH_FUSION_THRESHOLD,
        "disable_tracklet":                  not gallery.USE_TRACKLET_EMBEDDING,
        "tracklet_embedding_interval":       gallery.TRACKLET_EMBEDDING_INTERVAL,
        "disable_embedding_quality_gate":    not gallery.ENABLE_EMBEDDING_QUALITY_GATE,
        "tracklet_window":                   gallery.TRACKLET_MAX_EMBEDDINGS,
        "tracklet_min_embeddings":           gallery.TRACKLET_MIN_EMBEDDINGS_FOR_MATCH,
        "tracklet_max_age":                  gallery.TRACKLET_MAX_AGE,
        "geometry_assignment_mode":          gallery.GEO_ASSIGNMENT_MODE,
        "geometry_reid_margin":              gallery.GEO_REID_MARGIN,
    }

    path = Path(config_path)
    if not path.exists():
        return defaults

    raw = yaml.safe_load(path.read_text()) or {}
    source_configs = {
        "video_files": "configs/sources/video_files.txt",
        "folder_input": "configs/sources/folder_input.yaml",
        "rtsp_cameras": "configs/sources/rtsp_cameras.txt",
    }
    source_configs.update(raw.get("source_configs", {}) or {})
    source_mode = raw.get("source_mode", "video_files")
    defaults["sources"] = [source_configs.get(source_mode, defaults["sources"][0])]

    detection = raw.get("detection", {}) or {}
    tracker = raw.get("tracker", {}) or {}
    display = raw.get("display", {}) or {}
    runtime = raw.get("runtime", {}) or {}

    defaults["nvinfer_config"] = detection.get(
        "config_file", defaults["nvinfer_config"])
    defaults["reid_sgie_config"] = detection.get(
        "reid_sgie_config", defaults["reid_sgie_config"])
    defaults["tracker_config"] = tracker.get(
        "config_file", defaults["tracker_config"])
    defaults["tracker_width"] = tracker.get(
        "tracker_width", defaults["tracker_width"])
    defaults["tracker_height"] = tracker.get(
        "tracker_height", defaults["tracker_height"])
    defaults["tracker_sub_batches"] = tracker.get(
        "sub_batches", defaults["tracker_sub_batches"])
    defaults["tile_w"] = display.get("tile_width", defaults["tile_w"])
    defaults["tile_h"] = display.get("tile_height", defaults["tile_h"])
    defaults["osd_enabled"] = display.get(
        "osd_enabled", defaults["osd_enabled"])
    defaults["batch_size"] = raw.get("batch_size", defaults["batch_size"])
    defaults["max_sources"] = runtime.get(
        "max_sources", defaults["max_sources"])
    defaults["gpu_id"] = raw.get("gpu_id", defaults["gpu_id"])
    defaults["pretiler"] = runtime.get("pretiler", defaults["pretiler"])
    defaults["no_tiler"] = runtime.get("no_tiler", defaults["no_tiler"])
    defaults["save_video"] = runtime.get("save_video", defaults["save_video"])
    defaults["record_bitrate"] = runtime.get(
        "record_bitrate", defaults["record_bitrate"])
    defaults["no_display"] = runtime.get("no_display", defaults["no_display"])
    defaults["no_sync"] = runtime.get("no_sync", defaults["no_sync"])
    defaults["disable_gallery"] = runtime.get(
        "disable_gallery", defaults["disable_gallery"])
    defaults["show_trajectories"] = runtime.get(
        "show_trajectories", defaults["show_trajectories"])
    defaults["trajectory_history"] = runtime.get(
        "trajectory_history", defaults["trajectory_history"])
    defaults["trajectory_sample_interval"] = runtime.get(
        "trajectory_sample_interval", defaults["trajectory_sample_interval"])
    defaults["trajectory_max_segments"] = runtime.get(
        "trajectory_max_segments", defaults["trajectory_max_segments"])

    # reid: section — all keys map directly to their defaults dict counterpart
    reid = raw.get("reid", {}) or {}
    _bool = lambda v: bool(v)
    for key, yaml_key in [
        ("similarity_threshold",           "similarity_threshold"),
        ("gallery_max_age",                "gallery_max_age"),
        ("assignment_max_candidates",      "assignment_max_candidates"),
        ("id_switch_margin",               "id_switch_margin"),
        ("match_ambiguity_margin",         "match_ambiguity_margin"),
        ("global_merge_threshold",         "global_merge_threshold"),
        ("global_merge_min_embeddings",    "global_merge_min_embeddings"),
        ("global_merge_margin",            "global_merge_margin"),
        ("global_merge_interval",          "global_merge_interval"),
        ("global_merge_max_candidates",    "global_merge_max_candidates"),
        ("tracklet_embedding_interval",    "tracklet_embedding_interval"),
        ("tracklet_window",                "tracklet_window"),
        ("tracklet_min_embeddings",        "tracklet_min_embeddings"),
        ("tracklet_max_age",               "tracklet_max_age"),
        ("geometry_assignment_mode",       "geometry_assignment_mode"),
        ("geometry_reid_margin",           "geometry_reid_margin"),
        ("fusion_interval",                "fusion_interval"),
        ("fusion_threshold",               "fusion_threshold"),
    ]:
        if yaml_key in reid:
            defaults[key] = reid[yaml_key]
    if "micro_batch_fusion" in reid:
        defaults["micro_batch_fusion"] = bool(reid["micro_batch_fusion"])
    # Boolean toggles stored as positive flags in YAML for readability
    if "id_stickiness" in reid:
        defaults["disable_id_stickiness"] = not reid["id_stickiness"]
    if "ambiguous_match_rejection" in reid:
        defaults["allow_ambiguous_match"] = not reid["ambiguous_match_rejection"]
    if "global_merge" in reid:
        defaults["disable_global_merge"] = not reid["global_merge"]
    if "tracklet_embedding" in reid:
        defaults["disable_tracklet"] = not reid["tracklet_embedding"]
    if "embedding_quality_gate" in reid:
        defaults["disable_embedding_quality_gate"] = not reid["embedding_quality_gate"]

    return defaults


