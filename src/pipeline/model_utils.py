"""Helpers for model-specific DeepStream settings."""

from pathlib import Path

import yaml


def infer_person_class_id(nvinfer_config: str, default: int = 2) -> int:
    """Return the class id for "person" from an nvinfer YAML label file."""
    config_path = Path(nvinfer_config)
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
        label_path = raw.get("property", {}).get("labelfile-path")
        if not label_path:
            return default

        labels_file = Path(label_path)
        if not labels_file.is_absolute():
            labels_file = config_path.parent / labels_file

        labels = [
            line.strip().lower()
            for line in labels_file.read_text().splitlines()
            if line.strip()
        ]
        return labels.index("person")
    except (OSError, ValueError, yaml.YAMLError):
        return default


def deepstream_tracker_lib_path() -> str:
    """Return the nvtracker low-level library path for common DS 9 layouts."""
    candidates = (
        "/opt/nvidia/deepstream/deepstream-9.0/lib/libnvds_nvmultiobjecttracker.so",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    )
    for path in candidates:
        if Path(path).exists():
            return path
    return candidates[0]


def infer_source_id_from_tiled_box(
    rect_params,
    tile_width: int,
    tile_height: int,
    columns: int,
    num_sources: int,
    fallback: int = 0,
) -> int:
    """Infer source id from a bbox whose coords have been scaled by tiler."""
    if tile_width <= 0 or tile_height <= 0 or columns <= 0:
        return fallback

    cx = float(rect_params.left) + float(rect_params.width) * 0.5
    cy = float(rect_params.top) + float(rect_params.height) * 0.5
    col = int(cx // tile_width)
    row = int(cy // tile_height)
    src = row * columns + col
    if 0 <= src < num_sources:
        return src
    return fallback
