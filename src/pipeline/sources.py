"""
Source URI loaders.

WHY THIS EXISTS:
  nvurisrcbin accepts any URI: file://, rtsp://, http://.
  This module converts whatever the user configured (file paths, folders,
  RTSP URLs) into a uniform list of URI strings that the pipeline builder
  can iterate and attach to nvstreammux.

  Keeping this logic separate means the pipeline builder never needs to
  know HOW sources are specified — it just gets a list of URIs.
"""

import os
from pathlib import Path
from typing import List


# Video file extensions recognised when scanning a folder
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".h264", ".h265", ".ts", ".mov"}


def path_to_uri(path: str) -> str:
    """
    Convert a local file path to a file:// URI.

    nvurisrcbin requires URIs, not bare paths.
    Paths that already start with a scheme (file://, rtsp://, http://)
    are returned unchanged.

    Examples:
        "/home/user/video.mp4"  → "file:///home/user/video.mp4"
        "file:///home/user/v.mp4" → unchanged
        "rtsp://192.168.1.1/cam" → unchanged
    """
    if "://" in path:
        return path
    return "file://" + os.path.abspath(path)


def load_uris_from_txt(txt_path: str) -> List[str]:
    """
    Read a text file listing video sources, one per line.

    - Lines starting with # are comments → skipped
    - Empty lines → skipped
    - Local paths are converted to file:// URIs automatically

    TODO (Milestone 2): Call this from your pipeline script after implementing it.
    """
    # TODO: Implement this function
    #   1. Open txt_path for reading
    #   2. For each line: strip whitespace, skip if empty or starts with "#"
    #   3. Convert to URI with path_to_uri()
    #   4. Append to a list and return it
    #
    # Starter code:
    uris = []
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {txt_path}")

    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            uris.append(path_to_uri(line))

    if not uris:
        raise ValueError(
            f"No video sources found in {txt_path}.\n"
            "Edit the file and add at least one video path."
        )
    return uris


def load_uris_from_folder(folder_path: str, extensions=None, max_files: int = 0, sort_by: str = "name") -> List[str]:
    """
    Scan a folder and return file:// URIs for all matching video files.

    TODO (Milestone 2 extension): Activate this by setting
         source_mode: folder_input in pipeline.yaml.
    """
    # TODO: Implement folder scanning
    #   1. Use Path(folder_path).glob() or rglob() to find files
    #   2. Filter by extension (case-insensitive)
    #   3. Sort by name or mtime
    #   4. Apply max_files limit if > 0
    #   5. Convert each path to file:// URI
    #
    if extensions is None:
        extensions = _VIDEO_EXTENSIONS

    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    files = [f for f in folder.iterdir() if f.suffix.lower() in {e.lower() for e in extensions}]

    if sort_by == "name":
        files.sort(key=lambda f: f.name)
    elif sort_by == "mtime":
        files.sort(key=lambda f: f.stat().st_mtime)

    if max_files > 0:
        files = files[:max_files]

    uris = [path_to_uri(str(f)) for f in files]

    if not uris:
        raise ValueError(f"No video files found in: {folder_path}")
    return uris


def load_uris_from_rtsp_txt(txt_path: str) -> List[str]:
    """
    Read RTSP URLs from a text file, one URL per line.
    Comments (#) and blank lines are skipped.

    TODO (future milestone): Activate by setting source_mode: rtsp_cameras.
    Note: RTSP sources need live-source=1 on nvstreammux and sync=0 on sink.
    """
    uris = []
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"RTSP source file not found: {txt_path}")

    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("rtsp://"):
                print(f"[WARNING] Skipping non-RTSP line: {line}")
                continue
            uris.append(line)

    if not uris:
        raise ValueError(
            f"No RTSP URLs found in {txt_path}.\n"
            "Uncomment and fill in your camera URLs."
        )
    return uris


def load_uris_from_folder_yaml(yaml_path: str) -> List[str]:
    """Load folder config from YAML and call load_uris_from_folder."""
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    return load_uris_from_folder(
        folder_path=cfg.get("folder_path", "."),
        extensions=cfg.get("extensions", list(_VIDEO_EXTENSIONS)),
        max_files=cfg.get("max_files", 0),
        sort_by=cfg.get("sort_by", "name"),
    )


def load_uris(source_mode: str, source_config_path: str) -> List[str]:
    """
    Dispatch to the correct loader based on source_mode.

    This is the single entry point used by PipelineBuilder and milestones.

    Args:
        source_mode: "video_files" | "folder_input" | "rtsp_cameras"
        source_config_path: path to the corresponding config file

    Returns:
        List of URI strings (file:// or rtsp://)
    """
    if source_mode == "video_files":
        return load_uris_from_txt(source_config_path)
    elif source_mode == "folder_input":
        return load_uris_from_folder_yaml(source_config_path)
    elif source_mode == "rtsp_cameras":
        return load_uris_from_rtsp_txt(source_config_path)
    else:
        raise ValueError(
            f"Unknown source_mode: '{source_mode}'. "
            "Choose: video_files | folder_input | rtsp_cameras"
        )
