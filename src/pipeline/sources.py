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
import hashlib
import shutil
import subprocess
import sys
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

    This is used by the main app and milestones for fixed camera/file lists.
    """
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

    Used for quick experiments where a directory represents one camera set.
    """
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


def resolve_sources(inputs: List[str]) -> tuple[list[str], bool]:
    """Turn flexible CLI inputs into a (uris, is_live) pair.

    Accepts, in order of precedence:
      - a single .txt list file        -> load_uris_from_txt
      - a single directory             -> load_uris_from_folder (scans videos)
      - one or more paths / URIs        -> path_to_uri on each

    is_live is True if any URI is an rtsp:// stream (mux/sink need live settings).
    """
    if len(inputs) == 1:
        only = inputs[0]
        if "://" not in only and Path(only).is_dir():
            uris = load_uris_from_folder(only)
        elif only.endswith(".txt") and "://" not in only:
            uris = load_uris_from_txt(only)
        else:
            uris = [path_to_uri(only)]
    else:
        uris = [path_to_uri(p) for p in inputs]

    if not uris:
        raise ValueError(f"No video sources resolved from: {inputs}")
    is_live = any(u.startswith("rtsp://") for u in uris)
    return uris, is_live


def trim_sources(uris: List[str], seconds: float, start: float = 0.0,
                 cache_dir: str = "output/_trimmed") -> List[str]:
    """Pre-cut each file source to `seconds` of footage and return new URIs.

    Uses ffmpeg stream-copy (no re-encode → near-instant). Trimmed clips are
    cached, so re-running the same trim reuses them instead of cutting again.
    rtsp:// (live) sources cannot be trimmed and are passed through unchanged.
    """
    if not seconds:
        return uris

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[reid][trim] ffmpeg not found. Install it: sudo apt install -y ffmpeg")
        sys.exit(1)

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for uri in uris:
        if not uri.startswith("file://"):
            print(f"[reid][trim] skip non-file source (cannot trim): {uri}")
            result.append(uri)
            continue

        src = Path(uri[len("file://"):])
        suffix = src.suffix if src.suffix else ".mp4"
        key = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:10]
        out = out_dir / (
            f"{src.stem}_{key}_s{int(start)}_t{int(seconds)}{suffix}"
        )

        if out.exists() and out.stat().st_size > 0:
            print(f"[reid][trim] reuse {out.name}")
        else:
            print(f"[reid][trim] {src.name} -> {out.name} "
                  f"({seconds}s from {start}s)")
            cmd = [ffmpeg, "-y"]
            if start > 0:
                cmd += ["-ss", str(start)]   # fast keyframe seek before -i
            cmd += ["-i", str(src), "-t", str(seconds),
                    "-c", "copy", "-loglevel", "error", str(out)]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[reid][trim] ffmpeg failed for {src.name}: {e}")
                sys.exit(1)
        result.append("file://" + str(out.resolve()))

    return result
