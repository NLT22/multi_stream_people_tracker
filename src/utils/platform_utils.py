"""
Platform utilities — detect GPU, choose correct sink element.

WHY THIS EXISTS:
  DeepStream has different sink elements per platform:
    x86_64  → nveglglessink  (EGL/OpenGL renderer)
    Jetson  → nv3dsink        (3D hardware compositor)
  Hardcoding one breaks the other platform.
  This module centralises the decision so all milestones stay portable.
"""

import platform
import os


def get_sink_element() -> str:
    """Return the correct display sink element name for this platform."""
    if platform.processor() == "aarch64":
        return "nv3dsink"
    return "nveglglessink"


def get_sink_properties(is_live: bool = False) -> dict:
    """
    Return baseline sink properties.

    is_live: set True for RTSP sources or tee-split pipelines.
             Adds async=0 to prevent state transition deadlock.
             See: LEARNING_NOTES.md § async=0 for live sources
    """
    props = {"sync": 0 if is_live else 1, "qos": 0}
    if is_live:
        # CRITICAL: Without async=0 the pipeline stalls in PAUSED state
        # when any upstream element is a live source or tee is used.
        props["async"] = 0
    return props


def check_display() -> bool:
    """Return True if a display is available (DISPLAY env var is set)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def gpu_info_str() -> str:
    """Return a human-readable GPU info string for logging."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "GPU info unavailable"
    except Exception:
        return "GPU info unavailable"
