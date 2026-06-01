"""
Wildtrack multi-camera dataset loader.

Folder layout expected:
    <root>/
        cam1.mp4 … cam7.mp4
        annotations_positions/
            00000000.json   # annotation frame 0  (t = 0.00 s)
            00000005.json   # annotation frame 1  (t = 0.50 s)
            …
            00001995.json   # annotation frame 399 (t = 199.75 s)

Annotation format per JSON file (list of person entries):
    [
      { "personID": 247,
        "views": [
          { "viewNum": 0, "xmin": 1345, "xmax": 1380, "ymin": 107, "ymax": 222 },
          …                           # viewNum 0-6 → cam1-cam7
        ]
      }, …
    ]
    Values of -1 mean the person is not visible in that view.

Timing:
    Annotations are at 2 fps; video is at ~59.94 fps (60000/1001).
    Every 5-step filename maps to one 0.5 s annotation slot:
        annotation_index = int(stem) // 5        (0-based)
        video_frame      ≈ annotation_index * (VIDEO_FPS / ANN_FPS)
                         ≈ annotation_index * 29.97
    Coverage: 400 annotations × 0.5 s = 200 s of annotated video.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


# Wildtrack camera names and their viewNum index in the annotation files.
CAM_NAMES = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]
CAM_IDS = list(range(7))   # 0-based, maps 1:1 to CAM_NAMES

VIDEO_FPS = 60000 / 1001   # ≈ 59.94
ANN_FPS   = 2.0             # annotation rate
# frames per annotation step in the video
FRAMES_PER_ANN = VIDEO_FPS / ANN_FPS   # ≈ 29.97


class WildtrackDataset:
    """Loader for the Wildtrack dataset (7 cameras, 2 fps GT annotations)."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self._ann_dir = self.root / "annotations_positions"
        if not self._ann_dir.is_dir():
            raise FileNotFoundError(
                f"Wildtrack annotations directory not found: {self._ann_dir}"
            )
        self._ann_files: list[Path] = sorted(self._ann_dir.glob("*.json"))
        if not self._ann_files:
            raise FileNotFoundError(
                f"No annotation JSON files found in {self._ann_dir}"
            )

    # ------------------------------------------------------------------
    # Video sources
    # ------------------------------------------------------------------

    def get_cam_ids(self) -> list[int]:
        """Return cam IDs (0-based) for which a video file actually exists."""
        return [
            i for i, name in enumerate(CAM_NAMES)
            if (self.root / f"{name}.mp4").exists()
        ]

    def get_video_paths(self, cam_ids: list[int] | None = None) -> list[Path]:
        """Return sorted list of existing video Paths.

        cam_ids: subset of 0-based IDs to include (default: all available).
        """
        available = self.get_cam_ids()
        selected = [i for i in (cam_ids or available) if i in available]
        if not selected:
            raise FileNotFoundError(
                f"No cam*.mp4 files found under {self.root}"
            )
        return [self.root / f"{CAM_NAMES[i]}.mp4" for i in selected]

    def get_video_uris(self, cam_ids: list[int] | None = None) -> list[str]:
        """Return file:// URIs suitable for passing to the pipeline."""
        return [f"file://{p.resolve()}" for p in self.get_video_paths(cam_ids)]

    # ------------------------------------------------------------------
    # Ground-truth annotations
    # ------------------------------------------------------------------

    def load_gt(
        self,
        cam_id: int,
        max_seconds: float | None = None,
    ) -> pd.DataFrame:
        """Load GT annotations for one camera (0-based cam_id).

        Converts annotation timestamps to the nearest video frame number.

        Returns a DataFrame with columns:
            frame (int), person_id (int),
            left (float), top (float), width (float), height (float)
        """
        rows = []
        for ann_file in self._ann_files:
            ann_idx = int(ann_file.stem) // 5
            video_frame = round(ann_idx * FRAMES_PER_ANN)

            if max_seconds is not None and ann_idx / ANN_FPS > max_seconds:
                break

            try:
                entries = json.loads(ann_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            for entry in entries:
                view = entry["views"][cam_id]
                xmin, xmax = view["xmin"], view["xmax"]
                ymin, ymax = view["ymin"], view["ymax"]
                if xmin < 0 or ymin < 0:   # not visible in this camera
                    continue
                rows.append({
                    "frame":     video_frame,
                    "person_id": int(entry["personID"]),
                    "left":      float(xmin),
                    "top":       float(ymin),
                    "width":     float(max(1, xmax - xmin)),
                    "height":    float(max(1, ymax - ymin)),
                })

        return pd.DataFrame(
            rows,
            columns=["frame", "person_id", "left", "top", "width", "height"],
        )

    def load_all_gt(
        self,
        cam_ids: list[int] | None = None,
        max_seconds: float | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Load GT for all selected cameras. Returns {cam_id: DataFrame}."""
        selected = cam_ids if cam_ids is not None else self.get_cam_ids()
        return {c: self.load_gt(c, max_seconds=max_seconds) for c in selected}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def annotation_count(self) -> int:
        """Total number of annotation frames (= JSON files)."""
        return len(self._ann_files)

    @property
    def annotated_duration_seconds(self) -> float:
        """Wall-clock seconds covered by annotations."""
        return self.annotation_count / ANN_FPS
