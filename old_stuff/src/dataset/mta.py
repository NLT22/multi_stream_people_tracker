"""
MTA (Multi-Target multi-cAmera) dataset loader.

Folder layout expected:
    <root>/<split>/cam_<N>/cam_<N>.mp4
    <root>/<split>/cam_<N>/coords_fib_cam_<N>.csv

GT CSV columns: frame_no_cam, person_id,
                x_top_left_BB, y_top_left_BB, x_bottom_right_BB, y_bottom_right_BB
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class MtaDataset:
    """Loader for one split (train / test) of an MTA_ext_short folder."""

    CAM_IDS = list(range(6))  # cam_0 … cam_5

    def __init__(self, root: str, split: str = "test") -> None:
        self.root = Path(root)
        self.split = split
        self._split_dir = self.root / split
        if not self._split_dir.is_dir():
            raise FileNotFoundError(
                f"MTA split directory not found: {self._split_dir}"
            )

    # ------------------------------------------------------------------
    # Video sources
    # ------------------------------------------------------------------

    def get_video_paths(self) -> list[Path]:
        """Return sorted list of existing video Paths (cam_0 first)."""
        paths = []
        for cam_id in self.CAM_IDS:
            p = self._split_dir / f"cam_{cam_id}" / f"cam_{cam_id}.mp4"
            if p.exists():
                paths.append(p)
        if not paths:
            raise FileNotFoundError(
                f"No cam_*.mp4 files found under {self._split_dir}"
            )
        return paths

    def get_video_uris(self) -> list[str]:
        """Return file:// URIs suitable for passing to the pipeline."""
        return [f"file://{p.resolve()}" for p in self.get_video_paths()]

    def get_cam_ids(self) -> list[int]:
        """Return cam IDs for which a video file actually exists."""
        return [
            cam_id for cam_id in self.CAM_IDS
            if (self._split_dir / f"cam_{cam_id}" / f"cam_{cam_id}.mp4").exists()
        ]

    # ------------------------------------------------------------------
    # Ground-truth annotations
    # ------------------------------------------------------------------

    def load_gt(self, cam_id: int) -> pd.DataFrame:
        """Load GT for one camera.

        Returns a DataFrame with columns:
            frame (int), person_id (int),
            left (float), top (float), width (float), height (float)
        """
        csv_path = (
            self._split_dir / f"cam_{cam_id}" / f"coords_fib_cam_{cam_id}.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"GT CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # Rename to canonical column names used by eval pipeline.
        df = df.rename(columns={"frame_no_cam": "frame"})
        df["left"] = df["x_top_left_BB"].astype(float)
        df["top"] = df["y_top_left_BB"].astype(float)
        df["width"] = (df["x_bottom_right_BB"] - df["x_top_left_BB"]).astype(float)
        df["height"] = (df["y_bottom_right_BB"] - df["y_top_left_BB"]).astype(float)
        return df[["frame", "person_id", "left", "top", "width", "height"]]

    def load_all_gt(self) -> dict[int, pd.DataFrame]:
        """Load GT for all available cameras. Returns {cam_id: DataFrame}."""
        return {cam_id: self.load_gt(cam_id) for cam_id in self.get_cam_ids()}
