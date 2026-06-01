"""
Prediction exporter for offline evaluation.

CrossCameraGalleryProbe calls PredictionExporter.record() once per tracked
person per frame. The exporter writes one CSV per camera:

    <output_dir>/cam_<N>_predictions.csv

CSV columns:
    frame_no_cam, cam_id, local_track_id, global_id, left, top, width, height
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


_FIELDNAMES = [
    "frame_no_cam", "cam_id", "local_track_id", "global_id",
    "left", "top", "width", "height",
]


class PredictionExporter:
    """Writes per-camera prediction CSVs during pipeline execution."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-open: one file + writer per cam_id encountered.
        self._files: dict[int, object] = {}
        self._writers: dict[int, csv.DictWriter] = {}

    def record(
        self,
        frame_no: int,
        cam_id: int,
        local_track_id: int,
        global_id: int | None,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> None:
        """Append one detection row to the CSV for cam_id."""
        writer = self._get_writer(cam_id)
        writer.writerow({
            "frame_no_cam": frame_no,
            "cam_id": cam_id,
            "local_track_id": local_track_id,
            "global_id": global_id if global_id is not None else -1,
            "left": round(left, 2),
            "top": round(top, 2),
            "width": round(width, 2),
            "height": round(height, 2),
        })

    def close(self) -> None:
        """Flush and close all open CSV files."""
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()

    def _get_writer(self, cam_id: int) -> csv.DictWriter:
        if cam_id not in self._writers:
            path = self._output_dir / f"cam_{cam_id}_predictions.csv"
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            self._files[cam_id] = f
            self._writers[cam_id] = writer
        return self._writers[cam_id]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
