"""
Prediction exporter for offline evaluation.

CrossCameraGalleryProbe calls PredictionExporter.record() once per tracked
person per frame. The exporter writes one CSV per camera:

    <output_dir>/cam_<N>_predictions.csv

CSV columns:
    frame_no_cam, cam_id, local_track_id, global_id, left, top, width, height

For offline / nearline MTMC association, the exporter also writes compact
tracklet-level embedding summaries:

    <output_dir>/tracklets.csv
    <output_dir>/tracklet_embeddings.npz
"""

from __future__ import annotations

import csv
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
        self._tracklets: dict[tuple[int, int, int], dict] = {}

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
        embedding: list[float] | None = None,
    ) -> None:
        """Append one detection row to the CSV for cam_id."""
        writer = self._get_writer(cam_id)
        gid = global_id if global_id is not None else -1
        writer.writerow({
            "frame_no_cam": frame_no,
            "cam_id": cam_id,
            "local_track_id": local_track_id,
            "global_id": gid,
            "left": round(left, 2),
            "top": round(top, 2),
            "width": round(width, 2),
            "height": round(height, 2),
        })
        self._update_tracklet_summary(
            frame_no, cam_id, local_track_id, gid, width, height, embedding)

    def close(self) -> None:
        """Flush and close all open CSV files."""
        self._write_tracklet_summaries()
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

    def _update_tracklet_summary(
        self,
        frame_no: int,
        cam_id: int,
        local_track_id: int,
        global_id: int,
        width: float,
        height: float,
        embedding: list[float] | None,
    ) -> None:
        key = (cam_id, local_track_id, global_id)
        entry = self._tracklets.setdefault(key, {
            "start_frame": frame_no,
            "end_frame": frame_no,
            "num_detections": 0,
            "sum_width": 0.0,
            "sum_height": 0.0,
            "sum_area": 0.0,
            "num_embeddings": 0,
            "sum_embedding": None,
        })
        entry["start_frame"] = min(entry["start_frame"], frame_no)
        entry["end_frame"] = max(entry["end_frame"], frame_no)
        entry["num_detections"] += 1
        entry["sum_width"] += float(width)
        entry["sum_height"] += float(height)
        entry["sum_area"] += float(width) * float(height)

        if embedding:
            if entry["sum_embedding"] is None:
                entry["sum_embedding"] = [0.0] * len(embedding)
            if len(entry["sum_embedding"]) == len(embedding):
                for i, value in enumerate(embedding):
                    entry["sum_embedding"][i] += float(value)
                entry["num_embeddings"] += 1

    def _write_tracklet_summaries(self) -> None:
        if not self._tracklets:
            return

        try:
            import numpy as np
        except ImportError:
            print("[eval export] numpy not found; skipping tracklet_embeddings.npz")
            np = None

        rows = []
        embeddings = []
        fieldnames = [
            "tracklet_id", "cam_id", "local_track_id", "global_id",
            "start_frame", "end_frame", "num_detections", "num_embeddings",
            "mean_width", "mean_height", "mean_area",
        ]

        for idx, (key, entry) in enumerate(sorted(self._tracklets.items())):
            cam_id, local_track_id, global_id = key
            num_det = max(1, entry["num_detections"])
            rows.append({
                "tracklet_id": idx,
                "cam_id": cam_id,
                "local_track_id": local_track_id,
                "global_id": global_id,
                "start_frame": entry["start_frame"],
                "end_frame": entry["end_frame"],
                "num_detections": entry["num_detections"],
                "num_embeddings": entry["num_embeddings"],
                "mean_width": round(entry["sum_width"] / num_det, 3),
                "mean_height": round(entry["sum_height"] / num_det, 3),
                "mean_area": round(entry["sum_area"] / num_det, 3),
            })

            if np is not None and entry["sum_embedding"] is not None:
                emb = np.asarray(entry["sum_embedding"], dtype=np.float32)
                emb = emb / max(1, entry["num_embeddings"])
                norm = np.linalg.norm(emb)
                if norm > 0.0:
                    emb = emb / norm
                embeddings.append((idx, emb))

        csv_path = self._output_dir / "tracklets.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if np is None or not embeddings:
            return

        tracklet_ids = np.asarray([idx for idx, _ in embeddings], dtype=np.int64)
        vectors = np.stack([emb for _, emb in embeddings]).astype(np.float32)
        np.savez_compressed(
            self._output_dir / "tracklet_embeddings.npz",
            tracklet_ids=tracklet_ids,
            embeddings=vectors,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
