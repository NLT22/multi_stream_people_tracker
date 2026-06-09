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

    def __init__(self, output_dir: str, delay_frames: int = 0) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-open: one file + writer per cam_id encountered.
        self._files: dict[int, object] = {}
        self._writers: dict[int, csv.DictWriter] = {}
        self._tracklets: dict[tuple[int, int, int], dict] = {}

        # Near-realtime delayed-flush buffer. When delay_frames > 0 (micro-batch
        # fusion path), rows are held for `delay_frames` and the latest Global-ID
        # remap is applied at flush time, so each frame gets a few seconds of
        # cross-camera merge correction before it is written — the production
        # "near-realtime authoritative Global ID" semantics. When delay_frames
        # == 0 the buffer is flushed on every tick (behaviour unchanged: rows
        # are written in arrival order with no remap).
        self._delay_frames = max(0, int(delay_frames))
        self._buffer: dict[int, list[dict]] = {}   # cam_id -> pending row dicts
        self._remap: dict[int, int] = {}
        self._max_frame = -1

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
        """Buffer one detection row for cam_id (raw Global ID)."""
        gid = global_id if global_id is not None else -1
        self._max_frame = max(self._max_frame, frame_no)
        self._buffer.setdefault(cam_id, []).append({
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

    def set_remap(self, remap: dict[int, int]) -> None:
        """Provide the latest Global-ID remap applied at flush time."""
        self._remap = remap or {}

    def tick(self, current_frame: int, remap: dict[int, int] | None = None) -> None:
        """Flush rows older than the delay window, applying the current remap.

        Called once per batch by the gallery probe. Rows with
        frame_no <= current_frame - delay_frames are written now (with the
        latest remap resolved); newer rows stay buffered so a later micro-batch
        merge can still correct them.
        """
        if remap is not None:
            self._remap = remap
        safe_frame = current_frame - self._delay_frames
        self._flush(safe_frame)

    def close(self) -> None:
        """Flush all remaining rows with the final remap, then write summaries."""
        self._flush(None)               # None = flush everything
        self._write_tracklet_summaries()
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()

    def _resolve(self, gid: int) -> int:
        """Follow the remap chain to the final stable Global ID."""
        if gid < 0 or not self._remap:
            return gid
        seen = gid
        # final_remap() is path-compressed, but guard against cycles anyway.
        for _ in range(64):
            nxt = self._remap.get(seen, seen)
            if nxt == seen:
                break
            seen = nxt
        return seen

    def _flush(self, safe_frame: int | None) -> None:
        """Write buffered rows with frame_no <= safe_frame (or all if None)."""
        for cam_id, rows in self._buffer.items():
            if not rows:
                continue
            writer = self._get_writer(cam_id)
            kept: list[dict] = []
            for row in rows:
                if safe_frame is None or row["frame_no_cam"] <= safe_frame:
                    out = dict(row)
                    out["global_id"] = self._resolve(row["global_id"])
                    writer.writerow(out)
                else:
                    kept.append(row)
            self._buffer[cam_id] = kept

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
            global_id = self._resolve(global_id)
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
