"""Per-camera tracklet memory: one rolling embedding history per (cam, track).

Owns the `(src, track_id) -> tracklet` dict and the update / sampling / averaging
logic, separated from the DeepStream probe so it can be unit-tested. The probe
holds a TrackletStore and aliases `self._tracklets = store.tracklets` for reads.
"""

from __future__ import annotations

import numpy as np

from src.reid.matching import _mean_embedding


class TrackletStore:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.tracklets: dict[tuple, dict] = {}   # (src, track_id) -> state
        self._next_tracklet_id = 1

    def update(self, track_key: tuple, src: int, track_id: int,
               embedding: list[float], frame_count: int,
               initial_gid: int | None = None, use_fusion: bool = False,
               quality_ok: bool = True, foot_world=None) -> dict:
        """Create/refresh the tracklet for `track_key`; return it."""
        tracklet = self.tracklets.get(track_key)
        if tracklet is None:
            tracklet = {
                "src": src,
                "track_id": track_id,
                "embeddings": [],
                "age": 0,
                "gid": initial_gid,
                "last_embedding_frame": -self._cfg.tracklet_embedding_interval,
                "foot_world": None,
                # Fields below feed the micro-batch fusion engine.
                "tracklet_id": self._next_tracklet_id,
                "start_frame": frame_count,
                "end_frame": frame_count,
                "num_detections": 0,
                # Running (uncapped) embedding sum+count for the fusion engine —
                # the rolling "embeddings" list is capped for matching stability,
                # but fusion wants the full-tracklet mean like the exporter.
                "fusion_emb_sum": None,
                "fusion_emb_count": 0,
            }
            self.tracklets[track_key] = tracklet
            self._next_tracklet_id += 1
        tracklet["age"] = 0
        tracklet["src"] = src
        tracklet["track_id"] = track_id
        tracklet["end_frame"] = frame_count
        tracklet["num_detections"] += 1
        if foot_world is not None:
            tracklet["foot_world"] = foot_world
        should_sample = self.should_sample(tracklet, frame_count)
        tracklet["sampled_this_frame"] = bool(embedding and quality_ok and should_sample)
        if tracklet["sampled_this_frame"]:
            tracklet["embeddings"].append(embedding)
            tracklet["last_embedding_frame"] = frame_count
            if len(tracklet["embeddings"]) > self._cfg.tracklet_max_embeddings:
                del tracklet["embeddings"][:-self._cfg.tracklet_max_embeddings]
            if use_fusion:
                v = np.asarray(embedding, dtype=np.float32)
                if tracklet["fusion_emb_sum"] is None:
                    tracklet["fusion_emb_sum"] = v.copy()
                else:
                    tracklet["fusion_emb_sum"] += v
                tracklet["fusion_emb_count"] += 1
        return tracklet

    def should_sample(self, tracklet: dict, frame_count: int) -> bool:
        if len(tracklet["embeddings"]) < self._cfg.tracklet_min_embeddings_for_match:
            return True
        interval = max(1, self._cfg.tracklet_embedding_interval)
        return frame_count - tracklet.get("last_embedding_frame", -interval) >= interval

    def tracklet_embedding(self, tracklet: dict,
                           fallback: list[float] | None = None) -> list[float]:
        """Mean of the rolling embeddings, or `fallback` if too few sampled."""
        if not self._cfg.use_tracklet_embedding:
            return fallback or []
        embeddings = tracklet.get("embeddings", [])
        if len(embeddings) < self._cfg.tracklet_min_embeddings_for_match:
            return fallback or []
        return _mean_embedding(embeddings) or (fallback or [])
