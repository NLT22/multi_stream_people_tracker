"""Micro-batch cross-camera fusion engine (production-shaped Global ID service).

This is the architecture every real-world MTMC system uses (NVIDIA Metropolis
MTMC/RTLS, AICity winners): per-camera perception runs every frame on the GPU and
emits tracklet embeddings; a SEPARATE fusion stage clusters those embeddings
across cameras on a MICRO-BATCH cadence (sub-second to a few seconds) to assign
stable Global IDs. It is NOT per-frame online gallery matching — that approach
cannot run realtime and is not how the field builds these systems.

`MicroBatchFusion` is a stateful streaming engine:

    fusion = MicroBatchFusion(interval_frames=125, geo_weight=0.25, ...)
    for each tracklet update from perception:
        fusion.ingest_tracklet(tracklet_id, cam_id, local_track_id, global_id,
                               start_frame, end_frame, num_detections,
                               num_embeddings, embedding)
    events = fusion.step(current_frame_no)   # fuses when an interval boundary passes
    gid = fusion.resolve(raw_global_id)      # current stable Global ID

It reuses the exact clustering primitives validated offline in
`src.eval.offline_merge`, so a window decision here is identical to one nearline
window. The difference is purely streaming: decisions are made incrementally on
a fixed cadence and are sticky (once two Global IDs merge, they stay merged),
which is what makes it a realtime service rather than a batch pass.

See `src/eval/online_fusion.py` for a CLI driver that replays exported
predictions through this engine (used to validate IDF1 parity with nearline).
"""

from __future__ import annotations

import numpy as np

from src.eval import offline_merge


def _compress_remap(remap: dict[int, int]) -> dict[int, int]:
    """Path-compress a source->target remap chain in place."""
    def find(gid: int) -> int:
        nxt = remap.get(gid, gid)
        if nxt != gid:
            remap[gid] = find(nxt)
        return remap.get(gid, gid)

    for gid in list(remap):
        find(gid)
    return remap


class MicroBatchFusion:
    """Streaming cross-camera Global ID fusion on a fixed micro-batch cadence.

    Parameters mirror `src.eval.nearline_merge` so behaviour is comparable:

    interval_frames   Decision cadence. Every `interval_frames` the engine runs
                      one clustering pass over all tracklets seen so far. At
                      25 FPS, 125 = a decision every 5s. This is the realtime
                      latency knob (smaller = fresher Global IDs but more
                      fragmentation, exactly the Metropolis RTLS tradeoff).
    delay_frames      Latency added to the event timestamp before a remap is
                      considered "applied" (models fusion compute + transport).
    threshold/margin  Embedding cosine gate and runner-up margin for a merge.
    geo_weight        Blend weight for ground-plane geometry score (0 = appearance
                      only). `geometry_points` must be provided to use it.
    """

    def __init__(
        self,
        interval_frames: int = 125,
        delay_frames: int = 0,
        threshold: float = 0.55,
        margin: float = 0.02,
        min_gid_embeddings: int = 4,
        min_tracklet_detections: int = 6,
        max_candidates_per_gid: int = 5,
        temporal_tolerance: int = 0,
        geo_weight: float = 0.0,
        geo_min_overlaps: int = 8,
        geometry_points: dict | None = None,
        geo_mode: str = "cooccur",
    ) -> None:
        self.interval_frames = max(1, interval_frames)
        self.delay_frames = max(0, delay_frames)
        self.threshold = threshold
        self.margin = margin
        self.min_gid_embeddings = min_gid_embeddings
        self.min_tracklet_detections = min_tracklet_detections
        self.max_candidates_per_gid = max_candidates_per_gid
        self.temporal_tolerance = temporal_tolerance
        self.geo_weight = max(0.0, min(1.0, geo_weight))
        self.geo_min_overlaps = max(1, geo_min_overlaps)
        self.geometry_points = geometry_points or {}
        self.geo_mode = geo_mode

        # Rolling perception evidence, keyed by tracklet_id.
        self._tracklets: dict[int, dict] = {}
        self._emb: dict[int, np.ndarray] = {}

        # Fusion state.
        self._remap: dict[int, int] = {}
        self._accepted_seen: set[tuple[int, int]] = set()
        self._events: list[dict] = []
        self._next_boundary = self.interval_frames

    # ------------------------------------------------------------------ ingest
    def ingest_tracklet(
        self,
        tracklet_id: int,
        cam_id: int,
        local_track_id: int,
        global_id: int,
        start_frame: int,
        end_frame: int,
        num_detections: int,
        num_embeddings: int,
        embedding: np.ndarray | None,
    ) -> None:
        """Upsert one tracklet's running summary and mean embedding."""
        self._tracklets[tracklet_id] = {
            "tracklet_id": tracklet_id,
            "cam_id": cam_id,
            "local_track_id": local_track_id,
            "global_id": global_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_detections": num_detections,
            "num_embeddings": num_embeddings,
        }
        if embedding is not None:
            self._emb[tracklet_id] = np.asarray(embedding, dtype=np.float32)

    # -------------------------------------------------------------------- step
    def step(self, frame_no: int) -> list[dict]:
        """Run fusion for every interval boundary at or before `frame_no`.

        Returns the list of new remap events emitted this call.
        """
        new_events: list[dict] = []
        while frame_no >= self._next_boundary:
            new_events.extend(self._fuse_window(self._next_boundary))
            self._next_boundary += self.interval_frames
        return new_events

    def flush(self, frame_no: int) -> list[dict]:
        """Final fusion pass to capture tail tracklets after the last frame."""
        return self._fuse_window(frame_no)

    def _fuse_window(self, boundary: int) -> list[dict]:
        visible = [
            row for row in self._tracklets.values()
            if row["end_frame"] <= boundary
        ]
        if len(visible) < 2:
            return []

        gids, vectors, intervals = offline_merge._build_gid_summaries(
            visible,
            self._emb,
            min_gid_embeddings=self.min_gid_embeddings,
            min_tracklet_detections=self.min_tracklet_detections,
        )
        pairs = offline_merge._candidate_pairs(
            gids,
            vectors,
            threshold=self.threshold,
            margin=self.margin,
            max_candidates_per_gid=self.max_candidates_per_gid,
            intervals=intervals,
            geometry_points=self.geometry_points,
            geo_weight=self.geo_weight,
            geo_min_overlaps=self.geo_min_overlaps,
            geo_mode=self.geo_mode,
        )
        _, accepted = offline_merge._merge_map(
            gids,
            pairs,
            intervals,
            temporal_tolerance=self.temporal_tolerance,
        )

        new_events: list[dict] = []
        for source_gid, target_gid, score in accepted:
            source_gid = self._remap.get(source_gid, source_gid)
            target_gid = self._remap.get(target_gid, target_gid)
            if source_gid == target_gid:
                continue
            source_gid, target_gid = (
                max(source_gid, target_gid),
                min(source_gid, target_gid),
            )
            key = (source_gid, target_gid)
            if key in self._accepted_seen:
                continue
            self._accepted_seen.add(key)
            self._remap[source_gid] = target_gid
            _compress_remap(self._remap)
            event = {
                "event_frame": boundary + self.delay_frames,
                "window_end_frame": boundary,
                "source_global_id": source_gid,
                "target_global_id": target_gid,
                "score": round(float(score), 6),
            }
            self._events.append(event)
            new_events.append(event)
        return new_events

    # --------------------------------------------------------------- recluster
    def recluster(self) -> dict[int, int]:
        """Stateless full re-clustering over ALL ingested tracklets.

        This matches how production MTMC fusion works (NVIDIA Metropolis:
        "clustering happens for every micro-batch") — each call re-derives the
        Global-ID grouping from scratch over all evidence accumulated so far,
        rather than accreting sticky pairwise merges. With full evidence (end of
        stream) this reproduces the offline `offline_merge` result exactly, which
        the sticky incremental `step()` path can miss due to streaming-order and
        partial-embedding sensitivity.

        Returns the full source->target remap (path-compressed).
        """
        tracklets = list(self._tracklets.values())
        if len(tracklets) < 2:
            return {}
        gids, vectors, intervals = offline_merge._build_gid_summaries(
            tracklets,
            self._emb,
            min_gid_embeddings=self.min_gid_embeddings,
            min_tracklet_detections=self.min_tracklet_detections,
        )
        pairs = offline_merge._candidate_pairs(
            gids,
            vectors,
            threshold=self.threshold,
            margin=self.margin,
            max_candidates_per_gid=self.max_candidates_per_gid,
            intervals=intervals,
            geometry_points=self.geometry_points,
            geo_weight=self.geo_weight,
            geo_min_overlaps=self.geo_min_overlaps,
            geo_mode=self.geo_mode,
        )
        remap, _ = offline_merge._merge_map(
            gids, pairs, intervals, temporal_tolerance=self.temporal_tolerance,
        )
        # Drop identity entries so callers can treat the dict as merges-only.
        self._remap = {g: t for g, t in remap.items() if g != t}
        return self._remap

    # ----------------------------------------------------------------- resolve
    def resolve(self, global_id: int) -> int:
        """Current stable Global ID for a raw per-camera global id."""
        seen = global_id
        while seen in self._remap and self._remap[seen] != seen:
            seen = self._remap[seen]
        return seen

    def final_remap(self) -> dict[int, int]:
        """Full source->target table after all events (path-compressed)."""
        remap: dict[int, int] = {}
        for event in self._events:
            source = int(event["source_global_id"])
            target = int(event["target_global_id"])
            remap[source] = min(target, remap.get(source, target))
            _compress_remap(remap)
        return _compress_remap(remap)

    @property
    def events(self) -> list[dict]:
        return self._events
