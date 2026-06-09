"""
Cross-camera ReID gallery: tracker-embedding extraction + Global-ID assignment.

Contains the metadata probes used by src/main.py:
  - SourceIdCollectorProbe (pre-tiler): reads source_id + ReID embeddings
  - CrossCameraGalleryProbe (post-tiler): matches embeddings -> stable Global ID

Tuning lives in a typed ReIDConfig (src/reid/config.py), built from CLI args
via configure_from_args(args) and passed into the probe as self._cfg.
"""

import math
import traceback

import numpy as np

import pyservicemaker as psm


from src.pipeline.model_utils import (
    infer_source_id_from_tiled_box,
    set_object_label,
)
from src.reid.visualization import style_object_by_id
from src.reid.geometry import GroundPlaneGeometry
from src.reid import quality
from src.reid.config import ReIDConfig

# Tuning constants now live in ReIDConfig (src/reid/config.py).

from src.reid.matching import (  # noqa: F401  (re-exported for callers)
    _cosine_similarity,
    _mean_embedding,
    max_weight_assignment,
)

from src.reid.metadata import SourceIdCollectorProbe  # noqa: F401


class CrossCameraGalleryProbe(psm.BatchMetadataOperator):
    """
    Post-tiler: tiled canvas coordinates — draw labels here.

    Maintains a cross-camera gallery:
      gallery:        global_id → {"prototypes": [...], "age": int}
                      if self._cfg.gallery_max_prototypes == 0:
                      global_id → {"embedding": [...], "age": int}
      track_to_gid:   (src, track_id) → global_id   ← stable while tracker holds the ID

    For each detected person each frame:
      1. Update a short local tracklet for (src, track_id)
      2. If (src, track_id) already mapped → reuse global_id, update embedding
      3. Else → compare tracklet embedding against all gallery prototypes
             match ≥ threshold → reuse that global_id
             no match         → new global_id
      4. Draw "G#{global_id} Cam{src}#{track_id}" on screen

    WHY track_to_gid works:
      Within one camera a tracker ID is stable while the person is visible.
      Cross-camera: different src → different key → forces embedding match,
      which is how global IDs link across cameras.
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int,
                 debug_similarity: bool = False,
                 use_hungarian_assignment: bool = True,
                 enforce_unique_per_stream: bool = True,
                 pretiler: bool = False,
                 extract_embeddings: bool = False,
                 trajectory_visualizer=None,
                 exporter=None,
                 frame_numbers: dict | None = None,
                 frame_sizes: dict | None = None,
                 geometry: "GroundPlaneGeometry | None" = None,
                 config: "ReIDConfig | None" = None):
        super().__init__()
        self._cfg = config if config is not None else ReIDConfig()
        self._id_map = id_map
        self._embeddings = embeddings
        self._frame_numbers = frame_numbers  # source_id → frame_number from pre-tiler
        self._frame_sizes = frame_sizes      # source_id → source frame size from pre-tiler
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        # pretiler=True: attach this probe to the tracker (before the tiler), so
        # frame_meta.source_id is exact — no geometric guessing from tile coords.
        # extract_embeddings=True: read the ReID vector straight off obj_meta
        # here too, so no separate SourceIdCollectorProbe / shared dict is needed.
        self._pretiler = pretiler
        self._extract_embeddings = extract_embeddings
        self._gallery: dict[int, dict] = {}      # global_id → gallery entry
        self._track_to_gid: dict[tuple, int] = {}  # (src, track_id) → global_id
        self._tracklets: dict[tuple, dict] = {}   # (src, track_id) → state
        self._next_gid = 1
        self._next_tracklet_id = 1
        self._frame_count = 0

        # Micro-batch cross-camera fusion (opt-in). When enabled, a live
        # MicroBatchFusion engine periodically clusters tracklet embeddings
        # across cameras and remaps raw gids -> stable Global IDs at draw/export
        # time. See module constants above.
        self._use_fusion = self._cfg.use_micro_batch_fusion
        self._fusion = None
        self._fusion_remap: dict[int, int] = {}
        self._fusion_tid_by_key: dict[tuple, int] = {}  # (cam,local,gid) -> tracklet_id
        # gid -> frame -> [(cam_id, world_x_mm, world_y_mm)] for geometry scoring,
        # accumulated live (mirrors offline_merge._load_geometry_points).
        self._fusion_geo_points: dict[int, dict[int, list]] = {}
        if self._use_fusion:
            from src.reid.micro_batch_fusion import MicroBatchFusion
            geo_on = geometry is not None
            self._fusion = MicroBatchFusion(
                interval_frames=self._cfg.micro_batch_fusion_interval,
                threshold=self._cfg.micro_batch_fusion_threshold,
                margin=self._cfg.micro_batch_fusion_margin,
                min_gid_embeddings=self._cfg.micro_batch_fusion_min_gid_embeddings,
                min_tracklet_detections=self._cfg.micro_batch_fusion_min_tracklet_detections,
                geo_weight=self._cfg.micro_batch_fusion_geo_weight if geo_on else 0.0,
                geo_min_overlaps=self._cfg.micro_batch_fusion_geo_min_overlaps,
                geometry_points=self._fusion_geo_points,
            )
            print(f"[reid] micro-batch fusion ON: interval="
                  f"{self._cfg.micro_batch_fusion_interval}f thr={self._cfg.micro_batch_fusion_threshold} "
                  f"geo={'on' if geo_on else 'off'}")
        self._debug_similarity = debug_similarity
        self._use_hungarian_assignment = use_hungarian_assignment
        self._enforce_unique_per_stream = enforce_unique_per_stream
        self._trajectory_visualizer = trajectory_visualizer
        self._exporter = exporter
        # Optional ground-plane geometry for calibration-assisted matching.
        # When provided, foot positions in world-mm are stored per row and used
        # to blend a geometric similarity score into the ReID score for
        # cross-camera candidate pairs.
        self._geometry: GroundPlaneGeometry | None = geometry

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            print("[reid ERROR] CrossCameraGalleryProbe failed:")
            traceback.print_exc()

    def _handle_metadata(self, batch_meta):
        self._frame_count += 1
        log = self._frame_count % 60 == 0

        # Expire stale gallery entries
        stale = [gid for gid, v in self._gallery.items()
                 if v["age"] > self._cfg.gallery_max_age]
        for gid in stale:
            del self._gallery[gid]
            # Also clean stale track mappings pointing to this gid
            self._track_to_gid = {k: v for k, v in self._track_to_gid.items()
                                   if v != gid}
            for tracklet in self._tracklets.values():
                if tracklet.get("gid") == gid:
                    tracklet["gid"] = None

        stale_tracks = [
            key for key, tracklet in self._tracklets.items()
            if tracklet["age"] > self._cfg.tracklet_max_age
        ]
        for key in stale_tracks:
            del self._tracklets[key]
            self._track_to_gid.pop(key, None)

        # Micro-batch cross-camera fusion: run once per batch on the cadence,
        # over tracklet evidence accumulated so far. Refreshes self._fusion_remap
        # which is applied to displayed/exported Global IDs below.
        if (
            self._use_fusion
            and self._frame_count % self._cfg.micro_batch_fusion_interval == 0
        ):
            self._run_micro_batch_fusion()

        for frame_meta in batch_meta.frame_items:
            rows = []
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue

                oid = obj_meta.object_id
                if self._pretiler:
                    # Pre-tiler: source_id is exact.
                    src = frame_meta.source_id
                    if self._extract_embeddings:
                        embedding = SourceIdCollectorProbe._extract_embedding(
                            obj_meta)[0]
                    else:
                        embedding = self._embeddings.get((src, oid), [])
                else:
                    # Post-tiler: look up source_id from SourceIdCollectorProbe's
                    # id_map (filled pre-tiler where source_id is exact).
                    # Fall back to geometric guessing only if the object wasn't
                    # seen by the collector (e.g. appeared between the two probes).
                    src = self._id_map.get(oid)
                    if src is None:
                        src = infer_source_id_from_tiled_box(
                            obj_meta.rect_params, self._tile_w, self._tile_h,
                            self._cols, self._num_sources)
                    embedding = self._embeddings.get((src, oid), [])
                track_key = (src, oid)
                rect = self._local_rect(obj_meta.rect_params, src, frame_meta)
                foot_world = None
                if self._geometry is not None:
                    # cam_id in calibration is 1-based (same as source_id + 1
                    # for MMP).  We store foot in world-mm for geo blending.
                    foot_world = self._geometry.bbox_foot(
                        src + 1,   # source_id is 0-based; cam_id is 1-based
                        rect["left"], rect["top"],
                        rect["width"], rect["height"],
                    )
                rows.append({
                    "src": src,
                    "track_id": oid,
                    "track_key": track_key,
                    "rect": rect,
                    "embedding": [],
                    "raw_embedding": embedding,
                    "tracklet_len": 0,
                    "gid": None,
                    "previous_gid": None,
                    "foot_world": foot_world,   # (X_mm, Y_mm) or None
                })

            self._annotate_embedding_quality(rows)
            for row in rows:
                tracklet = self._update_tracklet(
                    row["track_key"], row["src"], row["track_id"],
                    row["raw_embedding"], row["embedding_quality_ok"],
                    foot_world=row.get("foot_world"))
                row["tracklet_len"] = len(tracklet["embeddings"])
                row["update_gallery"] = tracklet.get("sampled_this_frame", False)
                match_embedding = self._tracklet_embedding(
                    tracklet,
                    # Matching is allowed to be softer than memory update:
                    # a border/back-view/overlap crop may still be enough to
                    # match an existing cross-camera GID, but it must not be
                    # stored into the long-term gallery unless it passes the
                    # stricter quality gate.
                    fallback=row["raw_embedding"],
                )
                previous_gid = self._track_to_gid.get(row["track_key"])
                if previous_gid is None:
                    previous_gid = tracklet.get("gid")
                if previous_gid not in self._gallery:
                    previous_gid = None
                row["previous_gid"] = previous_gid
                row["gid"] = previous_gid
                row["had_previous_gid"] = previous_gid is not None
                row["embedding"] = match_embedding
                row["allow_new_gid"] = row["embedding_quality_ok"]
                row["identity_conflict"] = False
                row["suppress_gallery_update"] = False
                row["release_previous_gid"] = False
                # Known tracks keep their GID through low-quality frames, but
                # low-quality new tracks may only match existing IDs. They wait
                # for a cleaner crop before creating a brand-new Global ID.
                row["defer_assignment"] = (
                    previous_gid is None
                    and (not match_embedding)
                )

            if self._use_hungarian_assignment:
                if self._enforce_unique_per_stream:
                    self._mark_duplicate_known_conflicts(rows)
                self._assign_new_tracks_with_hungarian(rows, log)
            else:
                self._assign_new_tracks_greedy(rows, log)

            for row in rows:
                gid = row["gid"]
                if gid is not None:
                    self._track_to_gid[row["track_key"]] = gid
                    self._tracklets[row["track_key"]]["gid"] = gid
                    if (
                        row.get("gallery_updated") is not True
                        and not row.get("suppress_gallery_update")
                    ):
                        self._update_gallery(
                            gid,
                            row["raw_embedding"]
                            if row.get("update_gallery") else [],
                            row["src"],
                        )
                elif row.get("release_previous_gid"):
                    self._track_to_gid.pop(row["track_key"], None)
                    self._tracklets[row["track_key"]]["gid"] = None

            if (
                self._cfg.enable_global_id_merge
                and self._frame_count % self._cfg.global_id_merge_interval == 0
            ):
                self._merge_duplicate_global_ids(rows, log)

            if self._use_fusion and self._geometry is not None:
                self._accumulate_fusion_geo(rows, frame_meta)

            row_by_key = {
                (row["src"], row["track_id"]): row
                for row in rows
            }
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                if self._pretiler:
                    src = frame_meta.source_id
                else:
                    src = self._id_map.get(obj_meta.object_id)
                    if src is None:
                        src = infer_source_id_from_tiled_box(
                            obj_meta.rect_params, self._tile_w, self._tile_h,
                            self._cols, self._num_sources)
                row = row_by_key.get((src, obj_meta.object_id))
                if row is None:
                    continue
                draw_gid = self._display_gid(row["gid"])
                label = (
                    f"GID:{draw_gid if draw_gid is not None else '?'} "
                    # f"LID:{row['track_id']}"
                )
                set_object_label(obj_meta, label)
                style_object_by_id(obj_meta, draw_gid)

            if self._exporter is not None:
                for row in rows:
                    rect = row["rect"]
                    src = row["src"]
                    # In post-tiler mode frame_meta.frame_number is the batch
                    # counter (same for all sources). Use the per-source frame
                    # number collected by SourceIdCollectorProbe pre-tiler.
                    fn = (self._frame_numbers.get(src, 0)
                          if self._frame_numbers is not None
                          else frame_meta.frame_number)
                    # Export the RAW gid; the exporter applies the (delayed)
                    # fusion remap at flush time so early frames still get
                    # cross-camera merge correction within the delay window.
                    self._exporter.record(
                        frame_no=fn,
                        cam_id=src,
                        local_track_id=row["track_id"],
                        global_id=row["gid"],
                        left=rect["left"],
                        top=rect["top"],
                        width=rect["width"],
                        height=rect["height"],
                        embedding=(
                            row["raw_embedding"]
                            if (
                                row.get("update_gallery")
                                and not row.get("suppress_gallery_update")
                            )
                            else None
                        ),
                    )

            if self._trajectory_visualizer is not None:
                self._trajectory_visualizer.update_and_draw(
                    batch_meta, frame_meta, rows, self._frame_count)

        # Flush exporter rows older than the delay window, applying the latest
        # fusion remap. Once per batch (after all per-source frames handled).
        if self._exporter is not None:
            self._exporter.tick(self._frame_count, self._fusion_remap)

        # Age gallery once per batch
        for v in self._gallery.values():
            v["age"] += 1
        for tracklet in self._tracklets.values():
            tracklet["age"] += 1

        if log:
            active = len(self._gallery)
            active_tracklets = len(self._tracklets)
            print(f"[reid] frame={self._frame_count:06d}  "
                  f"gallery={active}  tracklets={active_tracklets}  "
                  f"active_gids={active}  "
                  f"total_gids_ever_assigned={self._next_gid - 1}")

    def _display_gid(self, gid: int | None) -> int | None:
        """Apply the micro-batch fusion remap to a raw gid (no-op when off)."""
        if gid is None or not self._use_fusion:
            return gid
        return self._fusion_remap.get(gid, gid)

    def _accumulate_fusion_geo(self, rows: list[dict], frame_meta) -> None:
        """Accumulate per-(gid, frame) foot positions for geometry-assisted fusion.

        Mirrors offline_merge._load_geometry_points but built live: keyed by the
        raw Global ID (same as the exporter), sampled every GEO_SAMPLE_STEP
        frames. Lets the fusion engine boost true cross-camera duplicates while
        same-camera look-alikes stay blocked by temporal conflict.
        """
        for row in rows:
            gid = row["gid"]
            foot = row.get("foot_world")
            if gid is None or gid < 0 or foot is None:
                continue
            src = row["src"]
            if self._frame_numbers is not None:
                frame_no = self._frame_numbers.get(src, self._frame_count)
            else:
                frame_no = frame_meta.frame_number
            if frame_no % self._cfg.micro_batch_fusion_geo_sample_step != 0:
                continue
            self._fusion_geo_points.setdefault(gid, {}).setdefault(
                frame_no, []).append((src, float(foot[0]), float(foot[1])))

    def _run_micro_batch_fusion(self) -> None:
        """One micro-batch pass: cluster tracklet embeddings across cameras.

        Ingests every active tracklet's mean embedding (keyed by its current raw
        Global ID) into the streaming engine, advances the engine clock to the
        current frame, and refreshes the raw->stable remap table. This is the
        in-pipeline equivalent of `src.eval.online_fusion`.

        Evidence source: the exporter's live-accumulating per-(cam, local, gid)
        tracklet summaries when available — these are the exact same vectors the
        offline pass clusters (a track that changed Global ID over time is split
        into clean single-gid segments, which is what lets a cross-camera merge
        be found). Falls back to the gallery's own end-state tracklets when not
        exporting (OSD-only live mode).
        """
        from src.reid.micro_batch_fusion import MicroBatchFusion

        # Build the tracklet evidence list (one entry per clean single-gid
        # segment), preferring the exporter's per-(cam, local, gid) summaries —
        # the exact vectors the offline pass clusters.
        records: list[tuple] = []  # (tid, cam, local, gid, start, end, ndet, nemb, emb)
        exporter_tracklets = (
            getattr(self._exporter, "_tracklets", None)
            if self._exporter is not None else None
        )
        if exporter_tracklets:
            for key, entry in exporter_tracklets.items():
                cam_id, local_track_id, gid = key
                if gid < 0:
                    continue
                tid = self._fusion_tid_by_key.setdefault(
                    key, len(self._fusion_tid_by_key))
                emb_sum = entry.get("sum_embedding")
                emb_count = entry.get("num_embeddings", 0)
                mean = None
                if emb_sum is not None and emb_count > 0:
                    v = np.asarray(emb_sum, dtype=np.float32) / emb_count
                    norm = float(np.linalg.norm(v))
                    if norm > 0.0:
                        mean = (v / norm).astype(np.float32)
                records.append((
                    tid, cam_id, local_track_id, gid,
                    entry.get("start_frame", 0),
                    entry.get("end_frame", self._frame_count),
                    entry.get("num_detections", 0), emb_count, mean,
                ))
        else:
            for (src, tid_key), tracklet in self._tracklets.items():
                raw_gid = self._track_to_gid.get((src, tid_key), tracklet.get("gid"))
                if raw_gid is None:
                    continue
                emb_sum = tracklet.get("fusion_emb_sum")
                emb_count = tracklet.get("fusion_emb_count", 0)
                mean = None
                if emb_sum is not None and emb_count > 0:
                    norm = float(np.linalg.norm(emb_sum))
                    if norm > 0.0:
                        mean = (emb_sum / norm).astype(np.float32)
                records.append((
                    tracklet["tracklet_id"], src, tid_key, raw_gid,
                    tracklet.get("start_frame", 0),
                    tracklet.get("end_frame", self._frame_count),
                    tracklet.get("num_detections", 0), emb_count, mean,
                ))

        # Replay through a FRESH engine in end_frame order with delayed step()
        # decisions — identical to the validated `src.eval.online_fusion` path,
        # which fires sticky merges at the moment evidence completes (before a
        # later same-camera look-alike can block them). A fresh engine each tick
        # re-derives the authoritative remap over all evidence-so-far.
        engine = MicroBatchFusion(
            interval_frames=self._cfg.micro_batch_fusion_interval,
            threshold=self._cfg.micro_batch_fusion_threshold,
            margin=self._cfg.micro_batch_fusion_margin,
            min_gid_embeddings=self._cfg.micro_batch_fusion_min_gid_embeddings,
            min_tracklet_detections=self._cfg.micro_batch_fusion_min_tracklet_detections,
            geo_weight=self._fusion.geo_weight,
            geo_min_overlaps=self._fusion.geo_min_overlaps,
            geometry_points=self._fusion_geo_points,
        )
        for rec in sorted(records, key=lambda r: r[5]):  # by end_frame
            tid, cam, local, gid, start, end, ndet, nemb, emb = rec
            engine.ingest_tracklet(tid, cam, local, gid, start, end, ndet, nemb, emb)
            engine.step(end)
        if records:
            engine.flush(max(r[5] for r in records))
        self._fusion_remap = engine.final_remap()
        if self._debug_similarity:
            print(f"  [fusion] frame={self._frame_count} "
                  f"tracklets={len(records)} "
                  f"merges={len(self._fusion_remap)} remap={self._fusion_remap}")

    def _find_or_create(self, embedding: list[float], src: int,
                        track_id: int, log: bool,
                        tracklet_len: int = 0,
                        previous_gid: int | None = None) -> int:
        """Match embedding against gallery; return existing or new global_id."""
        ranked = self._rank_gallery(embedding)
        best_gid = ranked[0][0] if ranked else -1
        best_score = ranked[0][1] if ranked else 0.0
        allowed, block_reason = self._is_gid_match_allowed(
            embedding, best_gid, previous_gid, ranked)

        matched = best_gid != -1 and allowed
        reason = "no_embedding" if not embedding else (
            "empty_gallery" if best_gid == -1 else "below_threshold"
        )
        if best_gid != -1 and not allowed:
            reason = block_reason
        should_log_similarity = self._debug_similarity or (log and matched)
        if should_log_similarity:
            status = "MATCH" if matched else "NEW"
            display_reason = "above_threshold" if matched else reason
            top = ", ".join(
                f"G{gid}={score:.3f}" for gid, score in ranked[:self._cfg.debug_top_k]
            ) or "none"
            print(
                f"  [Re-ID similarity] Cam{src}#{track_id} "
                f"best_gid={best_gid if best_gid != -1 else 'None'} "
                f"max_similarity={best_score:.3f} "
                f"threshold={self._cfg.similarity_threshold:.3f} "
                f"tracklet_len={tracklet_len} "
                f"previous_gid={previous_gid if previous_gid is not None else 'None'} "
                f"status={status} reason={display_reason} top{self._cfg.debug_top_k}=[{top}]"
            )

        if matched:
            if log:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{best_gid} "
                      f"(similarity={best_score:.3f})")
            return best_gid

        # If the track already has a known global ID still alive in the gallery,
        # keep it rather than minting a new one. This prevents ID explosion when
        # a track's embedding temporarily dips below threshold (occlusion, blur).
        if previous_gid is not None and previous_gid in self._gallery:
            if should_log_similarity:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{previous_gid} "
                      f"(sticky: below threshold but known track)")
            return previous_gid

        # New person
        gid = self._allocate_new_gid()
        self._gallery[gid] = self._new_gallery_entry()
        return gid

    def _mark_duplicate_known_conflicts(self, rows: list[dict]) -> None:
        """Release weaker same-stream duplicate GIDs before assignment.

        A single Global ID cannot represent two simultaneous tracks in the same
        camera. Keep the stronger holder stable and send the weaker row back
        through Hungarian assignment so it can take another existing ID or open
        a new one. This prevents same-frame duplicate GIDs from poisoning IDF1.
        """
        active: dict[tuple[int, int], dict] = {}
        for row in rows:
            gid = row["gid"]
            if gid is None:
                continue

            key = (row["src"], gid)
            existing = active.get(key)
            if existing is None:
                active[key] = row
                continue

            existing_score = self._score_gid(gid, existing["embedding"])
            row_score = self._score_gid(gid, row["embedding"])

            if self._prefer_conflict_gallery_update(
                row, row_score, existing, existing_score
            ):
                suppressed = existing
                active[key] = row
            else:
                suppressed = row

            active[key]["identity_conflict"] = True
            suppressed["identity_conflict"] = True
            suppressed["suppress_gallery_update"] = True
            suppressed["release_previous_gid"] = True
            suppressed["previous_gid"] = gid
            suppressed["gid"] = None
            if self._debug_similarity:
                print(
                    f"  [Re-ID conflict] Cam{suppressed['src']}#{suppressed['track_id']} "
                    f"duplicate_known_gid=G{gid} "
                    f"held_by=Cam{active[key]['src']}#{active[key]['track_id']} "
                    f"suppressed_gallery_update_score={self._score_gid(gid, suppressed['embedding']):.3f} "
                    f"held_score={self._score_gid(gid, active[key]['embedding']):.3f}"
                )

    @staticmethod
    def _prefer_conflict_gallery_update(candidate: dict, candidate_score: float,
                                        incumbent: dict,
                                        incumbent_score: float) -> bool:
        """Return True when candidate should be the gallery updater."""
        candidate_len = candidate.get("tracklet_len", 0)
        incumbent_len = incumbent.get("tracklet_len", 0)
        if candidate_len != incumbent_len:
            return candidate_len > incumbent_len
        if candidate_score != incumbent_score:
            return candidate_score > incumbent_score
        return candidate.get("track_id", 0) < incumbent.get("track_id", 0)

    def _assign_new_tracks_greedy(self, rows: list[dict], log: bool) -> None:
        for row in rows:
            if row["gid"] is None and not row.get("defer_assignment"):
                if row.get("allow_new_gid"):
                    row["gid"] = self._find_or_create(
                        row["embedding"], row["src"], row["track_id"], log,
                        row["tracklet_len"], row.get("previous_gid"))
                else:
                    ranked = self._rank_gallery(row["embedding"])
                    best_gid = ranked[0][0] if ranked else -1
                    allowed, _ = self._is_gid_match_allowed(
                        row["embedding"], best_gid, row.get("previous_gid"),
                        ranked)
                    if allowed:
                        row["gid"] = best_gid
                    else:
                        continue
                # Greedy fallback: once a new track is assigned, later
                # detections in the same tiled frame can match it.
                self._update_gallery(
                    row["gid"],
                    row["raw_embedding"] if row.get("update_gallery") else [],
                    row["src"],
                )
                row["gallery_updated"] = True

    def _assign_new_tracks_with_hungarian(self, rows: list[dict],
                                          log: bool) -> None:
        """
        Assign new tracks per stream with one-to-one Global ID constraints.

        Known local tracks keep their existing global ID. New tracks in the same
        stream compete for currently available global IDs plus one private
        "new ID" slot per track. This prevents the physically impossible state
        where one global ID appears twice in one camera frame, while still
        allowing different cameras to match the same global ID.
        """
        rows_by_src: dict[int, list[dict]] = {}
        occupied_by_src: dict[int, set[int]] = {}
        for row in rows:
            src = row["src"]
            if row["gid"] is None and not row.get("defer_assignment"):
                rows_by_src.setdefault(src, []).append(row)
            else:
                if row["gid"] is not None:
                    occupied_by_src.setdefault(src, set()).add(row["gid"])

        for src, new_rows in rows_by_src.items():
            occupied = occupied_by_src.setdefault(src, set())
            existing_gids = self._candidate_gids(
                exclude=occupied,
                max_count=self._cfg.global_assignment_max_candidates,
            )
            columns = [("gid", gid) for gid in existing_gids]
            columns += [("new", i) for i in range(len(new_rows))]

            weights = []
            for row in new_rows:
                # Pre-compute scores and ranked list once per row (not per cell).
                # Without this, ranked is recomputed for every (row, gid) pair
                # → O(rows × gids²) calls to _score_gid instead of O(rows × gids).
                scores_for_row = {
                    gid: self._score_gid(gid, row["embedding"])
                    for gid in existing_gids
                }
                ranked = sorted(scores_for_row.items(),
                                key=lambda item: item[1], reverse=True)
                row_weights = []
                best_reid_score = ranked[0][1] if ranked else 0.0
                for kind, value in columns:
                    if kind == "gid":
                        reid_score = scores_for_row[value]
                        assignment_score = self._assignment_score(
                            reid_score, best_reid_score, row, value)
                        allowed, _ = self._is_gid_match_allowed(
                            row["embedding"], value, row.get("previous_gid"),
                            ranked)
                        row_weights.append(assignment_score if allowed else -1.0)
                    else:
                        row_weights.append(0.0)
                weights.append(row_weights)

            assignment = max_weight_assignment(weights)
            for row_idx, col_idx in enumerate(assignment):
                row = new_rows[row_idx]
                kind, value = columns[col_idx]
                if kind == "gid":
                    gid = value
                    score = self._score_gid(gid, row["embedding"])
                    row["gid"] = gid
                    occupied.add(gid)
                    status = "MATCH"
                    reason = "hungarian"
                else:
                    if not row.get("allow_new_gid"):
                        row["gid"] = None
                        score = 0.0
                        status = "DEFER"
                        reason = row.get(
                            "embedding_quality_reason",
                            "low_quality_new_track",
                        )
                        if self._debug_similarity:
                            ranked = [
                                (gid, self._score_gid(gid, row["embedding"]))
                                for gid in existing_gids
                            ]
                            ranked.sort(key=lambda item: item[1], reverse=True)
                            top = ", ".join(
                                f"G{gid}={s:.3f}" for gid, s in ranked[:self._cfg.debug_top_k]
                            ) or "none"
                            print(
                                f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                                f"assigned=None score={score:.3f} "
                                f"threshold={self._cfg.similarity_threshold:.3f} "
                                f"tracklet_len={row['tracklet_len']} "
                                f"quality={row.get('embedding_quality_reason')} "
                                f"status={status} reason={reason} "
                                f"top{self._cfg.debug_top_k}=[{top}]"
                            )
                        continue

                    gid = self._allocate_new_gid()
                    self._gallery[gid] = self._new_gallery_entry()
                    row["gid"] = gid
                    occupied.add(gid)
                    score = self._score_gid(gid, row["embedding"])
                    status = "NEW"
                    reason = "new_slot"

                if self._debug_similarity:
                    ranked = [
                        (gid, self._score_gid(gid, row["embedding"]))
                        for gid in existing_gids
                    ]
                    ranked.sort(key=lambda item: item[1], reverse=True)
                    top = ", ".join(
                        f"G{gid}={s:.3f}" for gid, s in ranked[:self._cfg.debug_top_k]
                    ) or "none"
                    print(
                        f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                        f"assigned=G{row['gid']} score={score:.3f} "
                        f"threshold={self._cfg.similarity_threshold:.3f} "
                        f"tracklet_len={row['tracklet_len']} "
                        f"previous_gid={row.get('previous_gid') if row.get('previous_gid') is not None else 'None'} "
                        f"status={status} reason={reason} top{self._cfg.debug_top_k}=[{top}]"
                    )

            for row in new_rows:
                if row["gid"] is not None:
                    self._update_gallery(
                        row["gid"],
                        row["raw_embedding"] if row.get("update_gallery") else [],
                        row["src"],
                    )
                    row["gallery_updated"] = True

    def _merge_duplicate_global_ids(self, rows: list[dict], log: bool) -> None:
        """Merge stable duplicate Global IDs created by difficult cross views."""
        active_by_src: dict[int, set[int]] = {}
        for row in rows:
            active_by_src.setdefault(row["src"], set()).add(row["gid"])

        for row in rows:
            source_gid = row["gid"]
            if source_gid is None or source_gid not in self._gallery:
                continue
            if self._gallery[source_gid].get("age", 0) > 1:
                continue
            if row["tracklet_len"] < self._cfg.global_id_merge_min_tracklet_embeddings:
                continue
            if not row["embedding"]:
                continue

            candidate = self._best_merge_candidate(
                source_gid, row, active_by_src)
            if candidate is None:
                continue

            target_gid, score, runner_up = candidate
            self._merge_gid(source_gid, target_gid)
            for update_row in rows:
                if update_row["gid"] == source_gid:
                    update_row["gid"] = target_gid
                    update_row["previous_gid"] = target_gid
                    self._track_to_gid[update_row["track_key"]] = target_gid
                    self._tracklets[update_row["track_key"]]["gid"] = target_gid

            if self._debug_similarity or log:
                print(
                    f"  [Re-ID merge] G{source_gid} -> G{target_gid} "
                    f"score={score:.3f} runner_up={runner_up:.3f} "
                    f"tracklet_len={row['tracklet_len']} "
                    f"Cam{row['src']}#{row['track_id']}"
                )

    def _best_merge_candidate(self, source_gid: int, row: dict,
                              active_by_src: dict[int, set[int]]
                              ) -> tuple[int, float, float] | None:
        candidates = self._candidate_gids(
            exclude=active_by_src.get(row["src"], set()),
            max_count=self._cfg.global_id_merge_max_candidates,
            only_older_than=source_gid,
        )

        scores = []
        for target_gid in candidates:
            reid_score = self._score_gid(target_gid, row["embedding"])
            scores.append((
                target_gid,
                self._blend_geo_score(reid_score, row, target_gid),
            ))

        if not scores:
            return None

        scores.sort(key=lambda item: item[1], reverse=True)
        target_gid, best_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        if best_score < self._cfg.global_id_merge_threshold:
            return None
        if runner_up > 0.0 and best_score < runner_up + self._cfg.global_id_merge_margin:
            return None
        return target_gid, best_score, runner_up

    def _candidate_gids(self, exclude: set[int] | None = None,
                        max_count: int = 80,
                        only_older_than: int | None = None) -> list[int]:
        """Return a bounded list of recent gallery IDs for expensive matching."""
        exclude = exclude or set()
        candidates = []
        for gid, entry in self._gallery.items():
            if gid in exclude:
                continue
            if only_older_than is not None and gid >= only_older_than:
                continue
            candidates.append((entry.get("age", 0), gid))

        candidates.sort(key=lambda item: (item[0], -item[1]))
        return [gid for _, gid in candidates[:max_count]]

    def _merge_gid(self, source_gid: int, target_gid: int) -> None:
        if source_gid == target_gid or source_gid not in self._gallery:
            return
        if target_gid not in self._gallery:
            self._gallery[target_gid] = self._new_gallery_entry()

        self._merge_gallery_entries(source_gid, target_gid)
        for track_key, gid in list(self._track_to_gid.items()):
            if gid == source_gid:
                self._track_to_gid[track_key] = target_gid
        for tracklet in self._tracklets.values():
            if tracklet.get("gid") == source_gid:
                tracklet["gid"] = target_gid
        del self._gallery[source_gid]

    def _merge_gallery_entries(self, source_gid: int, target_gid: int) -> None:
        source = self._gallery.get(source_gid, self._new_gallery_entry())
        target = self._gallery.setdefault(target_gid, self._new_gallery_entry())
        target["age"] = min(target.get("age", 0), source.get("age", 0))

        if not self._use_prototypes():
            source_embedding = source.get("embedding", [])
            # len() not truthiness: source_embedding may be an np array now.
            if len(source_embedding) > 0:
                target["embedding"] = source_embedding
            return

        target_prototypes = target.setdefault("prototypes", [])
        target_prototypes.extend(source.get("prototypes", []))
        target_prototypes.sort(key=lambda p: p.get("last_seen", 0))
        if len(target_prototypes) > self._cfg.gallery_max_prototypes:
            del target_prototypes[:-self._cfg.gallery_max_prototypes]

    def _local_rect(self, rect_params, src: int, frame_meta) -> dict:
        """Return bbox coordinates in source-local/tile-local space."""
        left = float(rect_params.left)
        top = float(rect_params.top)
        width = float(rect_params.width)
        height = float(rect_params.height)

        if self._pretiler:
            frame_w, frame_h = self._frame_size(frame_meta)
        else:
            col = src % max(1, self._cols)
            row = src // max(1, self._cols)
            left -= col * self._tile_w
            top -= row * self._tile_h
            if self._frame_sizes is not None:
                if src in self._frame_sizes:
                    frame_w, frame_h = self._frame_sizes[src]
                elif self._frame_sizes:
                    # Use minimum valid frame width as fallback so all cameras
                    # land in the same coordinate space.  The min corresponds to
                    # the actual source resolution (e.g. 640×360) while the
                    # tile_w default (1280) creates a mixed PRED/GT space that
                    # breaks the single-scale assumption in metrics_mmp.py.
                    frame_w = min(w for w, _ in self._frame_sizes.values())
                    frame_h = min(h for _, h in self._frame_sizes.values())
                else:
                    frame_w, frame_h = float(self._tile_w), float(self._tile_h)
            else:
                frame_w, frame_h = float(self._tile_w), float(self._tile_h)
            sx = frame_w / max(1.0, float(self._tile_w))
            sy = frame_h / max(1.0, float(self._tile_h))
            left *= sx
            top *= sy
            width *= sx
            height *= sy

        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    @staticmethod
    def _frame_size(frame_meta) -> tuple[float, float]:
        """Best-effort frame size for pre-tiler quality checks."""
        width_names = ("source_frame_width", "frame_width", "source_width", "width")
        height_names = ("source_frame_height", "frame_height", "source_height", "height")
        width = next(
            (float(getattr(frame_meta, name)) for name in width_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            1920.0,
        )
        height = next(
            (float(getattr(frame_meta, name)) for name in height_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            1080.0,
        )
        return width, height

    def _annotate_embedding_quality(self, rows: list[dict]) -> None:
        for row in rows:
            ok, reason = self._embedding_quality(row, rows)
            row["embedding_quality_ok"] = ok
            row["embedding_quality_reason"] = reason

    def _embedding_quality(self, row: dict,
                           rows: list[dict]) -> tuple[bool, str]:
        # Pure logic lives in src/reid/quality.py; pass the current tuning values.
        return quality.embedding_quality(
            row, rows,
            enabled=self._cfg.enable_embedding_quality_gate,
            edge_margin_ratio=self._cfg.reid_edge_margin_ratio,
            min_height_ratio=self._cfg.reid_min_bbox_height_ratio,
            min_area_ratio=self._cfg.reid_min_bbox_area_ratio,
            min_aspect=self._cfg.reid_min_bbox_aspect_ratio,
            max_aspect=self._cfg.reid_max_bbox_aspect_ratio,
            max_overlap_iou=self._cfg.reid_max_overlap_iou_for_update,
        )

    @staticmethod
    def _rect_iou(a: dict, b: dict) -> float:
        return quality.rect_iou(a, b)

    def _update_tracklet(self, track_key: tuple, src: int, track_id: int,
                         embedding: list[float],
                         quality_ok: bool = True,
                         foot_world=None) -> dict:
        tracklet = self._tracklets.get(track_key)
        if tracklet is None:
            tracklet = {
                "src": src,
                "track_id": track_id,
                "embeddings": [],
                "age": 0,
                "gid": self._track_to_gid.get(track_key),
                "last_embedding_frame": -self._cfg.tracklet_embedding_interval,
                "foot_world": None,
                # Fields below feed the micro-batch fusion engine. Cheap to keep
                # even when fusion is off.
                "tracklet_id": self._next_tracklet_id,
                "start_frame": self._frame_count,
                "end_frame": self._frame_count,
                "num_detections": 0,
                # Running (uncapped) embedding sum+count for the fusion engine.
                # The rolling "embeddings" list above is capped for matching
                # stability; fusion wants the full-tracklet mean like the
                # offline exporter, so accumulate separately.
                "fusion_emb_sum": None,
                "fusion_emb_count": 0,
            }
            self._tracklets[track_key] = tracklet
            self._next_tracklet_id += 1
        tracklet["age"] = 0
        tracklet["src"] = src
        tracklet["track_id"] = track_id
        tracklet["end_frame"] = self._frame_count
        tracklet["num_detections"] += 1
        if foot_world is not None:
            tracklet["foot_world"] = foot_world
        should_sample = self._should_sample_tracklet_embedding(tracklet)
        tracklet["sampled_this_frame"] = bool(embedding and quality_ok and should_sample)
        if tracklet["sampled_this_frame"]:
            tracklet["embeddings"].append(embedding)
            tracklet["last_embedding_frame"] = self._frame_count
            if len(tracklet["embeddings"]) > self._cfg.tracklet_max_embeddings:
                del tracklet["embeddings"][:-self._cfg.tracklet_max_embeddings]
            if self._use_fusion:
                v = np.asarray(embedding, dtype=np.float32)
                if tracklet["fusion_emb_sum"] is None:
                    tracklet["fusion_emb_sum"] = v.copy()
                else:
                    tracklet["fusion_emb_sum"] += v
                tracklet["fusion_emb_count"] += 1
        return tracklet

    def _should_sample_tracklet_embedding(self, tracklet: dict) -> bool:
        if len(tracklet["embeddings"]) < self._cfg.tracklet_min_embeddings_for_match:
            return True
        interval = max(1, self._cfg.tracklet_embedding_interval)
        return self._frame_count - tracklet.get("last_embedding_frame", -interval) >= interval

    def _tracklet_embedding(self, tracklet: dict,
                            fallback: list[float] | None = None) -> list[float]:
        if not self._cfg.use_tracklet_embedding:
            return fallback or []

        embeddings = tracklet.get("embeddings", [])
        if len(embeddings) < self._cfg.tracklet_min_embeddings_for_match:
            return fallback or []
        return _mean_embedding(embeddings) or (fallback or [])

    def _rank_gallery(self, embedding: list[float]) -> list[tuple[int, float]]:
        """Return global IDs ranked by single embedding or prototype similarity."""
        if not embedding:
            return []

        scores = []
        for gid, entry in self._gallery.items():
            if self._use_prototypes():
                score = self._best_prototype_score(embedding, entry)
            else:
                score = _cosine_similarity(embedding, entry.get("embedding", []))
            scores.append((gid, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _is_gid_match_allowed(self, embedding: list[float],
                              candidate_gid: int | None,
                              previous_gid: int | None,
                              ranked: list[tuple[int, float]]) -> tuple[bool, str]:
        """Apply threshold, stickiness, and ambiguity gates for a candidate ID."""
        if not embedding:
            return False, "no_embedding"
        if candidate_gid is None or candidate_gid == -1:
            return False, "empty_gallery"
        if candidate_gid not in self._gallery:
            return False, "stale_candidate"

        candidate_score = self._score_gid(candidate_gid, embedding)
        if candidate_score < self._cfg.similarity_threshold:
            return False, "below_threshold"

        if (
            self._cfg.enable_id_stickiness
            and previous_gid is not None
            and previous_gid in self._gallery
            and candidate_gid != previous_gid
        ):
            previous_score = self._score_gid(previous_gid, embedding)
            if candidate_score < previous_score + self._cfg.id_switch_margin:
                return (
                    False,
                    f"switch_margin(prev=G{previous_gid},"
                    f"prev_score={previous_score:.3f})",
                )

        if (
            self._cfg.enable_ambiguous_match_rejection
            and candidate_gid != previous_gid
        ):
            runner_up = max(
                (score for gid, score in ranked if gid != candidate_gid),
                default=0.0,
            )
            if runner_up > 0.0 and candidate_score < runner_up + self._cfg.match_ambiguity_margin:
                return False, f"ambiguous(runner_up={runner_up:.3f})"

        return True, "ok"

    def _allocate_new_gid(self) -> int:
        while self._next_gid in self._gallery:
            self._next_gid += 1
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _score_gid(self, gid: int, embedding: list[float]) -> float:
        if not embedding or gid not in self._gallery:
            return 0.0
        entry = self._gallery[gid]
        if self._use_prototypes():
            return self._best_prototype_score(embedding, entry)
        return _cosine_similarity(embedding, entry.get("embedding", []))

    def _blend_geo_score(self, reid_score: float, row: dict,
                         candidate_gid: int) -> float:
        """
        Blend geometry score into the ReID score for cross-camera pairs.

        Same-camera assignments are left unchanged (geometry doesn't help
        when the local tracker already handles intra-camera identity).
        Returns reid_score unchanged when geometry is disabled or unavailable.
        """
        if self._geometry is None or self._cfg.geo_weight <= 0.0:
            return reid_score

        foot_q = row.get("foot_world")
        if foot_q is None:
            return reid_score

        # Look for the best geo score among all tracklets mapped to this gid
        # that come from a *different* source.
        best_geo = 0.0
        for (t_src, _t_id), t_gid in self._track_to_gid.items():
            if t_gid != candidate_gid:
                continue
            if t_src == row["src"]:
                continue   # same camera — skip
            foot_t = self._tracklets.get((t_src, _t_id), {}).get("foot_world")
            g = GroundPlaneGeometry.geo_score(foot_q, foot_t)
            if g > best_geo:
                best_geo = g

        if best_geo == 0.0:
            return reid_score

        return (1.0 - self._cfg.geo_weight) * reid_score + self._cfg.geo_weight * best_geo

    def _assignment_score(self, reid_score: float, best_reid_score: float,
                          row: dict, candidate_gid: int) -> float:
        if self._cfg.geo_assignment_mode == "close_reid_only":
            if best_reid_score - reid_score > self._cfg.geo_reid_margin:
                return reid_score
        return self._blend_geo_score(reid_score, row, candidate_gid)

    def _use_prototypes(self) -> bool:
        return self._cfg.gallery_max_prototypes > 0

    def _new_gallery_entry(self) -> dict:
        if self._use_prototypes():
            return {"prototypes": [], "age": 0}
        return {"embedding": [], "age": 0}

    @staticmethod
    def _best_prototype_score(embedding, entry: dict,
                              src: int | None = None) -> float:
        prototypes = entry.get("prototypes", [])
        if src is not None:
            prototypes = [p for p in prototypes if p.get("src") == src]
        if not prototypes or embedding is None or len(embedding) == 0:
            return 0.0

        # Vectorized cosine against all prototypes at once: one (k, d) @ (d,)
        # matmul instead of k separate Python-level cosine calls. Prototype
        # embeddings are stored as np.float32 by _update_gallery, so stacking
        # is cheap and matching cost scales with a single BLAS call.
        q = np.asarray(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return 0.0
        mat = np.asarray([p["embedding"] for p in prototypes], dtype=np.float32)
        denom = np.linalg.norm(mat, axis=1) * q_norm
        sims = np.where(denom > 0.0, (mat @ q) / np.where(denom > 0.0, denom, 1.0), 0.0)
        best = float(sims.max())
        return best if math.isfinite(best) else 0.0

    def _update_gallery(self, gid: int, embedding: list[float], src: int) -> None:
        """Refresh a global identity using single-vector or prototype mode."""
        entry = self._gallery.setdefault(gid, self._new_gallery_entry())
        entry["age"] = 0
        if not embedding:
            return

        if not self._use_prototypes():
            # Store as np.float32 so repeated comparisons skip list→array
            # reconversion. Query embeddings stay lists (sentinel-safe), and
            # _cosine_similarity accepts either type.
            entry["embedding"] = np.asarray(embedding, dtype=np.float32)
            return

        prototypes = entry["prototypes"]
        same_src_score = self._best_prototype_score(embedding, entry, src=src)
        all_score = self._best_prototype_score(embedding, entry)
        has_src = any(p.get("src") == src for p in prototypes)
        should_add = (
            not prototypes
            or not has_src
            or same_src_score < self._cfg.prototype_add_threshold
            or all_score < self._cfg.prototype_add_threshold
        )
        if not should_add:
            return

        prototypes.append({
            "embedding": np.asarray(embedding, dtype=np.float32),
            "src": src,
            "last_seen": self._frame_count,
        })
        if len(prototypes) > self._cfg.gallery_max_prototypes:
            # Keep the most recent prototypes so the gallery can adapt without
            # collapsing to a single latest embedding.
            del prototypes[:-self._cfg.gallery_max_prototypes]



# =============================================================================
# CLI integration
# =============================================================================
def configure_from_args(args) -> ReIDConfig:
    """Build a ReIDConfig from parsed CLI args."""
    return ReIDConfig.from_args(args)


def config_summary(config: ReIDConfig) -> str:
    """Startup-log summary of the active ReID/Global-ID tuning."""
    return config.summary()
