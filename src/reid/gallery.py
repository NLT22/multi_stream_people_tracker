"""
Cross-camera ReID gallery: tracker-embedding extraction + Global-ID assignment.

Contains the metadata probes used by src/main.py:
  - SourceIdCollectorProbe (pre-tiler): reads source_id + ReID embeddings
  - CrossCameraGalleryProbe (post-tiler): matches embeddings -> stable Global ID

Tuning lives in a typed ReIDConfig (src/reid/config.py), built from CLI args
via configure_from_args(args) and passed into the probe as self._cfg.
"""

import os
import traceback

import pyservicemaker as psm


from src.pipeline.model_utils import (
    infer_source_id_from_tiled_box,
    set_object_label,
)
from src.reid.visualization import style_object_by_id
from src.reid.geometry import GroundPlaneGeometry
from src.reid import fusion_bridge
from src.reid.tracklet_store import TrackletStore
from src.reid.gallery_store import GalleryStore
from src.reid.config import ReIDConfig
from src.reid.detection_row import DetectionRow

# SourceIdCollectorProbe is re-exported here: runner.py builds it via
# gallery.SourceIdCollectorProbe, and gallery uses it internally (pretiler path).
from src.reid.metadata import SourceIdCollectorProbe  # noqa: F401

# Probe logic split by concern into mixins (see each module). They operate on the
# probe's shared state via self; behavior is identical to the single-class form.
from src.reid.gallery_rows import GalleryRowsMixin
from src.reid.gallery_conflict import GalleryConflictMixin
from src.reid.gallery_assignment import GalleryAssignmentMixin
from src.reid.gallery_merge import GalleryMergeMixin


class CrossCameraGalleryProbe(
    psm.BatchMetadataOperator,
    GalleryRowsMixin,
    GalleryConflictMixin,
    GalleryAssignmentMixin,
    GalleryMergeMixin,
):
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
                 config: "ReIDConfig | None" = None,
                 passthrough_export: bool = False,
                 buffered_remap_path: str | None = None):
        super().__init__()
        self._cfg = config if config is not None else ReIDConfig()
        # passthrough_export: skip the expensive online cross-camera matching +
        # OSD label drawing, but still export per-detection rows + embeddings so
        # the authoritative Global IDs are computed offline by src.mtmc.live_buffered.
        # This is the lean production-ingest path (much higher throughput).
        self._passthrough_export = passthrough_export
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
        self._gs = GalleryStore(self._cfg)        # owns the Global-ID memory
        self._gallery = self._gs.gallery          # alias (only mutated, never rebound)
        self._track_to_gid: dict[tuple, int] = {}  # (src, track_id) → global_id
        self._ts = TrackletStore(self._cfg)       # owns the tracklet memory
        self._tracklets = self._ts.tracklets      # alias (only mutated, never rebound)
        self._frame_count = 0

        # Route (a): consume live_buffered's gids-csv ((cam,ltid)->gid) as the OSD
        # display remap, so labels show the authoritative buffered/anchor-guided
        # Global IDs (with ~window latency) instead of the volatile online greedy
        # IDs. Falls back to the online/fusion gid for tracks not yet clustered.
        self._buffered_remap_path = buffered_remap_path
        self._buffered_track_gid: dict[tuple[int, int], int] = {}
        self._buffered_remap_mtime = 0.0

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
        self._gs.frame_count = self._frame_count
        log = self._frame_count % 60 == 0

        # Expire stale gallery entries + tracklets (store-owned mutation). The
        # probe owns track_to_gid, so it clears references to the expired gids
        # and deleted tracklet keys the stores report back.
        expired_gids = self._gs.expire(self._cfg.gallery_max_age)
        for gid in expired_gids:
            self._track_to_gid = {k: v for k, v in self._track_to_gid.items()
                                  if v != gid}
        for key in self._ts.expire(self._cfg.tracklet_max_age, expired_gids):
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
                rows.append(DetectionRow(
                    src=src,
                    track_id=oid,
                    track_key=track_key,
                    rect=rect,
                    raw_embedding=embedding,
                    foot_world=foot_world,   # (X_mm, Y_mm) or None
                ))

            # Lean ingest: export raw rows + embeddings, skip online matching/draw.
            # Cross-camera Global IDs are recomputed offline by live_buffered.
            if self._passthrough_export:
                for row in rows:
                    row.gid = -1
                if self._exporter is not None:
                    self._export_rows(frame_meta, rows)
                continue

            self._annotate_embedding_quality(rows)
            for row in rows:
                tracklet = self._ts.update(
                    row.track_key, row.src, row.track_id,
                    row.raw_embedding, self._frame_count,
                    initial_gid=self._track_to_gid.get(row.track_key),
                    use_fusion=self._use_fusion,
                    quality_ok=row.embedding_quality_ok,
                    foot_world=row.get("foot_world"))
                row.tracklet_len = len(tracklet["embeddings"])
                row.update_gallery = tracklet.get("sampled_this_frame", False)
                match_embedding = self._ts.tracklet_embedding(
                    tracklet,
                    # Matching is allowed to be softer than memory update:
                    # a border/back-view/overlap crop may still be enough to
                    # match an existing cross-camera GID, but it must not be
                    # stored into the long-term gallery unless it passes the
                    # stricter quality gate.
                    fallback=row.raw_embedding,
                )
                previous_gid = self._track_to_gid.get(row.track_key)
                if previous_gid is None:
                    previous_gid = tracklet.get("gid")
                if previous_gid not in self._gallery:
                    previous_gid = None
                row.previous_gid = previous_gid
                row.gid = previous_gid
                row.had_previous_gid = previous_gid is not None
                row.embedding = match_embedding
                row.allow_new_gid = row.embedding_quality_ok
                row.identity_conflict = False
                row.suppress_gallery_update = False
                row.release_previous_gid = False
                # Known tracks keep their GID through low-quality frames, but
                # low-quality new tracks may only match existing IDs. They wait
                # for a cleaner crop before creating a brand-new Global ID.
                row.defer_assignment = (
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
                gid = row.gid
                if gid is not None:
                    self._track_to_gid[row.track_key] = gid
                    self._tracklets[row.track_key]["gid"] = gid
                    if (
                        row.get("gallery_updated") is not True
                        and not row.get("suppress_gallery_update")
                    ):
                        self._gs.update(
                            gid,
                            row.raw_embedding
                            if row.get("update_gallery") else [],
                            row.src,
                        )
                elif row.get("release_previous_gid"):
                    self._track_to_gid.pop(row.track_key, None)
                    self._tracklets[row.track_key]["gid"] = None

            if (
                self._cfg.enable_global_id_merge
                and self._frame_count % self._cfg.global_id_merge_interval == 0
            ):
                self._merge_duplicate_global_ids(rows, log)

            if self._use_fusion and self._geometry is not None:
                self._accumulate_fusion_geo(rows, frame_meta)

            self._draw_labels(frame_meta, rows)
            if self._exporter is not None:
                self._export_rows(frame_meta, rows)

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
                  f"total_gids_ever_assigned={self._gs._next_gid - 1}")

    def _reload_buffered_remap(self) -> None:
        """Reload live_buffered's (cam,ltid)->gid map if the file changed (cheap stat)."""
        path = self._buffered_remap_path
        if not path:
            return
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return
        if mt == self._buffered_remap_mtime:
            return
        self._buffered_remap_mtime = mt
        m: dict[tuple[int, int], int] = {}
        try:
            with open(path) as f:
                next(f, None)  # header: group,cam_id,local_track_id,global_id
                for line in f:
                    p = line.strip().split(",")
                    if len(p) >= 4:
                        m[(int(p[1]), int(p[2]))] = int(p[3])
            self._buffered_track_gid = m
        except (OSError, ValueError):
            pass

    def _draw_labels(self, frame_meta, rows: list[DetectionRow]) -> None:
        """Write the Global-ID label + style onto each person's OSD box."""
        if self._buffered_remap_path:
            self._reload_buffered_remap()
        row_by_key = {(row.src, row.track_id): row for row in rows}
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
            if self._buffered_remap_path:
                bgid = self._buffered_track_gid.get((src, obj_meta.object_id))
                draw_gid = bgid if bgid is not None else self._display_gid(row.gid)
            else:
                draw_gid = self._display_gid(row.gid)
            label = f"GID:{draw_gid if draw_gid is not None else '?'} "
            set_object_label(obj_meta, label)
            style_object_by_id(obj_meta, draw_gid)

    def _export_rows(self, frame_meta, rows: list[DetectionRow]) -> None:
        """Record each row to the prediction exporter (RAW gid; fusion remap is
        applied at flush time so late merges still correct earlier frames)."""
        for row in rows:
            rect = row.rect
            # Post-tiler frame_meta.frame_number is the batch counter (shared by
            # all sources); use the per-source frame number from the pre-tiler
            # SourceIdCollectorProbe when available.
            fn = (self._frame_numbers.get(row.src, 0)
                  if self._frame_numbers is not None
                  else frame_meta.frame_number)
            self._exporter.record(
                frame_no=fn,
                cam_id=row.src,
                local_track_id=row.track_id,
                global_id=row.gid,
                left=rect["left"],
                top=rect["top"],
                width=rect["width"],
                height=rect["height"],
                embedding=(
                    row.raw_embedding
                    if (row.get("update_gallery")
                        and not row.get("suppress_gallery_update"))
                    else None
                ),
                foot_world=row.foot_world,
                # Per-detection embedding for every detection (faithful
                # reference reproduction needs per-frame, not per-tracklet).
                det_embedding=row.raw_embedding or None,
            )

    def _display_gid(self, gid: int | None) -> int | None:
        """Apply the micro-batch fusion remap to a raw gid (no-op when off)."""
        if gid is None or not self._use_fusion:
            return gid
        return self._fusion_remap.get(gid, gid)

    def _accumulate_fusion_geo(self, rows: list[dict], frame_meta) -> None:
        """Accumulate per-(gid, frame) foot positions for geometry-assisted fusion."""
        fusion_bridge.accumulate_geo(
            rows, self._frame_numbers, frame_meta.frame_number,
            self._frame_count, self._fusion_geo_points,
            self._cfg.micro_batch_fusion_geo_sample_step)

    def _run_micro_batch_fusion(self) -> None:
        """One micro-batch pass: cluster tracklet embeddings across cameras.

        In-pipeline equivalent of `src.eval.online_fusion`. Record-building and
        the fresh-engine replay live in src/reid/fusion_bridge.py (testable).
        """
        exporter_tracklets = (
            getattr(self._exporter, "_tracklets", None)
            if self._exporter is not None else None
        )
        records = fusion_bridge.build_records(
            exporter_tracklets, self._tracklets, self._track_to_gid,
            self._fusion_tid_by_key, self._frame_count)
        self._fusion_remap = fusion_bridge.run_fusion_pass(
            records, self._cfg, self._fusion_geo_points,
            self._fusion.geo_weight, self._fusion.geo_min_overlaps)
        if self._debug_similarity:
            print(f"  [fusion] frame={self._frame_count} "
                  f"tracklets={len(records)} "
                  f"merges={len(self._fusion_remap)} remap={self._fusion_remap}")

# =============================================================================
# CLI integration
# =============================================================================
def configure_from_args(args) -> ReIDConfig:
    """Build a ReIDConfig from parsed CLI args."""
    return ReIDConfig.from_args(args)


def config_summary(config: ReIDConfig) -> str:
    """Startup-log summary of the active ReID/Global-ID tuning."""
    return config.summary()
