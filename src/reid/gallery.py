"""
Cross-camera ReID gallery: tracker-embedding extraction + Global-ID assignment.

Contains the metadata probes used by src/main.py:
  - SourceIdCollectorProbe (pre-tiler): reads source_id + ReID embeddings
  - CrossCameraGalleryProbe (post-tiler): matches embeddings -> stable Global ID

Tuning lives in the module-level constants below. The pipeline CLI overrides
them via configure_from_args(args) and logs them via config_summary().
"""

import math
import traceback

import numpy as np

import pyservicemaker as psm

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.pipeline.model_utils import (
    infer_source_id_from_tiled_box,
    set_object_label,
)
from src.reid.visualization import style_object_by_id
from src.reid.geometry import GroundPlaneGeometry


# =============================================================================
# ReID / Global-ID Tuning
# (ordered: local stream → global)
# =============================================================================

# --- 1. Embedding quality gate (per-frame crop filter) -----------------------
#   A bad crop can be useful for drawing the current bbox, but it should not
#   create/update long-term identity memory. Keep the current GID if known, and
#   wait for a cleaner crop before matching a brand-new local track.
ENABLE_EMBEDDING_QUALITY_GATE = True
REID_EDGE_MARGIN_RATIO = 0.005  # MMPTracking: indoor wide-angle, persons often near edge
REID_MIN_BBOX_HEIGHT_RATIO = 0.04
REID_MIN_BBOX_AREA_RATIO = 0.0008
REID_MIN_BBOX_ASPECT_RATIO = 0.12
REID_MAX_BBOX_ASPECT_RATIO = 0.95
REID_MAX_OVERLAP_IOU_FOR_UPDATE = 0.25

# --- 2. Tracklet (local per-camera track memory) -----------------------------
#   Tracklet mode averages recent embeddings for each (camera, local_track_id).
#   This is more stable than matching on a single noisy frame crop.
USE_TRACKLET_EMBEDDING = True

#   Do not store every frame embedding. DeepStream's native ReID tracker has a
#   similar reidExtractionInterval knob; sampling lowers cost and avoids letting
#   long runs of back-view / partial crops overwrite a clean appearance.
TRACKLET_EMBEDDING_INTERVAL = 5

#   Number of recent embeddings kept per local track.
#   Larger -> smoother but slower to adapt if tracker switches identity.
TRACKLET_MAX_EMBEDDINGS = 8

#   Use raw frame embedding until the local tracklet has this many embeddings.
#   Larger -> more stable first match, but slower cross-camera linking.
TRACKLET_MIN_EMBEDDINGS_FOR_MATCH = 3

#   Drop inactive local tracklets after this many batches.
TRACKLET_MAX_AGE = 1800

# --- 3. Gallery (global identity store) --------------------------------------
#   Gallery stores known Global IDs after a local track disappears.
#   Increase age if people leave/re-enter after a long gap.
#   Decrease age if old identities are reused incorrectly.
GALLERY_MAX_AGE = 1800

#   Each Global ID can keep multiple appearance vectors for different views.
#   More prototypes improve view coverage but increase matching cost.
#   Set to 0 to disable multi-prototype mode and keep one vector per Global ID.
GALLERY_MAX_PROTOTYPES = 24

#   Add a new prototype when the current embedding is visually different enough.
#   Higher -> compact gallery, less noise.
#   Lower  -> more view coverage, more chance of storing bad crops.
PROTOTYPE_ADD_THRESHOLD = 0.72

# --- 4. Match threshold (local track → global ID similarity) -----------------
#   Higher -> fewer false matches, but more ID splits.
#   Lower  -> easier to reconnect the same person, but more merge risk.
SIMILARITY_THRESHOLD = 0.68

# --- 5. Assignment (local tracks competing for global IDs) -------------------
#   Hungarian solves one-to-one assignment for new local tracks within a stream.
#   This prevents multiple people in the same camera from selecting one Global ID.
USE_HUNGARIAN_ASSIGNMENT = True
GLOBAL_ASSIGNMENT_MAX_CANDIDATES = 80

#   Keeps already-known local tracks from displaying the same Global ID twice in
#   one stream. Keep this on with Hungarian; disable only for A/B debugging.
ENFORCE_UNIQUE_GLOBAL_PER_STREAM = True

# --- 6. ID stickiness & ambiguity guard (stabilize assignments) --------------
#   A local track that already has a Global ID should not switch to another ID
#   unless the new candidate is clearly better than the current one.
#   This prevents labels from bouncing between two visually similar IDs.
ENABLE_ID_STICKINESS = True
ID_SWITCH_MARGIN = 0.12

#   For a new/released local track, accept an existing Global ID only when the
#   best match beats the runner-up by this margin. If G14=0.64 and G8=0.62,
#   create/keep a separate ID instead of randomly bouncing between them.
ENABLE_AMBIGUOUS_MATCH_REJECTION = True
MATCH_AMBIGUITY_MARGIN = 0.06

# --- 7. Global-ID merge (cross-camera duplicate resolution) ------------------
#   A cross-view track may first become a new Global ID because the opposite
#   camera crop looks very different. After enough tracklet evidence, merge the
#   duplicate ID into the best older Global ID if the match is strong and does
#   not create two copies of one Global ID in the same stream frame.
ENABLE_GLOBAL_ID_MERGE = True
GLOBAL_ID_MERGE_THRESHOLD = 0.76
GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS = 6
GLOBAL_ID_MERGE_MARGIN = 0.04
GLOBAL_ID_MERGE_INTERVAL = 5
GLOBAL_ID_MERGE_MAX_CANDIDATES = 80

# --- 8. Debug ----------------------------------------------------------------
#   Number of nearest Global IDs printed when --debug-similarity is enabled.
DEBUG_TOP_K = 3

# --- 9. Ground-plane geometry (calibration-assisted matching) ----------------
#   When GroundPlaneGeometry is provided, the combined score is:
#     score = (1 - GEO_WEIGHT) * reid_score + GEO_WEIGHT * geo_score
#   GEO_WEIGHT=0 disables geometry blending entirely (pure ReID).
#   GEO_WEIGHT=0.35 gives meaningful spatial cues without overriding appearance.
#   Only cross-camera pairs (src_a != src_b) use geometry; same-camera pairs
#   skip it because the DeepSORT local tracker already handles those.
GEO_WEIGHT = 0.35
GEO_ASSIGNMENT_MODE = "weight_only"  # weight_only | close_reid_only
GEO_REID_MARGIN = 1.0                # only used by close_reid_only


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0–1.0.

    Accepts list or np.ndarray for either argument. Use len() rather than
    truthiness so multi-element np arrays (gallery storage) don't raise
    "truth value of an array is ambiguous".
    """
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(va, vb) / (norm_a * norm_b))
    if not math.isfinite(score):
        return 0.0
    return score


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    """Average same-sized embeddings and L2-normalize the result."""
    valid = [e for e in embeddings if e]
    if not valid:
        return []

    dim = len(valid[0])
    same_dim = [e for e in valid if len(e) == dim]
    if not same_dim:
        return []

    arr = np.array(same_dim, dtype=np.float32)   # shape (n, dim)
    mean = arr.mean(axis=0)                        # shape (dim,)
    norm = np.linalg.norm(mean)
    if norm == 0.0:
        return mean.tolist()
    return (mean / norm).tolist()


def max_weight_assignment(weights: list[list[float]]) -> list[int]:
    """
    Hungarian assignment for max-weight rectangular matrices.

    Returns a list where result[row] = assigned column. The implementation uses
    the classic O(n^2*m) shortest augmenting path form for min-cost assignment,
    converting max weights to costs internally. It assumes columns >= rows; the
    caller always provides enough "new identity" dummy columns.
    """
    if not weights:
        return []

    n = len(weights)
    m = len(weights[0])
    if m < n:
        raise ValueError("Hungarian assignment requires columns >= rows")

    max_w = max(max(row) for row in weights)
    cost = [[max_w - w for w in row] for row in weights]

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Pre-tiler: source_id is valid here — collect it for each tracked person.
    Also extracts Re-ID embeddings from tracker ReID metadata.

    Fills shared dicts:
      embeddings: (source_id, object_id) → embedding vector (list[float]) or []
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int,
                 debug: bool = False, frame_numbers: dict | None = None,
                 frame_sizes: dict | None = None):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
        self._frame_numbers = frame_numbers  # source_id → frame_number (for exporter)
        self._frame_sizes = frame_sizes      # source_id → (width, height)
        self._person_class_id = person_class_id
        self._debug = debug
        self._frame_count = 0
        self._persons_seen = 0
        self._embeddings_seen = 0
        self._object_reid_metas = 0
        self._object_tensor_metas = 0
        self._frame_tensor_metas = 0
        self._debug_failures_printed = 0

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            print("[reid ERROR] SourceIdCollectorProbe failed:")
            traceback.print_exc()

    def _handle_metadata(self, batch_meta):
        self._frame_count += 1
        batch_persons = 0
        batch_embeddings = 0
        batch_obj_reids = 0
        batch_obj_tensors = 0
        batch_frame_tensors = 0

        # The shared dicts are a per-batch handoff to the post-tiler gallery
        # probe, which runs synchronously on the same buffer right after this
        # one. Clearing here bounds memory: without it, every (src, object_id)
        # ever seen would accumulate forever on long/multi-camera videos.
        self._embeddings.clear()
        self._id_map.clear()
        if self._frame_numbers is not None:
            self._frame_numbers.clear()
        if self._frame_sizes is not None:
            self._frame_sizes.clear()

        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            if self._frame_numbers is not None:
                self._frame_numbers[src] = frame_meta.frame_number
            if self._frame_sizes is not None:
                size = self._source_frame_size(frame_meta)
                if size is not None:
                    self._frame_sizes[src] = size
            frame_tensor_count = self._count_iter(frame_meta.tensor_items)
            batch_frame_tensors += frame_tensor_count
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                oid = obj_meta.object_id
                self._id_map[oid] = src
                batch_persons += 1

                # Try to extract Re-ID embedding from tracker metadata.
                # NvDeepSORT and NvDCF+ReAssoc can expose this when their ReID
                # config has outputReidTensor: 1. If the selected tracker does
                # not export embeddings, the Python global gallery falls back
                # to GID:?/new IDs because it has no appearance evidence.
                embedding, obj_reid_count, obj_tensor_count, reason = (
                    self._extract_embedding(obj_meta))
                batch_obj_reids += obj_reid_count
                batch_obj_tensors += obj_tensor_count
                if embedding:
                    batch_embeddings += 1
                elif self._debug and self._debug_failures_printed < 12:
                    print(
                        f"  [Re-ID tensor debug] Cam{src}#{oid} "
                        f"embedding=empty reason={reason} "
                        f"obj_reid_items={obj_reid_count} "
                        f"obj_tensor_items={obj_tensor_count} "
                        f"frame_tensor_items={frame_tensor_count} "
                        f"torch_available={_TORCH_AVAILABLE}"
                    )
                    self._debug_failures_printed += 1
                self._embeddings[(src, oid)] = embedding

        self._persons_seen += batch_persons
        self._embeddings_seen += batch_embeddings
        self._object_reid_metas += batch_obj_reids
        self._object_tensor_metas += batch_obj_tensors
        self._frame_tensor_metas += batch_frame_tensors

        if self._debug and self._frame_count % 60 == 0:
            print(
                f"[reid tensor debug] frame={self._frame_count:06d} "
                f"batch_persons={batch_persons} "
                f"batch_embeddings={batch_embeddings} "
                f"batch_obj_reid_items={batch_obj_reids} "
                f"batch_obj_tensor_items={batch_obj_tensors} "
                f"batch_frame_tensor_items={batch_frame_tensors} "
                f"total_embeddings={self._embeddings_seen}/{self._persons_seen} "
                f"torch_available={_TORCH_AVAILABLE}"
            )

    @staticmethod
    def _count_iter(items) -> int:
        return sum(1 for _ in items)

    @staticmethod
    def _source_frame_size(frame_meta) -> tuple[float, float] | None:
        width_names = ("source_frame_width", "frame_width", "source_width", "width")
        height_names = ("source_frame_height", "frame_height", "source_height", "height")
        width = next(
            (float(getattr(frame_meta, name)) for name in width_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            None,
        )
        height = next(
            (float(getattr(frame_meta, name)) for name in height_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            None,
        )
        if width and height:
            return width, height
        return None

    @staticmethod
    def _extract_embedding(obj_meta) -> tuple[list[float], int, int, str]:
        reid_count = 0
        try:
            for reid_meta in obj_meta.obj_reid_items:
                reid_count += 1
                reid = reid_meta.as_obj_reid()
                feature = reid.feature_vector
                if callable(feature):
                    feature = feature()
                embedding = list(feature) if feature is not None else []
                if embedding:
                    return embedding, reid_count, 0, f"ok_obj_reid_dim_{len(embedding)}"
            if reid_count > 0:
                return [], reid_count, 0, "empty_obj_reid_feature_vector"
        except Exception as e:
            return [], reid_count, 0, f"obj_reid_{type(e).__name__}: {e}"

        tensor_count = 0
        if not _TORCH_AVAILABLE:
            return [], reid_count, tensor_count, "torch_unavailable"

        try:
            for tensor_meta in obj_meta.tensor_items:
                tensor_count += 1
                layers = tensor_meta.as_tensor_output().get_layers()
                if not layers:
                    continue
                raw = next(iter(layers.values()))
                feat = torch.utils.dlpack.from_dlpack(raw)
                embedding = feat.cpu().numpy().flatten().tolist()
                if embedding:
                    return embedding, reid_count, tensor_count, f"ok_tensor_dim_{len(embedding)}"
            return [], reid_count, tensor_count, "no_reid_or_tensor_layers"
        except Exception as e:
            return [], reid_count, tensor_count, f"tensor_{type(e).__name__}: {e}"


class CrossCameraGalleryProbe(psm.BatchMetadataOperator):
    """
    Post-tiler: tiled canvas coordinates — draw labels here.

    Maintains a cross-camera gallery:
      gallery:        global_id → {"prototypes": [...], "age": int}
                      if GALLERY_MAX_PROTOTYPES == 0:
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
                 geometry: "GroundPlaneGeometry | None" = None):
        super().__init__()
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
        self._frame_count = 0
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
                 if v["age"] > GALLERY_MAX_AGE]
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
            if tracklet["age"] > TRACKLET_MAX_AGE
        ]
        for key in stale_tracks:
            del self._tracklets[key]
            self._track_to_gid.pop(key, None)

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
                ENABLE_GLOBAL_ID_MERGE
                and self._frame_count % GLOBAL_ID_MERGE_INTERVAL == 0
            ):
                self._merge_duplicate_global_ids(rows, log)

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
                label = (
                    f"GID:{row['gid'] if row['gid'] is not None else '?'} "
                    # f"LID:{row['track_id']}"
                )
                set_object_label(obj_meta, label)
                style_object_by_id(obj_meta, row["gid"])

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
                f"G{gid}={score:.3f}" for gid, score in ranked[:DEBUG_TOP_K]
            ) or "none"
            print(
                f"  [Re-ID similarity] Cam{src}#{track_id} "
                f"best_gid={best_gid if best_gid != -1 else 'None'} "
                f"max_similarity={best_score:.3f} "
                f"threshold={SIMILARITY_THRESHOLD:.3f} "
                f"tracklet_len={tracklet_len} "
                f"previous_gid={previous_gid if previous_gid is not None else 'None'} "
                f"status={status} reason={display_reason} top{DEBUG_TOP_K}=[{top}]"
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
                max_count=GLOBAL_ASSIGNMENT_MAX_CANDIDATES,
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
                                f"G{gid}={s:.3f}" for gid, s in ranked[:DEBUG_TOP_K]
                            ) or "none"
                            print(
                                f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                                f"assigned=None score={score:.3f} "
                                f"threshold={SIMILARITY_THRESHOLD:.3f} "
                                f"tracklet_len={row['tracklet_len']} "
                                f"quality={row.get('embedding_quality_reason')} "
                                f"status={status} reason={reason} "
                                f"top{DEBUG_TOP_K}=[{top}]"
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
                        f"G{gid}={s:.3f}" for gid, s in ranked[:DEBUG_TOP_K]
                    ) or "none"
                    print(
                        f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                        f"assigned=G{row['gid']} score={score:.3f} "
                        f"threshold={SIMILARITY_THRESHOLD:.3f} "
                        f"tracklet_len={row['tracklet_len']} "
                        f"previous_gid={row.get('previous_gid') if row.get('previous_gid') is not None else 'None'} "
                        f"status={status} reason={reason} top{DEBUG_TOP_K}=[{top}]"
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
            if row["tracklet_len"] < GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS:
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
            max_count=GLOBAL_ID_MERGE_MAX_CANDIDATES,
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
        if best_score < GLOBAL_ID_MERGE_THRESHOLD:
            return None
        if runner_up > 0.0 and best_score < runner_up + GLOBAL_ID_MERGE_MARGIN:
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
        if len(target_prototypes) > GALLERY_MAX_PROTOTYPES:
            del target_prototypes[:-GALLERY_MAX_PROTOTYPES]

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
            frame_w, frame_h = self._frame_sizes.get(
                src, (float(self._tile_w), float(self._tile_h))
            ) if self._frame_sizes is not None else (
                float(self._tile_w), float(self._tile_h)
            )
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
        if not row["raw_embedding"]:
            return False, "no_embedding"
        if not ENABLE_EMBEDDING_QUALITY_GATE:
            return True, "disabled"

        rect = row["rect"]
        frame_w = max(1.0, rect["frame_w"])
        frame_h = max(1.0, rect["frame_h"])
        left = rect["left"]
        top = rect["top"]
        width = max(0.0, rect["width"])
        height = max(0.0, rect["height"])
        right = left + width
        bottom = top + height

        margin_x = frame_w * REID_EDGE_MARGIN_RATIO
        margin_y = frame_h * REID_EDGE_MARGIN_RATIO
        if (
            left <= margin_x
            or top <= margin_y
            or right >= frame_w - margin_x
            or bottom >= frame_h - margin_y
        ):
            return False, "edge_crop"

        if height / frame_h < REID_MIN_BBOX_HEIGHT_RATIO:
            return False, "small_height"
        if (width * height) / (frame_w * frame_h) < REID_MIN_BBOX_AREA_RATIO:
            return False, "small_area"

        aspect = width / height if height > 0.0 else 999.0
        if aspect < REID_MIN_BBOX_ASPECT_RATIO:
            return False, "thin_crop"
        if aspect > REID_MAX_BBOX_ASPECT_RATIO:
            return False, "wide_or_merged_crop"

        max_iou = 0.0
        for other in rows:
            if other is row or other["src"] != row["src"]:
                continue
            max_iou = max(max_iou, self._rect_iou(rect, other["rect"]))
        if max_iou > REID_MAX_OVERLAP_IOU_FOR_UPDATE:
            return False, f"overlap_iou={max_iou:.2f}"

        return True, "ok"

    @staticmethod
    def _rect_iou(a: dict, b: dict) -> float:
        ax1, ay1 = a["left"], a["top"]
        ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
        bx1, by1 = b["left"], b["top"]
        bx2, by2 = bx1 + b["width"], by1 + b["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a["width"]) * max(0.0, a["height"])
        area_b = max(0.0, b["width"]) * max(0.0, b["height"])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0.0 else 0.0

    def _update_tracklet(self, track_key: tuple, src: int, track_id: int,
                         embedding: list[float],
                         quality_ok: bool = True,
                         foot_world=None) -> dict:
        tracklet = self._tracklets.setdefault(track_key, {
            "src": src,
            "track_id": track_id,
            "embeddings": [],
            "age": 0,
            "gid": self._track_to_gid.get(track_key),
            "last_embedding_frame": -TRACKLET_EMBEDDING_INTERVAL,
            "foot_world": None,
        })
        tracklet["age"] = 0
        tracklet["src"] = src
        tracklet["track_id"] = track_id
        if foot_world is not None:
            tracklet["foot_world"] = foot_world
        should_sample = self._should_sample_tracklet_embedding(tracklet)
        tracklet["sampled_this_frame"] = bool(embedding and quality_ok and should_sample)
        if tracklet["sampled_this_frame"]:
            tracklet["embeddings"].append(embedding)
            tracklet["last_embedding_frame"] = self._frame_count
            if len(tracklet["embeddings"]) > TRACKLET_MAX_EMBEDDINGS:
                del tracklet["embeddings"][:-TRACKLET_MAX_EMBEDDINGS]
        return tracklet

    def _should_sample_tracklet_embedding(self, tracklet: dict) -> bool:
        if len(tracklet["embeddings"]) < TRACKLET_MIN_EMBEDDINGS_FOR_MATCH:
            return True
        interval = max(1, TRACKLET_EMBEDDING_INTERVAL)
        return self._frame_count - tracklet.get("last_embedding_frame", -interval) >= interval

    @staticmethod
    def _tracklet_embedding(tracklet: dict,
                            fallback: list[float] | None = None) -> list[float]:
        if not USE_TRACKLET_EMBEDDING:
            return fallback or []

        embeddings = tracklet.get("embeddings", [])
        if len(embeddings) < TRACKLET_MIN_EMBEDDINGS_FOR_MATCH:
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
        if candidate_score < SIMILARITY_THRESHOLD:
            return False, "below_threshold"

        if (
            ENABLE_ID_STICKINESS
            and previous_gid is not None
            and previous_gid in self._gallery
            and candidate_gid != previous_gid
        ):
            previous_score = self._score_gid(previous_gid, embedding)
            if candidate_score < previous_score + ID_SWITCH_MARGIN:
                return (
                    False,
                    f"switch_margin(prev=G{previous_gid},"
                    f"prev_score={previous_score:.3f})",
                )

        if (
            ENABLE_AMBIGUOUS_MATCH_REJECTION
            and candidate_gid != previous_gid
        ):
            runner_up = max(
                (score for gid, score in ranked if gid != candidate_gid),
                default=0.0,
            )
            if runner_up > 0.0 and candidate_score < runner_up + MATCH_AMBIGUITY_MARGIN:
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
        if self._geometry is None or GEO_WEIGHT <= 0.0:
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

        return (1.0 - GEO_WEIGHT) * reid_score + GEO_WEIGHT * best_geo

    def _assignment_score(self, reid_score: float, best_reid_score: float,
                          row: dict, candidate_gid: int) -> float:
        if GEO_ASSIGNMENT_MODE == "close_reid_only":
            if best_reid_score - reid_score > GEO_REID_MARGIN:
                return reid_score
        return self._blend_geo_score(reid_score, row, candidate_gid)

    @staticmethod
    def _use_prototypes() -> bool:
        return GALLERY_MAX_PROTOTYPES > 0

    @staticmethod
    def _new_gallery_entry() -> dict:
        if CrossCameraGalleryProbe._use_prototypes():
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
            or same_src_score < PROTOTYPE_ADD_THRESHOLD
            or all_score < PROTOTYPE_ADD_THRESHOLD
        )
        if not should_add:
            return

        prototypes.append({
            "embedding": np.asarray(embedding, dtype=np.float32),
            "src": src,
            "last_seen": self._frame_count,
        })
        if len(prototypes) > GALLERY_MAX_PROTOTYPES:
            # Keep the most recent prototypes so the gallery can adapt without
            # collapsing to a single latest embedding.
            del prototypes[:-GALLERY_MAX_PROTOTYPES]


# =============================================================================
# CLI integration: apply argparse overrides + log the active configuration.
# =============================================================================
def configure_from_args(args) -> None:
    """Apply main-app CLI overrides onto this module's tuning globals."""
    global SIMILARITY_THRESHOLD
    global GALLERY_MAX_AGE, GLOBAL_ASSIGNMENT_MAX_CANDIDATES
    global ENABLE_ID_STICKINESS, ID_SWITCH_MARGIN
    global ENABLE_AMBIGUOUS_MATCH_REJECTION, MATCH_AMBIGUITY_MARGIN
    global ENABLE_GLOBAL_ID_MERGE, GLOBAL_ID_MERGE_THRESHOLD
    global GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS, GLOBAL_ID_MERGE_MARGIN
    global GLOBAL_ID_MERGE_INTERVAL, GLOBAL_ID_MERGE_MAX_CANDIDATES
    global USE_TRACKLET_EMBEDDING, TRACKLET_MAX_EMBEDDINGS
    global TRACKLET_MIN_EMBEDDINGS_FOR_MATCH, TRACKLET_MAX_AGE
    global TRACKLET_EMBEDDING_INTERVAL, ENABLE_EMBEDDING_QUALITY_GATE
    global GEO_WEIGHT, GEO_ASSIGNMENT_MODE, GEO_REID_MARGIN

    SIMILARITY_THRESHOLD = max(0.0, args.similarity_threshold)
    GALLERY_MAX_AGE = max(1, args.gallery_max_age)
    GLOBAL_ASSIGNMENT_MAX_CANDIDATES = max(1, args.assignment_max_candidates)
    ENABLE_ID_STICKINESS = not args.disable_id_stickiness
    ID_SWITCH_MARGIN = max(0.0, args.id_switch_margin)
    ENABLE_AMBIGUOUS_MATCH_REJECTION = not args.allow_ambiguous_match
    MATCH_AMBIGUITY_MARGIN = max(0.0, args.match_ambiguity_margin)
    ENABLE_GLOBAL_ID_MERGE = not args.disable_global_merge
    GLOBAL_ID_MERGE_THRESHOLD = max(0.0, args.global_merge_threshold)
    GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS = max(
        1, args.global_merge_min_embeddings)
    GLOBAL_ID_MERGE_MARGIN = max(0.0, args.global_merge_margin)
    GLOBAL_ID_MERGE_INTERVAL = max(1, args.global_merge_interval)
    GLOBAL_ID_MERGE_MAX_CANDIDATES = max(1, args.global_merge_max_candidates)
    USE_TRACKLET_EMBEDDING = not args.disable_tracklet
    TRACKLET_EMBEDDING_INTERVAL = max(1, args.tracklet_embedding_interval)
    ENABLE_EMBEDDING_QUALITY_GATE = not args.disable_embedding_quality_gate
    TRACKLET_MAX_EMBEDDINGS = max(1, args.tracklet_window)
    TRACKLET_MIN_EMBEDDINGS_FOR_MATCH = max(1, args.tracklet_min_embeddings)
    TRACKLET_MAX_AGE = max(1, args.tracklet_max_age)
    _gw = getattr(args, "geo_weight", None)
    if _gw is not None:
        GEO_WEIGHT = max(0.0, min(1.0, float(_gw)))
    _mode = getattr(args, "geometry_assignment_mode", GEO_ASSIGNMENT_MODE)
    if _mode in {"weight_only", "close_reid_only"}:
        GEO_ASSIGNMENT_MODE = _mode
    _margin = getattr(args, "geometry_reid_margin", None)
    if _margin is not None:
        GEO_REID_MARGIN = max(0.0, float(_margin))


def config_summary() -> str:
    """Multi-line summary of the active ReID/Global-ID tuning for startup logs."""
    return "\n".join([
        f"[reid] Re-ID similarity threshold={SIMILARITY_THRESHOLD}",
        f"[reid] gallery_max_age={GALLERY_MAX_AGE}",
        f"[reid] assignment_max_candidates={GLOBAL_ASSIGNMENT_MAX_CANDIDATES}",
        (f"[reid] id_stickiness={ENABLE_ID_STICKINESS} "
         f"switch_margin={ID_SWITCH_MARGIN} "
         f"ambiguous_match_rejection={ENABLE_AMBIGUOUS_MATCH_REJECTION} "
         f"ambiguity_margin={MATCH_AMBIGUITY_MARGIN}"),
        (f"[reid] global_id_merge={ENABLE_GLOBAL_ID_MERGE} "
         f"threshold={GLOBAL_ID_MERGE_THRESHOLD} "
         f"min_tracklet_embeddings={GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS} "
         f"margin={GLOBAL_ID_MERGE_MARGIN} "
         f"interval={GLOBAL_ID_MERGE_INTERVAL} "
         f"max_candidates={GLOBAL_ID_MERGE_MAX_CANDIDATES}"),
        (f"[reid] tracklet_embedding={USE_TRACKLET_EMBEDDING} "
         f"window={TRACKLET_MAX_EMBEDDINGS} "
         f"min_embeddings={TRACKLET_MIN_EMBEDDINGS_FOR_MATCH} "
         f"sample_interval={TRACKLET_EMBEDDING_INTERVAL} "
         f"max_age={TRACKLET_MAX_AGE}"),
        (f"[reid] embedding_quality_gate={ENABLE_EMBEDDING_QUALITY_GATE} "
         f"edge_margin={REID_EDGE_MARGIN_RATIO} "
         f"max_overlap_iou={REID_MAX_OVERLAP_IOU_FOR_UPDATE}"),
        (f"[reid] geo_weight={GEO_WEIGHT} "
         f"assignment_mode={GEO_ASSIGNMENT_MODE} "
         f"reid_margin={GEO_REID_MARGIN}"),
    ])
