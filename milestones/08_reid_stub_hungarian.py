"""
=============================================================================
MILESTONE 8 — Cross-Camera Person Re-Identification + Hungarian Assignment
=============================================================================

WHAT YOU LEARN:
  - Why per-camera tracking IDs are not enough (same person = different IDs)
  - How NvDeepSORT uses embedding similarity for more stable tracking
  - How to build a cross-camera gallery in Python using a metadata probe
  - What "cosine similarity" means for feature vectors
  - The full pipeline: detect → track(DeepSORT) → [ReID gallery probe] → OSD

PROBLEM THIS SOLVES:
  Milestones 1–7: Person #42 in cam0 exits → enters cam1 → becomes Person #7.
  The tracker forgets them completely between cameras.

  Milestone 8: We build a Python-side gallery that links cam0:#42 and cam1:#7
  as the same physical person (Global ID #1).

                   cam0 ──→ track_id=42  ─┐
                                           ├─ embedding match → global_id=1
                   cam1 ──→ track_id=7   ─┘

TWO-LEVEL APPROACH IN THIS MILESTONE:
  Level 1 — NvDeepSORT tracker (configs/tracker/nvdeepsort_reid.yaml):
    Runs a Re-ID model on each person crop. Uses embedding similarity to
    maintain stable IDs within each stream (better than NvDCF on occlusion).

  Level 2 — CrossCameraGalleryProbe (Python probe, this file):
    Reads NvDeepSORT embeddings from tracker ReID metadata.
    Matches each (cam, track_id) against a cross-camera gallery using:
      - tracklet embedding averages for noisy frame-level ReID vectors
      - per-identity prototypes for different camera views
      - Hungarian assignment so one Global ID is not duplicated per stream
      - ID stickiness and ambiguity rejection to prevent ID bouncing
      - online Global ID merge to repair stable cross-view ID splits
      - bounded candidate search to keep long videos responsive
    Assigns a stable "global_id" that persists across cameras.

PIPELINE TOPOLOGY:
  [src_0..N] → [mux] → [nvinfer] → [nvtracker/DeepSORT] → [tiler]
                                           │                    │
                                  [SourceIdCollectorProbe] [CrossCameraGalleryProbe]
                                   (pre-tiler: source_id)  (post-tiler: draw labels)
                                                                 ↓
                                                           [nvosdbin] → [sink]

SETUP:
  The Re-ID model is already downloaded:
    models/reid/resnet50_market1501.etlt

  No manual engine build needed — DeepStream decodes the .etlt and builds the
  TensorRT engine automatically on first run (~1 min). The engine is saved next
  to the model under models/reid/. Subsequent runs load it in seconds.

  To skip Re-ID (e.g. test pipeline first):
    --tracker-config configs/tracker/nvdcf_perf.yaml
  Gallery probe still runs but embeddings are empty → no cross-cam matching.

RUN:
  python milestones/08_reid_stub_hungarian.py
  python milestones/08_reid_stub_hungarian.py --debug-similarity
  python milestones/08_reid_stub_hungarian.py --disable-global-merge

TODO EXERCISES:
  1. Run with nvdcf_perf.yaml first — gallery runs, cross-cam labels show G#N
     but IDs will split (no real embeddings). Good for testing the pipeline.
  2. Run with nvdeepsort_reid.yaml — first run builds Re-ID engine (~1 min).
     Watch console: "Re-ID: Cam1#7 → G#42 (similarity=0.81)" means a match.
  3. Walk a person from cam0 to cam1. Check if global_id stays the same.
  4. If IDs bounce between two labels, tune ID_SWITCH_MARGIN and
     MATCH_AMBIGUITY_MARGIN.
  5. If one person becomes two stable Global IDs across opposite cameras,
     tune GLOBAL_ID_MERGE_THRESHOLD and GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS.
  6. If long videos lag, lower GALLERY_MAX_AGE and the candidate limits.
=============================================================================
"""

import argparse
import math
from pathlib import Path
import sys
import traceback

import pyservicemaker as psm

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[M8] WARNING: torch not found — Re-ID embeddings will be empty. "
          "Install: pip install torch")

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
    infer_source_id_from_tiled_box,
    set_object_label,
)
from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


# =============================================================================
# ReID / Global-ID Tuning
# =============================================================================
#
# Match threshold:
#   Higher -> fewer false matches, but more ID splits.
#   Lower  -> easier to reconnect the same person, but more merge risk.
SIMILARITY_THRESHOLD = 0.6

# Gallery memory:
#   Gallery stores known Global IDs after a local track disappears.
#   Increase age if people leave/re-enter after a long gap.
#   Decrease age if old identities are reused incorrectly.
GALLERY_MAX_AGE = 1800

# Gallery prototypes:
#   Each Global ID can keep multiple appearance vectors for different views.
#   More prototypes improve view coverage but increase matching cost.
#   Set to 0 to disable multi-prototype mode and keep one vector per Global ID.
GALLERY_MAX_PROTOTYPES = 6

# Prototype admission:
#   Add a new prototype when the current embedding is visually different enough.
#   Higher -> compact gallery, less noise.
#   Lower  -> more view coverage, more chance of storing bad crops.
PROTOTYPE_ADD_THRESHOLD = 0.7

# Assignment:
#   Hungarian solves one-to-one assignment for new local tracks within a stream.
#   This prevents multiple people in the same camera from selecting one Global ID.
USE_HUNGARIAN_ASSIGNMENT = True
GLOBAL_ASSIGNMENT_MAX_CANDIDATES = 80

# Duplicate guard:
#   Keeps already-known local tracks from displaying the same Global ID twice in
#   one stream. Keep this on with Hungarian; disable only for A/B debugging.
ENFORCE_UNIQUE_GLOBAL_PER_STREAM = True

# ID stickiness:
#   A local track that already has a Global ID should not switch to another ID
#   unless the new candidate is clearly better than the current one.
#   This prevents labels from bouncing between two visually similar IDs.
ENABLE_ID_STICKINESS = True
ID_SWITCH_MARGIN = 0.1

# Ambiguous match rejection:
#   For a new/released local track, accept an existing Global ID only when the
#   best match beats the runner-up by this margin. If G14=0.64 and G8=0.62,
#   create/keep a separate ID instead of randomly bouncing between them.
ENABLE_AMBIGUOUS_MATCH_REJECTION = True
MATCH_AMBIGUITY_MARGIN = 0.05

# Global-ID merge:
#   A cross-view track may first become a new Global ID because the opposite
#   camera crop looks very different. After enough tracklet evidence, merge the
#   duplicate ID into the best older Global ID if the match is strong and does
#   not create two copies of one Global ID in the same stream frame.
ENABLE_GLOBAL_ID_MERGE = True
GLOBAL_ID_MERGE_THRESHOLD = 0.68
GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS = 8
GLOBAL_ID_MERGE_MARGIN = 0.05
GLOBAL_ID_MERGE_INTERVAL = 15
GLOBAL_ID_MERGE_MAX_CANDIDATES = 40

# Tracklet embedding:
#   Tracklet mode averages recent embeddings for each (camera, local_track_id).
#   This is more stable than matching on a single noisy frame crop.
USE_TRACKLET_EMBEDDING = True

# Tracklet memory:
#   Drop inactive local tracklets after this many batches.
TRACKLET_MAX_AGE = 1800

# Tracklet smoothing window:
#   Number of recent embeddings kept per local track.
#   Larger -> smoother but slower to adapt if tracker switches identity.
TRACKLET_MAX_EMBEDDINGS = 16

# Tracklet warmup:
#   Use raw frame embedding until the local tracklet has this many embeddings.
#   Larger -> more stable first match, but slower cross-camera linking.
TRACKLET_MIN_EMBEDDINGS_FOR_MATCH = 8

# Debug:
#   Number of nearest Global IDs printed when --debug-similarity is enabled.
DEBUG_TOP_K = 3


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0–1.0."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(dot / (norm_a * norm_b))
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

    mean = [sum(e[i] for e in same_dim) / len(same_dim) for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5
    if norm == 0.0:
        return mean
    return [float(x / norm) for x in mean]


def max_weight_assignment(weights: list[list[float]]) -> list[int]:
    """
    Hungarian assignment for max-weight rectangular matrices.

    Returns a list where result[row] = assigned column. The implementation uses
    the classic O(n^2*m) shortest augmenting path form for min-cost assignment,
    converting max weights to costs internally. It assumes columns >= rows; this
    milestone always provides enough "new identity" dummy columns.
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
                 debug: bool = False):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
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
            print("[M8-H ERROR] SourceIdCollectorProbe failed:")
            traceback.print_exc()

    def _handle_metadata(self, batch_meta):
        self._frame_count += 1
        batch_persons = 0
        batch_embeddings = 0
        batch_obj_reids = 0
        batch_obj_tensors = 0
        batch_frame_tensors = 0

        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            frame_tensor_count = self._count_iter(frame_meta.tensor_items)
            batch_frame_tensors += frame_tensor_count
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                oid = obj_meta.object_id
                self._id_map[oid] = src
                batch_persons += 1

                # Try to extract Re-ID embedding from tracker metadata.
                # NvDeepSORT exposes this via obj_meta.obj_reid_items in
                # pyservicemaker. If not available (e.g. NvDCF tracker),
                # embedding stays empty.
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
                f"[M8 tensor debug] frame={self._frame_count:06d} "
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
                 enforce_unique_per_stream: bool = True):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        self._gallery: dict[int, dict] = {}      # global_id → gallery entry
        self._track_to_gid: dict[tuple, int] = {}  # (src, track_id) → global_id
        self._tracklets: dict[tuple, dict] = {}   # (src, track_id) → state
        self._next_gid = 1
        self._frame_count = 0
        self._debug_similarity = debug_similarity
        self._use_hungarian_assignment = use_hungarian_assignment
        self._enforce_unique_per_stream = enforce_unique_per_stream

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            print("[M8-H ERROR] CrossCameraGalleryProbe failed:")
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
                src = infer_source_id_from_tiled_box(
                    obj_meta.rect_params, self._tile_w, self._tile_h,
                    self._cols, self._num_sources)
                embedding = self._embeddings.get((src, oid), [])
                track_key = (src, oid)
                tracklet = self._update_tracklet(
                    track_key, src, oid, embedding)
                match_embedding = self._tracklet_embedding(
                    tracklet, fallback=embedding)
                previous_gid = self._track_to_gid.get(track_key)
                if previous_gid is None:
                    previous_gid = tracklet.get("gid")
                if previous_gid not in self._gallery:
                    previous_gid = None
                gid = previous_gid

                rows.append({
                    "src": src,
                    "track_id": oid,
                    "track_key": track_key,
                    "embedding": match_embedding,
                    "raw_embedding": embedding,
                    "tracklet_len": len(tracklet["embeddings"]),
                    "gid": gid,
                    "previous_gid": previous_gid,
                })

            if self._use_hungarian_assignment:
                if self._enforce_unique_per_stream:
                    self._release_duplicate_known_assignments(rows)
                self._assign_new_tracks_with_hungarian(rows, log)
            else:
                self._assign_new_tracks_greedy(rows, log)

            for row in rows:
                gid = row["gid"]
                self._track_to_gid[row["track_key"]] = gid
                self._tracklets[row["track_key"]]["gid"] = gid
                if row.get("gallery_updated") is not True:
                    self._update_gallery(gid, row["embedding"], row["src"])

            if (
                ENABLE_GLOBAL_ID_MERGE
                and self._frame_count % GLOBAL_ID_MERGE_INTERVAL == 0
            ):
                self._merge_duplicate_global_ids(rows, log)

            label_by_track = {
                row["track_id"]: (
                    # f"Global:{row['gid']}|Cam:{row['src']}|Person:{row['track_id']}"
                    f"Global:{row['gid']}"
                )
                for row in rows
            }
            for obj_meta in frame_meta.object_items:
                label = label_by_track.get(obj_meta.object_id)
                if label is not None:
                    set_object_label(obj_meta, label)

        # Age gallery once per batch
        for v in self._gallery.values():
            v["age"] += 1
        for tracklet in self._tracklets.values():
            tracklet["age"] += 1

        if log:
            active = len(self._gallery)
            active_tracklets = len(self._tracklets)
            print(f"[M8] frame={self._frame_count:06d}  "
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

        # New person
        gid = self._allocate_new_gid()
        self._gallery[gid] = self._new_gallery_entry()
        return gid

    def _release_duplicate_known_assignments(self, rows: list[dict]) -> None:
        """Move duplicate known same-stream global IDs back to assignment."""
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
            if row_score > existing_score:
                released = existing
                active[key] = row
            else:
                released = row

            released["gid"] = None
            if self._debug_similarity:
                print(
                    f"  [Re-ID Hungarian] Cam{released['src']}#{released['track_id']} "
                    f"released_duplicate=G{gid} "
                    f"held_by=Cam{active[key]['src']}#{active[key]['track_id']} "
                    f"released_score={self._score_gid(gid, released['embedding']):.3f} "
                    f"held_score={self._score_gid(gid, active[key]['embedding']):.3f}"
                )

    def _assign_new_tracks_greedy(self, rows: list[dict], log: bool) -> None:
        for row in rows:
            if row["gid"] is None:
                row["gid"] = self._find_or_create(
                    row["embedding"], row["src"], row["track_id"], log,
                    row["tracklet_len"], row.get("previous_gid"))
                # Match the original milestone behavior: once a new track is
                # assigned, later detections in the same tiled frame can match it.
                self._update_gallery(row["gid"], row["embedding"], row["src"])
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
            if row["gid"] is None:
                rows_by_src.setdefault(src, []).append(row)
            else:
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
                row_weights = []
                for kind, value in columns:
                    if kind == "gid":
                        score = self._score_gid(value, row["embedding"])
                        ranked = [
                            (gid, self._score_gid(gid, row["embedding"]))
                            for gid in existing_gids
                        ]
                        ranked.sort(key=lambda item: item[1], reverse=True)
                        allowed, _ = self._is_gid_match_allowed(
                            row["embedding"], value, row.get("previous_gid"),
                            ranked)
                        row_weights.append(score if allowed else -1.0)
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
                self._update_gallery(row["gid"], row["embedding"], row["src"])
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
                source_gid, row["embedding"], row["src"], active_by_src)
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

    def _best_merge_candidate(self, source_gid: int, embedding: list[float],
                              src: int,
                              active_by_src: dict[int, set[int]]
                              ) -> tuple[int, float, float] | None:
        candidates = self._candidate_gids(
            exclude=active_by_src.get(src, set()),
            max_count=GLOBAL_ID_MERGE_MAX_CANDIDATES,
            only_older_than=source_gid,
        )

        scores = []
        for target_gid in candidates:
            scores.append((target_gid, self._score_gid(target_gid, embedding)))

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
            if source_embedding:
                target["embedding"] = source_embedding
            return

        target_prototypes = target.setdefault("prototypes", [])
        target_prototypes.extend(source.get("prototypes", []))
        target_prototypes.sort(key=lambda p: p.get("last_seen", 0))
        if len(target_prototypes) > GALLERY_MAX_PROTOTYPES:
            del target_prototypes[:-GALLERY_MAX_PROTOTYPES]

    def _update_tracklet(self, track_key: tuple, src: int, track_id: int,
                         embedding: list[float]) -> dict:
        tracklet = self._tracklets.setdefault(track_key, {
            "src": src,
            "track_id": track_id,
            "embeddings": [],
            "age": 0,
            "gid": self._track_to_gid.get(track_key),
        })
        tracklet["age"] = 0
        tracklet["src"] = src
        tracklet["track_id"] = track_id
        if embedding:
            tracklet["embeddings"].append(embedding)
            if len(tracklet["embeddings"]) > TRACKLET_MAX_EMBEDDINGS:
                del tracklet["embeddings"][:-TRACKLET_MAX_EMBEDDINGS]
        return tracklet

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

    @staticmethod
    def _use_prototypes() -> bool:
        return GALLERY_MAX_PROTOTYPES > 0

    @staticmethod
    def _new_gallery_entry() -> dict:
        if CrossCameraGalleryProbe._use_prototypes():
            return {"prototypes": [], "age": 0}
        return {"embedding": [], "age": 0}

    @staticmethod
    def _best_prototype_score(embedding: list[float], entry: dict,
                              src: int | None = None) -> float:
        prototypes = entry.get("prototypes", [])
        if src is not None:
            prototypes = [p for p in prototypes if p.get("src") == src]
        if not prototypes:
            return 0.0
        return max(_cosine_similarity(embedding, p["embedding"])
                   for p in prototypes)

    def _update_gallery(self, gid: int, embedding: list[float], src: int) -> None:
        """Refresh a global identity using single-vector or prototype mode."""
        entry = self._gallery.setdefault(gid, self._new_gallery_entry())
        entry["age"] = 0
        if not embedding:
            return

        if not self._use_prototypes():
            entry["embedding"] = embedding
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
            "embedding": embedding,
            "src": src,
            "last_seen": self._frame_count,
        })
        if len(prototypes) > GALLERY_MAX_PROTOTYPES:
            # Keep the most recent prototypes so the gallery can adapt without
            # collapsing to a single latest embedding.
            del prototypes[:-GALLERY_MAX_PROTOTYPES]


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def add_recording_branch(pipeline: psm.Pipeline, upstream: str,
                         output_path: str, bitrate: int) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pipeline.add("queue", "record_queue", {"leaky": 0})
    pipeline.add("nvvideoconvert", "record_convert", {"gpu-id": 0})
    pipeline.add("nvv4l2h264enc", "record_encoder", {
        "bitrate": bitrate,
        "insert-sps-pps": 1,
    })
    pipeline.add("h264parse", "record_h264parse")
    pipeline.add("qtmux", "record_mux")
    pipeline.add("filesink", "record_sink", {
        "location": str(out),
        "sync": 0,
        "async": 0,
    })
    pipeline.link(
        upstream,
        "record_queue",
        "record_convert",
        "record_encoder",
        "record_h264parse",
        "record_mux",
        "record_sink",
    )
    return str(out)


def run(sources_txt: str, nvinfer_config: str, tracker_config: str,
        tile_w: int, tile_h: int, debug_similarity: bool,
        use_hungarian_assignment: bool, enforce_unique_per_stream: bool,
        save_video: str | None, record_bitrate: int, no_display: bool):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    person_class_id = infer_person_class_id(nvinfer_config)
    total_w, total_h = tile_w * cols, tile_h * rows
    print(f"[M8-H] {n} stream(s) → {rows}×{cols} grid  canvas={total_w}×{total_h}")
    print(f"[M8] tracker={tracker_config}")
    print(f"[M8] person_class_id={person_class_id} inferred from {nvinfer_config}")
    print(f"[M8] Re-ID similarity threshold={SIMILARITY_THRESHOLD}")
    print(f"[M8] gallery_max_age={GALLERY_MAX_AGE}")
    print(f"[M8] debug_similarity={debug_similarity}")
    print(f"[M8] use_hungarian_assignment={use_hungarian_assignment}")
    print(f"[M8] enforce_unique_per_stream={enforce_unique_per_stream}")
    print(f"[M8] assignment_max_candidates={GLOBAL_ASSIGNMENT_MAX_CANDIDATES}")
    print(
        f"[M8] id_stickiness={ENABLE_ID_STICKINESS} "
        f"switch_margin={ID_SWITCH_MARGIN} "
        f"ambiguous_match_rejection={ENABLE_AMBIGUOUS_MATCH_REJECTION} "
        f"ambiguity_margin={MATCH_AMBIGUITY_MARGIN}"
    )
    print(
        f"[M8] global_id_merge={ENABLE_GLOBAL_ID_MERGE} "
        f"threshold={GLOBAL_ID_MERGE_THRESHOLD} "
        f"min_tracklet_embeddings={GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS} "
        f"margin={GLOBAL_ID_MERGE_MARGIN} "
        f"interval={GLOBAL_ID_MERGE_INTERVAL} "
        f"max_candidates={GLOBAL_ID_MERGE_MAX_CANDIDATES}"
    )
    print(
        f"[M8] tracklet_embedding={USE_TRACKLET_EMBEDDING} "
        f"window={TRACKLET_MAX_EMBEDDINGS} "
        f"min_embeddings={TRACKLET_MIN_EMBEDDINGS_FOR_MATCH} "
        f"max_age={TRACKLET_MAX_AGE}"
    )
    if save_video:
        print(f"[M8] save_video={save_video}")

    id_map: dict[int, int] = {}
    embeddings: dict[tuple, list] = {}  # (source_id, object_id) → embedding vector

    pipeline = psm.Pipeline("m8-reid-hungarian")

    pipeline.add("nvstreammux", "mux", {
        "batch-size": n, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": 0,
    })
    for i, uri in enumerate(uris):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": uri, "gpu-id": 0})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": n, "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384,
        "gpu-id": 0,
    })

    # Probe 1 (pre-tiler): source_id valid, extract embeddings
    pipeline.attach("tracker", psm.Probe(
        "src_collector",
        SourceIdCollectorProbe(
            id_map, embeddings, person_class_id, debug=debug_similarity),
    ))

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": total_w, "height": total_h, "gpu-id": 0,
    })

    # Probe 2 (post-tiler): tiled coords, run gallery matching + draw labels
    gallery_probe = CrossCameraGalleryProbe(
        id_map, embeddings, person_class_id, tile_w, tile_h, cols, n,
        debug_similarity=debug_similarity,
        use_hungarian_assignment=use_hungarian_assignment,
        enforce_unique_per_stream=enforce_unique_per_stream)
    pipeline.attach("tiler", psm.Probe("reid_probe", gallery_probe))

    pipeline.add("nvosdbin", "osd", {
        "gpu-id": 0,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size": 18,
    })
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")

    if save_video and not no_display:
        pipeline.add("tee", "output_tee")
        pipeline.add("queue", "display_queue")
        pipeline.add(get_sink_element(), "sink", {"sync": 1, "qos": 0})
        pipeline.link("osd", "output_tee", "display_queue", "sink")
        written_path = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate)
    elif save_video:
        written_path = add_recording_branch(
            pipeline, "osd", save_video, record_bitrate)
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": 1, "qos": 0})
        pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[M8] Running. Gallery stats print every 60 frames.")
        print("[M8] Labels show Global:<global_id>.")
        if save_video:
            print(f"[M8] Recording annotated video to: {written_path}")
        print("[M8] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M8] Stopped.")
        total_gids = gallery_probe._next_gid - 1
        print(f"[M8] Total unique global IDs assigned: {total_gids}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 8: Cross-camera Re-ID with Hungarian assignment")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_yolov8_people.yml",
                        help="nvinfer config. Default: YOLOv8. "
                             "Alternatives: configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml, "
                             "configs/models/nvinfer_yolov8_people.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdeepsort_reid.yaml",
                        help="Use nvdcf_perf.yaml to run without Re-ID engine")
    parser.add_argument("--tile-w", type=int, default=1280)
    parser.add_argument("--tile-h", type=int, default=720)
    parser.add_argument("--debug-similarity", action="store_true",
                        help="Print max cosine similarity for every new track")
    parser.add_argument("--gallery-max-age", type=int,
                        default=GALLERY_MAX_AGE,
                        help="Drop inactive Global IDs after this many batches")
    parser.add_argument("--disable-hungarian", action="store_true",
                        help="Use greedy gallery matching instead of per-stream "
                             "Hungarian assignment")
    parser.add_argument("--assignment-max-candidates", type=int,
                        default=GLOBAL_ASSIGNMENT_MAX_CANDIDATES,
                        help="Limit gallery IDs scanned by Hungarian assignment")
    parser.add_argument("--allow-duplicate-gid-per-stream", action="store_true",
                        help="Disable the guard that releases duplicate known "
                             "Global IDs within one stream before Hungarian")
    parser.add_argument("--disable-id-stickiness", action="store_true",
                        help="Allow a known local track to switch Global IDs "
                             "without requiring an extra similarity margin")
    parser.add_argument("--id-switch-margin", type=float,
                        default=ID_SWITCH_MARGIN,
                        help="Extra similarity required before switching from "
                             "a previous Global ID to another one")
    parser.add_argument("--allow-ambiguous-match", action="store_true",
                        help="Allow matching an existing Global ID even when "
                             "the runner-up ID is very close")
    parser.add_argument("--match-ambiguity-margin", type=float,
                        default=MATCH_AMBIGUITY_MARGIN,
                        help="Best existing Global ID must beat runner-up by "
                             "this margin before it is accepted")
    parser.add_argument("--disable-global-merge", action="store_true",
                        help="Disable online merging of duplicate Global IDs")
    parser.add_argument("--global-merge-threshold", type=float,
                        default=GLOBAL_ID_MERGE_THRESHOLD,
                        help="Similarity threshold to merge one Global ID into another")
    parser.add_argument("--global-merge-min-embeddings", type=int,
                        default=GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS,
                        help="Minimum local tracklet embeddings before merge is considered")
    parser.add_argument("--global-merge-margin", type=float,
                        default=GLOBAL_ID_MERGE_MARGIN,
                        help="Merge candidate must beat runner-up by this margin")
    parser.add_argument("--global-merge-interval", type=int,
                        default=GLOBAL_ID_MERGE_INTERVAL,
                        help="Run duplicate Global ID merge every N batches")
    parser.add_argument("--global-merge-max-candidates", type=int,
                        default=GLOBAL_ID_MERGE_MAX_CANDIDATES,
                        help="Limit older Global IDs scanned per merge candidate")
    parser.add_argument("--disable-tracklet", action="store_true",
                        help="Use only current-frame embeddings, without local "
                             "tracklet averaging")
    parser.add_argument("--tracklet-window", type=int,
                        default=TRACKLET_MAX_EMBEDDINGS,
                        help="Number of recent embeddings kept per (camera, track)")
    parser.add_argument("--tracklet-min-embeddings", type=int,
                        default=TRACKLET_MIN_EMBEDDINGS_FOR_MATCH,
                        help="Minimum embeddings before using the averaged tracklet "
                             "vector for matching")
    parser.add_argument("--tracklet-max-age", type=int,
                        default=TRACKLET_MAX_AGE,
                        help="Drop inactive local tracklets after this many batches")
    parser.add_argument("--save-video", nargs="?", const="output/videos/m8_reid.mp4",
                        default=None,
                        help="Save annotated output MP4. Default path when no value is "
                             "given: output/videos/m8_reid.mp4")
    parser.add_argument("--record-bitrate", type=int, default=8000000,
                        help="H.264 recording bitrate in bits/sec")
    parser.add_argument("--no-display", action="store_true",
                        help="Only valid with --save-video: record without opening a window")
    args = parser.parse_args()
    enforce_unique = (
        ENFORCE_UNIQUE_GLOBAL_PER_STREAM
        and not args.allow_duplicate_gid_per_stream
    )
    use_hungarian = USE_HUNGARIAN_ASSIGNMENT and not args.disable_hungarian
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
    TRACKLET_MAX_EMBEDDINGS = max(1, args.tracklet_window)
    TRACKLET_MIN_EMBEDDINGS_FOR_MATCH = max(1, args.tracklet_min_embeddings)
    TRACKLET_MAX_AGE = max(1, args.tracklet_max_age)
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h, args.debug_similarity, use_hungarian,
        enforce_unique, args.save_video, args.record_bitrate, args.no_display)
