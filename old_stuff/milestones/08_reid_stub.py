"""
=============================================================================
MILESTONE 8 — Cross-Camera Person Re-Identification
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
    Matches each (cam, track_id) against a cross-camera gallery.
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
  python milestones/08_reid_stub.py
  python milestones/08_reid_stub.py --tracker-config configs/tracker/nvdcf_perf.yaml

TODO EXERCISES:
  1. Run with nvdcf_perf.yaml first — gallery runs, cross-cam labels show G#N
     but IDs will split (no real embeddings). Good for testing the pipeline.
  2. Run with nvdeepsort_reid.yaml — first run builds Re-ID engine (~1 min).
     Watch console: "Re-ID: Cam1#7 → G#42 (similarity=0.81)" means a match.
  3. Walk a person from cam0 to cam1. Check if global_id stays the same.
  4. Tune SIMILARITY_THRESHOLD: lower = more matches (more false positives),
     higher = fewer matches (more identity splits). Try 0.3 and 0.7.
  5. Add trajectory logging: save (global_id, cam, frame, x, y) to CSV.
=============================================================================
"""

import argparse
import math
from pathlib import Path
import sys

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


SIMILARITY_THRESHOLD = 0.5  # cosine similarity to accept a Re-ID match
GALLERY_MAX_AGE = 1000       # drop gallery entry after N frames without a match
GALLERY_MAX_PROTOTYPES = 8    # 0 disables multi-prototype and keeps one vector
PROTOTYPE_ADD_THRESHOLD = 0.6  # add a new prototype if it is visually distinct
ENFORCE_UNIQUE_GLOBAL_PER_STREAM = True
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
    return float(dot / (norm_a * norm_b))


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
      1. If (src, track_id) already mapped → reuse global_id, update embedding
      2. Else → compare embedding against all gallery prototypes
             match ≥ threshold → reuse that global_id
             no match         → new global_id
      3. Draw "G#{global_id} Cam{src}#{track_id}" on screen

    WHY track_to_gid works:
      Within one camera a tracker ID is stable while the person is visible.
      Cross-camera: different src → different key → forces embedding match,
      which is how global IDs link across cameras.
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int,
                 debug_similarity: bool = False,
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
        self._next_gid = 1
        self._frame_count = 0
        self._debug_similarity = debug_similarity
        self._enforce_unique_per_stream = enforce_unique_per_stream

    def handle_metadata(self, batch_meta):
        self._frame_count += 1
        log = self._frame_count % 60 == 0
        active_gid_by_stream: dict[tuple[int, int], dict] = {}

        # Expire stale gallery entries
        stale = [gid for gid, v in self._gallery.items()
                 if v["age"] > GALLERY_MAX_AGE]
        for gid in stale:
            del self._gallery[gid]
            # Also clean stale track mappings pointing to this gid
            self._track_to_gid = {k: v for k, v in self._track_to_gid.items()
                                   if v != gid}

        for frame_meta in batch_meta.frame_items:
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue

                oid = obj_meta.object_id
                src = infer_source_id_from_tiled_box(
                    obj_meta.rect_params, self._tile_w, self._tile_h,
                    self._cols, self._num_sources)
                embedding = self._embeddings.get((src, oid), [])
                track_key = (src, oid)

                if track_key in self._track_to_gid:
                    # Known track — refresh age and update embedding
                    gid = self._track_to_gid[track_key]
                    if gid not in self._gallery:
                        gid = self._find_or_create(embedding, src, oid, log)
                        self._track_to_gid[track_key] = gid
                else:
                    # New (src, track_id) — try to match by embedding
                    gid = self._find_or_create(embedding, src, oid, log)
                    self._track_to_gid[track_key] = gid

                if self._enforce_unique_per_stream:
                    gid = self._ensure_unique_global_in_stream(
                        active_gid_by_stream, src, oid, track_key, gid,
                        embedding)
                self._update_gallery(gid, embedding, src)

                label = f"Global:{gid}|Cam:{src}|Person:{oid}"
                set_object_label(obj_meta, label)

        # Age gallery once per batch
        for v in self._gallery.values():
            v["age"] += 1

        if log:
            active = len(self._gallery)
            print(f"[M8] frame={self._frame_count:06d}  "
                  f"gallery={active}  total_gids_assigned={self._next_gid - 1}")

    def _find_or_create(self, embedding: list[float], src: int,
                        track_id: int, log: bool) -> int:
        """Match embedding against gallery; return existing or new global_id."""
        ranked = self._rank_gallery(embedding)
        best_gid = ranked[0][0] if ranked else -1
        best_score = ranked[0][1] if ranked else 0.0

        matched = best_score >= SIMILARITY_THRESHOLD and best_gid != -1
        reason = "no_embedding" if not embedding else (
            "empty_gallery" if best_gid == -1 else "below_threshold"
        )
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
                f"status={status} reason={display_reason} top{DEBUG_TOP_K}=[{top}]"
            )

        if matched:
            if log:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{best_gid} "
                      f"(similarity={best_score:.3f})")
            return best_gid

        # New person
        gid = self._next_gid
        self._next_gid += 1
        self._gallery[gid] = self._new_gallery_entry()
        return gid

    def _ensure_unique_global_in_stream(self, active: dict, src: int,
                                        track_id: int, track_key: tuple,
                                        gid: int,
                                        embedding: list[float]) -> int:
        """
        Prevent the same global ID from being drawn on two people in one stream.

        The first active track keeps the shared global ID. A later conflicting
        track is split into a new global ID, which is conservative but avoids
        the physically impossible state of one identity occupying two places in
        the same camera at the same time.
        """
        key = (src, gid)
        existing = active.get(key)
        if existing is None or existing["track_key"] == track_key:
            active[key] = {"track_key": track_key, "track_id": track_id}
            return gid

        new_gid = self._next_gid
        self._next_gid += 1
        self._gallery[new_gid] = self._new_gallery_entry()
        self._track_to_gid[track_key] = new_gid
        active[(src, new_gid)] = {"track_key": track_key, "track_id": track_id}

        if self._debug_similarity:
            score = self._score_gid(gid, embedding)
            print(
                f"  [Re-ID unique-stream] Cam{src}#{track_id} "
                f"conflicted_with=G{gid} held_by=Cam{src}#{existing['track_id']} "
                f"score={score:.3f} reassigned_to=G{new_gid}"
            )
        return new_gid

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
        enforce_unique_per_stream: bool, save_video: str | None,
        record_bitrate: int, no_display: bool):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    person_class_id = infer_person_class_id(nvinfer_config)
    total_w, total_h = tile_w * cols, tile_h * rows
    print(f"[M8] {n} stream(s) → {rows}×{cols} grid  canvas={total_w}×{total_h}")
    print(f"[M8] tracker={tracker_config}")
    print(f"[M8] person_class_id={person_class_id} inferred from {nvinfer_config}")
    print(f"[M8] Re-ID similarity threshold={SIMILARITY_THRESHOLD}")
    print(f"[M8] debug_similarity={debug_similarity}")
    print(f"[M8] enforce_unique_per_stream={enforce_unique_per_stream}")
    if save_video:
        print(f"[M8] save_video={save_video}")

    id_map: dict[int, int] = {}
    embeddings: dict[tuple, list] = {}  # (source_id, object_id) → embedding vector

    pipeline = psm.Pipeline("m8-reid")

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
        print("[M8] Labels show G#<global_id> Cam<src>#<track_id>.")
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
        description="Milestone 8: Cross-camera person Re-ID")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_yolov8_people.yml",
                        help="nvinfer config. Default: YOLOv8. "
                             "Alternatives: configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdeepsort_reid.yaml",
                        help="Use nvdcf_perf.yaml to run without Re-ID engine")
    parser.add_argument("--tile-w", type=int, default=1280)
    parser.add_argument("--tile-h", type=int, default=720)
    parser.add_argument("--debug-similarity", action="store_true",
                        help="Print max cosine similarity for every new track")
    parser.add_argument("--allow-duplicate-gid-per-stream", action="store_true",
                        help="Disable the guard that keeps each global ID unique "
                             "within one stream at the same frame")
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
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h, args.debug_similarity, enforce_unique,
        args.save_video, args.record_bitrate, args.no_display)
