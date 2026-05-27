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
    Reads NvDeepSORT embeddings from tensor metadata.
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
import sys

import pyservicemaker as psm
from pyservicemaker import osd

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
)
from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


SIMILARITY_THRESHOLD = 0.2  # cosine similarity to accept a Re-ID match
GALLERY_MAX_AGE = 300       # drop gallery entry after N frames without a match


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0–1.0."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Pre-tiler: source_id is valid here — collect it for each tracked person.
    Also extracts Re-ID embeddings from tensor metadata (NvDeepSORT output).

    Fills shared dicts:
      embeddings: (source_id, object_id) → embedding vector (list[float]) or []
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
        self._person_class_id = person_class_id

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                oid = obj_meta.object_id
                self._id_map[oid] = src

                # Try to extract Re-ID embedding from tensor metadata.
                # NvDeepSORT writes per-object tensor output readable via
                # obj_meta.tensor_items → as_tensor_output().get_layers().
                # If not available (e.g. NvDCF tracker), embedding stays empty.
                embedding: list[float] = []
                if _TORCH_AVAILABLE:
                    try:
                        for tensor_meta in obj_meta.tensor_items:
                            layers = tensor_meta.as_tensor_output().get_layers()
                            if layers:
                                raw = next(iter(layers.values()))
                                feat = torch.utils.dlpack.from_dlpack(raw)
                                embedding = feat.cpu().numpy().flatten().tolist()
                            break
                    except Exception:
                        pass
                self._embeddings[(src, oid)] = embedding


class CrossCameraGalleryProbe(psm.BatchMetadataOperator):
    """
    Post-tiler: tiled canvas coordinates — draw labels here.

    Maintains a cross-camera gallery:
      gallery:        global_id → {"embedding": [...], "age": int}
      track_to_gid:   (src, track_id) → global_id   ← stable while tracker holds the ID

    For each detected person each frame:
      1. If (src, track_id) already mapped → reuse global_id, update embedding
      2. Else → compare embedding against gallery (cosine similarity)
             match ≥ threshold → reuse that global_id
             no match         → new global_id
      3. Draw "G#{global_id} Cam{src}#{track_id}" on screen

    WHY track_to_gid works:
      Within one camera a tracker ID is stable while the person is visible.
      Cross-camera: different src → different key → forces embedding match,
      which is how global IDs link across cameras.
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        self._gallery: dict[int, dict] = {}      # global_id → {embedding, age}
        self._track_to_gid: dict[tuple, int] = {}  # (src, track_id) → global_id
        self._next_gid = 1
        self._frame_count = 0

    def handle_metadata(self, batch_meta):
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

        for frame_meta in batch_meta.frame_items:
            display_meta = batch_meta.acquire_display_meta()

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
                    if gid in self._gallery:
                        self._gallery[gid]["age"] = 0
                        if embedding:
                            self._gallery[gid]["embedding"] = embedding
                else:
                    # New (src, track_id) — try to match by embedding
                    gid = self._find_or_create(embedding, src, oid, log)
                    self._track_to_gid[track_key] = gid

                box = obj_meta.rect_params
                label = f"G#{gid} Cam{src}#{oid}"
                text = osd.Text()
                text.display_text = label.encode()
                text.x_offset = int(box.left)
                text.y_offset = max(0, int(box.top) - 50)
                text.font.name = osd.FontFamily.Serif
                text.font.size = 12
                text.font.color = osd.Color(0.2, 1.0, 0.2, 1.0)
                display_meta.add_text(text)

            frame_meta.append(display_meta)

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
        best_gid = -1
        best_score = 0.0

        if embedding:
            for gid, entry in self._gallery.items():
                score = _cosine_similarity(embedding, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_gid = gid

        if best_score >= SIMILARITY_THRESHOLD and best_gid != -1:
            if log:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{best_gid} "
                      f"(similarity={best_score:.3f})")
            self._gallery[best_gid]["age"] = 0
            if embedding:
                self._gallery[best_gid]["embedding"] = embedding
            return best_gid

        # New person
        gid = self._next_gid
        self._next_gid += 1
        self._gallery[gid] = {"embedding": embedding, "age": 0}
        return gid


def compute_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def run(sources_txt: str, nvinfer_config: str, tracker_config: str,
        tile_w: int, tile_h: int):
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
        "src_collector", SourceIdCollectorProbe(id_map, embeddings, person_class_id)))

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": total_w, "height": total_h, "gpu-id": 0,
    })

    # Probe 2 (post-tiler): tiled coords, run gallery matching + draw labels
    gallery_probe = CrossCameraGalleryProbe(
        id_map, embeddings, person_class_id, tile_w, tile_h, cols, n)
    pipeline.attach("tiler", psm.Probe("reid_probe", gallery_probe))

    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})
    pipeline.add(get_sink_element(), "sink", {"sync": 1, "qos": 0})

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[M8] Running. Gallery stats print every 60 frames.")
        print("[M8] Labels show G#<global_id> Cam<src>#<track_id>.")
        print("[M8] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M8] Stopped.")
        total_gids = gallery_probe._next_gid - 1
        print(f"[M8] Total unique global IDs assigned: {total_gids}")


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
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h)
