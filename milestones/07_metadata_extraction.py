"""
=============================================================================
MILESTONE 7 — Metadata Extraction
=============================================================================

WHAT YOU LEARN:
  - Full NvDsBatchMeta → NvDsFrameMeta → NvDsObjectMeta traversal
  - CRITICAL: frame_items and object_items are ITERATORS (not lists)
  - All available fields on ObjectMetadata
  - Counting unique tracked persons across an entire video
  - Optionally saving detection data to JSON for offline analysis

METADATA HIERARCHY:
  BatchMetadata               ← one per GPU buffer (one per batch)
    └── FrameMetadata         ← one per source stream in the batch
          ├── frame_number    ← absolute frame index for this source
          ├── source_id       ← which stream (0, 1, 2, ...)
          └── ObjectMetadata  ← one per detected object
                ├── object_id    ← tracker-assigned persistent ID
                ├── class_id     ← 0 = Person (YOLOv8/COCO default)
                ├── label        ← "Person" string from labelfile
                ├── confidence   ← 0.0–1.0
                └── rect_params  ← left, top, width, height (pixels)

ITERATOR RULES:
  frame_meta.object_items  →  ITERATOR — no len(), no second pass
  batch_meta.frame_items   →  ITERATOR — same rules
  Convert to list if you need multiple passes:
    objects = list(frame_meta.object_items)

RUN:
  python milestones/07_metadata_extraction.py
  python milestones/07_metadata_extraction.py --save-json

TODO EXERCISES:
  1. Uncomment --save-json to write per-frame JSON files.
  2. Track max simultaneous persons across all cameras.
  3. Build a trajectory: {object_id: [(frame, x, y), ...]} for each person.
  4. Calculate which camera has the most unique persons over the full video.
=============================================================================
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pyservicemaker as psm
from pyservicemaker import osd

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
    infer_source_id_from_tiled_box,
)
from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tracker' output (pre-tiler) where source_id is still valid.
    Fills two shared dicts:
      frame_src:    (source_id, object_id) → frame_number  for JSON/logging

    After nvmultistreamtiler source_id resets to 0 for every frame.
    """

    def __init__(self, id_map: dict, frame_src: dict, person_class_id: int):
        super().__init__()
        self._id_map = id_map
        self._frame_src = frame_src
        self._person_class_id = person_class_id

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            fn = frame_meta.frame_number
            for obj in frame_meta.object_items:
                if obj.class_id == self._person_class_id:
                    self._id_map[obj.object_id] = src
                    self._frame_src[(src, obj.object_id)] = fn


class MetadataExtractorProbe(psm.BatchMetadataOperator):
    """
    Runs on 'tiler' output (post-tiler) — tiled canvas coordinates.
    Draws labels and logs per-source stats using source id inferred from
    each tiled bbox position.
    """

    def __init__(self, id_map: dict, frame_src: dict,
                 person_class_id: int,
                 tile_w: int, tile_h: int, cols: int, num_sources: int,
                 save_json: bool = False, output_dir: str = "output/metadata"):
        super().__init__()
        self._id_map = id_map
        self._frame_src = frame_src
        self._person_class_id = person_class_id
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cols = cols
        self._num_sources = num_sources
        self._save_json = save_json
        self._out = Path(output_dir)
        self._total = 0
        self._unique_ids: set[int] = set()
        self._max_simultaneous = 0
        self._frames = 0
        self._t0 = time.time()
        if save_json:
            self._out.mkdir(parents=True, exist_ok=True)
            print(f"[M7] Saving JSON to {self._out}/")

    def handle_metadata(self, batch_meta):
        self._frames += 1
        log = self._frames % 60 == 0

        for frame_meta in batch_meta.frame_items:
            dm = batch_meta.acquire_display_meta()
            persons = []

            for obj in frame_meta.object_items:
                if obj.class_id != self._person_class_id:
                    continue

                src = infer_source_id_from_tiled_box(
                    obj.rect_params, self._tile_w, self._tile_h,
                    self._cols, self._num_sources)
                frame_number = self._frame_src.get(
                    (src, obj.object_id), frame_meta.frame_number)
                persons.append({
                    "object_id":  obj.object_id,
                    "source_id":  src,
                    "frame_number": frame_number,
                    "confidence": round(obj.confidence, 4),
                    "left":   round(obj.rect_params.left, 1),
                    "top":    round(obj.rect_params.top, 1),
                    "width":  round(obj.rect_params.width, 1),
                    "height": round(obj.rect_params.height, 1),
                })
                self._unique_ids.add(obj.object_id)
                self._total += 1

                b = obj.rect_params
                t = osd.Text()
                t.display_text = f"Cam{src} #{obj.object_id} {obj.confidence:.0%}".encode()
                t.x_offset = int(b.left)
                t.y_offset = max(0, int(b.top) - 50)
                t.font.name = osd.FontFamily.Serif
                t.font.size = 12
                t.font.color = osd.Color(0.0, 1.0, 0.3, 1.0)
                dm.add_text(t)

            frame_meta.append(dm)

            n = len(persons)
            self._max_simultaneous = max(self._max_simultaneous, n)

            if log and persons:
                # Group by source for readable output
                by_src: dict[int, list] = {}
                for p in persons:
                    by_src.setdefault(p["source_id"], []).append(p["object_id"])
                for src, ids in sorted(by_src.items()):
                    print(
                        f"[M7] Cam{src:02d} "
                        f"frame={frame_meta.frame_number:06d}  "
                        f"{len(ids)} person(s)  IDs={ids}"
                    )

            if self._save_json and persons:
                # One JSON file per (source, frame) — group persons by source
                by_src_json: dict[int, list] = {}
                for p in persons:
                    by_src_json.setdefault(p["source_id"], []).append(p)
                for src, src_persons in by_src_json.items():
                    frame_number = src_persons[0].get(
                        "frame_number", frame_meta.frame_number)
                    fname = self._out / f"s{src:02d}_f{frame_number:07d}.json"
                    fname.write_text(json.dumps({
                        "source_id":    src,
                        "frame_number": frame_number,
                        "timestamp_ms": int(time.time() * 1000),
                        "persons":      src_persons,
                    }, indent=2))

    def report(self):
        elapsed = time.time() - self._t0
        print("\n" + "=" * 55)
        print("METADATA SUMMARY")
        print("=" * 55)
        print(f"  Runtime             : {elapsed:.1f}s")
        print(f"  Frames processed    : {self._frames}")
        print(f"  Total detections    : {self._total}")
        print(f"  Unique tracking IDs : {len(self._unique_ids)}")
        print(f"  Max simultaneous    : {self._max_simultaneous}")
        print("=" * 55)


def compute_grid(n):
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def run(sources_txt: str, nvinfer_config: str, tracker_config: str, save_json: bool):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(uris)
    rows, cols = compute_grid(n)
    person_class_id = infer_person_class_id(nvinfer_config)
    print(f"[M7] person_class_id={person_class_id} inferred from {nvinfer_config}")

    # Shared dicts between the two probes
    id_map: dict[int, int] = {}
    frame_src: dict[tuple, int] = {}     # (source_id, object_id) → frame_number
    probe = MetadataExtractorProbe(
        id_map, frame_src, person_class_id, 1280, 720, cols, n,
        save_json=save_json)

    pipeline = psm.Pipeline("m7-metadata")

    pipeline.add("nvstreammux", "mux", {
        "batch-size": n, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": 0,
    })
    for i, uri in enumerate(uris):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": uri, "gpu-id": 0})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config, "batch-size": n, "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
    })

    # Probe 1 on tracker (pre-tiler): source_id still valid here
    pipeline.attach("tracker", psm.Probe(
        "src_collector", SourceIdCollectorProbe(id_map, frame_src, person_class_id)))

    # TILER first — composites streams, scales metadata to tile coords
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })

    # Probe 2 on tiler (post-tiler): tiled canvas coordinates for drawing
    pipeline.attach("tiler", psm.Probe("meta_probe", probe))
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    pipeline.add(get_sink_element(), "sink", {"sync": 1, "qos": 0})

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "tiler")
    pipeline.link("tiler", "osd")
    pipeline.link("osd", "sink")

    try:
        pipeline.start()
        print("[M7] Running. Metadata logged every 60 frames. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M7] Stopped.")
    finally:
        probe.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 7: Metadata extraction")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_yolov8_people.yml",
                        help="nvinfer config. Default: YOLOv8. "
                             "Alternatives: configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config, args.save_json)
