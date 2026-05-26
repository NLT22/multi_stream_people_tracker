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
                ├── class_id     ← 2 = Person (TrafficCamNet)
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

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2


class MetadataExtractorProbe(psm.BatchMetadataOperator):
    """Full metadata traversal — visual + console output."""

    def __init__(self, save_json: bool = False, output_dir: str = "output/metadata"):
        super().__init__()
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
            dm = psm.DisplayMeta(frame_meta)
            persons = []

            for obj in frame_meta.object_items:
                if obj.class_id != PERSON_CLASS_ID:
                    continue

                persons.append({
                    "object_id":  obj.object_id,
                    "confidence": round(obj.confidence, 4),
                    "left":  round(obj.rect_params.left, 1),
                    "top":   round(obj.rect_params.top, 1),
                    "width": round(obj.rect_params.width, 1),
                    "height":round(obj.rect_params.height, 1),
                })
                self._unique_ids.add(obj.object_id)
                self._total += 1

                # Still draw labels for visual context
                b = obj.rect_params
                dm.add_text(psm.Text(
                    f"#{obj.object_id} {obj.confidence:.0%}",
                    x=int(b.left), y=max(0, int(b.top) - 20),
                    font=psm.Font(psm.FontFamily.Sans, 12),
                    color=psm.Color(0.0, 1.0, 0.3, 1.0),
                ))

            n = len(persons)
            self._max_simultaneous = max(self._max_simultaneous, n)

            if log and persons:
                print(
                    f"[M7] src={frame_meta.source_id:02d} "
                    f"frame={frame_meta.frame_number:06d}  "
                    f"{n} person(s)  IDs={[p['object_id'] for p in persons]}"
                )

            # TODO Exercise 1: uncomment to save JSON
            # if self._save_json and persons:
            #     fname = self._out / f"s{frame_meta.source_id}_f{frame_meta.frame_number:07d}.json"
            #     fname.write_text(json.dumps({
            #         "source_id":    frame_meta.source_id,
            #         "frame_number": frame_meta.frame_number,
            #         "timestamp_ms": int(time.time() * 1000),
            #         "persons":      persons,
            #     }, indent=2))

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
    probe = MetadataExtractorProbe(save_json=save_json)

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
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream-9.0/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
    })

    pipeline.attach("tracker", psm.Probe("meta_probe", probe))
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

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
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config, args.save_json)
