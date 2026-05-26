"""
=============================================================================
MILESTONE 8 — Metadata Extraction
=============================================================================

WHAT YOU LEARN:
  - Full NvDsBatchMeta → NvDsFrameMeta → NvDsObjectMeta traversal
  - Why iterators (not lists): NEVER use len(), iterate to count
  - All available metadata fields on ObjectMetadata
  - How to aggregate stats across streams and frames
  - How to save metadata to JSON/CSV for offline analysis
  - How to count unique tracked people across a video

METADATA HIERARCHY:
  BatchMetadata (one per buffer)
    └── FrameMetadata (one per source stream in the batch)
          ├── frame_number   : absolute frame counter for this source
          ├── source_id      : which stream (0, 1, 2, ...)
          └── ObjectMetadata (one per detected object in this frame)
                ├── object_id    : tracker-assigned persistent ID
                ├── class_id     : model class (2=person for TrafficCamNet)
                ├── label        : string label from labelfile
                ├── confidence   : detection confidence 0.0–1.0
                └── rect_params  : NvOSD_RectParams (left, top, width, height)

ITERATOR RULES (critical):
  frame_meta.object_items → ITERATOR, not list → NO len(), NO second pass
  If you need to iterate twice:
    objects = list(frame_meta.object_items)  # ← allocates, OK once per frame

RUN:
  source venv/bin/activate
  python milestones/08_metadata_extraction.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Uncomment the JSON output section to save metadata to files.
  2. Track the maximum number of simultaneous persons seen.
  3. Build a dict mapping object_id → list of bounding boxes across frames
     to reconstruct each person's trajectory.
  4. Calculate average confidence across all person detections.
=============================================================================
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


PERSON_CLASS_ID = 2


class MetadataExtractorProbe(psm.BatchMetadataOperator):
    """
    Comprehensive metadata extraction probe.

    Demonstrates every useful field available on ObjectMetadata.
    Optionally saves per-frame metadata to JSON files.
    """

    def __init__(self, save_json: bool = False, output_dir: str = "output/metadata"):
        super().__init__()
        self._save_json = save_json
        self._output_dir = Path(output_dir)
        self._total_persons_seen = 0
        self._unique_ids: set[int] = set()
        self._max_simultaneous = 0
        self._frame_count = 0
        self._start_time = time.time()

        if save_json:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[M8] Saving metadata JSON to: {self._output_dir}")

    def execute(self, batch_meta):
        self._frame_count += 1

        # ── Outer loop: one iteration per source stream in the batch ─────────
        # batch_meta.frame_items is an ITERATOR — do NOT call len() on it
        for frame_meta in batch_meta.frame_items:

            frame_persons = []

            # ── Inner loop: one iteration per detected object in this frame ──
            # frame_meta.object_items is ALSO an ITERATOR
            for obj_meta in frame_meta.object_items:

                if obj_meta.class_id != PERSON_CLASS_ID:
                    continue

                # ── All available ObjectMetadata fields ───────────────────
                person_data = {
                    # Tracker-assigned persistent ID across frames
                    # (0 = not tracked, set only if nvtracker is in pipeline)
                    "object_id": obj_meta.object_id,

                    # Model class integer (2 for TrafficCamNet person)
                    "class_id": obj_meta.class_id,

                    # String label from labelfile-path (e.g. "Person")
                    "label": obj_meta.label,

                    # Detection confidence [0.0, 1.0]
                    "confidence": round(obj_meta.confidence, 4),

                    # Bounding box in pixels (relative to the stream resolution)
                    "left":   round(obj_meta.rect_params.left, 1),
                    "top":    round(obj_meta.rect_params.top, 1),
                    "width":  round(obj_meta.rect_params.width, 1),
                    "height": round(obj_meta.rect_params.height, 1),
                }

                frame_persons.append(person_data)
                self._total_persons_seen += 1
                self._unique_ids.add(obj_meta.object_id)

            # ── Per-frame stats ────────────────────────────────────────────
            n = len(frame_persons)
            self._max_simultaneous = max(self._max_simultaneous, n)

            # Print summary every 30 frames
            if self._frame_count % 30 == 0 and frame_persons:
                print(
                    f"[M8] src={frame_meta.source_id}  "
                    f"frame={frame_meta.frame_number}  "
                    f"persons={n}  "
                    f"IDs={[p['object_id'] for p in frame_persons]}"
                )

            # TODO Exercise 1: save to JSON
            # if self._save_json and frame_persons:
            #     fname = self._output_dir / f"src{frame_meta.source_id}_f{frame_meta.frame_number:07d}.json"
            #     payload = {
            #         "source_id":    frame_meta.source_id,
            #         "frame_number": frame_meta.frame_number,
            #         "timestamp_ms": int(time.time() * 1000),
            #         "persons":      frame_persons,
            #     }
            #     fname.write_text(json.dumps(payload, indent=2))

    def report(self):
        elapsed = time.time() - self._start_time
        print("\n" + "=" * 60)
        print("METADATA EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"  Runtime             : {elapsed:.1f}s")
        print(f"  Total frames        : {self._frame_count}")
        print(f"  Total persons seen  : {self._total_persons_seen}")
        print(f"  Unique tracking IDs : {len(self._unique_ids)}")
        print(f"  Max simultaneous    : {self._max_simultaneous}")
        if elapsed > 0:
            print(f"  Avg persons/frame   : {self._total_persons_seen / max(1, self._frame_count):.2f}")
        print("=" * 60)


def run(sources_txt: str, nvinfer_config: str, tracker_config: str,
        save_json: bool):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    rows, cols = math.ceil(math.sqrt(num_sources)), None
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)

    extractor_probe = MetadataExtractorProbe(save_json=save_json)

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m8-metadata")

    pipeline.add("nvstreammux", "mux", {
        "batch-size": num_sources, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": 0,
    })

    for i, uri in enumerate(uris):
        src_name = f"source_{i}"
        pipeline.add("nvurisrcbin", src_name, {"uri": uri, "gpu-id": 0})
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": num_sources, "gpu-id": 0,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe",
                    {"print-fps-interval": 5})

    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": (
            "/opt/nvidia/deepstream/deepstream/lib/"
            "libnvds_nvmultiobjecttracker.so"
        ),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384, "gpu-id": 0,
    })

    # METADATA EXTRACTOR PROBE
    pipeline.attach("tracker", extractor_probe, "meta_probe", {})

    # OSD (optional visualization while extracting)
    pipeline.add("nvosdbin", "osd", {"gpu-id": 0, "process-mode": 1})

    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows, "gpu-id": 0,
    })
    pipeline.add(get_sink_element(), "sink", {"sync": 0, "qos": 0})

    # ── Link ─────────────────────────────────────────────────────────────────
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    pipeline.link("tracker", "osd")
    pipeline.link("osd", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M8] Pipeline running. Metadata printed every 30 frames per stream.")
        print("[M8] Press Ctrl+C to stop and see summary.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M8] Stopped by user.")
    finally:
        extractor_probe.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 8: Metadata extraction")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument("--nvinfer-config",
                        default="configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config",
                        default="configs/tracker/nvdcf_perf.yaml")
    parser.add_argument("--save-json", action="store_true",
                        help="Save per-frame metadata to output/metadata/*.json")
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config, args.tracker_config, args.save_json)
