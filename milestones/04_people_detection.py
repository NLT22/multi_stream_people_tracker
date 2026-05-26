"""
=============================================================================
MILESTONE 4 — People Detection with nvinfer
=============================================================================

WHAT YOU LEARN:
  - How nvinfer (TensorRT inference engine) fits into the pipeline
  - nvinfer config file format (YAML, "property:" section)
  - FP16 precision: why it matters for 4GB VRAM
  - First-run engine build vs cached engine
  - How to attach the built-in FPS probe (and why NOT to attach it to a sink)
  - TrafficCamNet class IDs (class_id=2 is Person)

PIPELINE TOPOLOGY:
  [sources] → [nvstreammux] → [nvinfer/pgie] → [nvmultistreamtiler] → [sink]

  nvinfer reads each batch from the muxer, runs TensorRT inference,
  and writes NvDsObjectMeta (bounding boxes, class IDs, confidence scores)
  into the NvDsBatchMeta for each detected object.

  The sink just displays video — no bounding boxes yet (that's Milestone 6).
  But the detection metadata IS there; you can verify with a probe.

FIRST RUN — ENGINE BUILD:
  If the .engine file in the config doesn't exist yet, nvinfer builds it
  from the ONNX model. This takes 1-3 minutes on an RTX 3050Ti.
  Watch for: "[NvDsInferContextImpl] Building network engine..."
  After first run, the .engine file is cached and loading takes < 5 seconds.

FPS PROBE RULE (CRITICAL):
  pipeline.attach("pgie", "measure_fps_probe", ...)  ← CORRECT
  pipeline.attach("sink", "measure_fps_probe", ...)  ← RAISES RuntimeError!
  The built-in measure_fps_probe can ONLY attach to processing elements.

RUN:
  source venv/bin/activate
  python milestones/04_people_detection.py --sources configs/sources/video_files.txt

TODO EXERCISES:
  1. Change network-mode in nvinfer_trafficcamnet.yml to 0 (FP32).
     Compare memory usage (nvidia-smi) and FPS. Then switch back to 2 (FP16).
  2. Change interval from 0 to 2 in the config — nvinfer runs every 3rd frame.
     Observe FPS increase. Does it look smooth with the tracker (Milestone 5)?
  3. Lower pre-cluster-threshold to 0.1 — more detections but more noise.
  4. Add a probe to print class_id for every detected object this frame.
=============================================================================
"""

import argparse
import math
import sys

import pyservicemaker as psm

from src.pipeline.sources import load_uris_from_txt
from src.utils.platform_utils import get_sink_element


class DetectionVerifierProbe(psm.BatchMetadataOperator):
    """
    Learning probe: verify that nvinfer is producing detections.
    Prints a summary every N frames so you can confirm the model is working.
    """

    PERSON_CLASS_ID = 2   # TrafficCamNet: 0=vehicle, 1=bicycle, 2=person, 3=sign

    def __init__(self, print_interval: int = 30):
        super().__init__()
        self._print_interval = print_interval
        self._frame_count = 0

    def execute(self, batch_meta):
        self._frame_count += 1
        if self._frame_count % self._print_interval != 0:
            return

        for frame_meta in batch_meta.frame_items:  # ITERATOR
            class_counts: dict[int, int] = {}
            for obj_meta in frame_meta.object_items:  # ITERATOR
                class_counts[obj_meta.class_id] = class_counts.get(obj_meta.class_id, 0) + 1

            persons = class_counts.get(self.PERSON_CLASS_ID, 0)
            print(
                f"[M4] src={frame_meta.source_id}  "
                f"frame={frame_meta.frame_number}  "
                f"all_detections={class_counts}  "
                f"persons={persons}"
            )


def run(sources_txt: str, nvinfer_config: str):
    try:
        uris = load_uris_from_txt(sources_txt)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    num_sources = len(uris)
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)

    print(f"[M4] Sources       : {num_sources}")
    print(f"[M4] nvinfer config: {nvinfer_config}")
    print("[M4] NOTE: First run may take 1-3 min to build TensorRT engine.")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = psm.Pipeline("m4-people-detection")

    # MUXER
    pipeline.add("nvstreammux", "mux", {
        "batch-size": num_sources,
        "batched-push-timeout": 40000,
        "width":  1920,
        "height": 1080,
        "gpu-id": 0,
    })

    # SOURCES
    for i, uri in enumerate(uris):
        src_name = f"source_{i}"
        pipeline.add("nvurisrcbin", src_name, {"uri": uri, "gpu-id": 0})
        pipeline.link((src_name, "mux"), ("", "sink_%u"))

    # PRIMARY INFERENCE ENGINE (nvinfer)
    # config-file-path: YAML file with model, precision, and class settings
    # batch-size: must match muxer batch-size
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": nvinfer_config,
        "batch-size": num_sources,
        "gpu-id": 0,
    })

    # FPS PROBE — MUST attach to "pgie", NOT to "sink"
    # Attaching to a sink raises: RuntimeError: Probe failure
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe",
                    {"print-fps-interval": 5})

    # DETECTION VERIFIER PROBE — our custom probe to confirm detections exist
    pipeline.attach("pgie", DetectionVerifierProbe(), "det_probe", {})

    # TILER
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": 1280 * cols, "height": 720 * rows,
        "gpu-id": 0,
    })

    # SINK — note: no bounding boxes yet! That's Milestone 6 (nvosdbin).
    # The detections ARE in metadata, just not rendered visually yet.
    pipeline.add(get_sink_element(), "sink", {"sync": 1})

    # ── Link ─────────────────────────────────────────────────────────────────
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tiler")
    pipeline.link("tiler", "sink")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        pipeline.start()
        print("[M4] Pipeline running.")
        print("[M4] Detection counts print every 30 frames per stream.")
        print("[M4] FPS prints every 5 seconds.")
        print("[M4] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[M4] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 4: People detection")
    parser.add_argument("--sources", default="configs/sources/video_files.txt")
    parser.add_argument(
        "--nvinfer-config",
        default="configs/models/nvinfer_trafficcamnet.yml",
        help="Path to nvinfer config YAML",
    )
    args = parser.parse_args()
    run(args.sources, args.nvinfer_config)
