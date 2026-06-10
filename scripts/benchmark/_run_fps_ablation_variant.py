"""Internal helper for scripts/benchmark/benchmark_fps_ablation.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyservicemaker as psm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.engine_prep import prepare_nvinfer_config
from src.pipeline.model_utils import deepstream_tracker_lib_path


TRACKER_CONFIGS = {
    "tracker_iou": "configs/tracker/iou.yaml",
    "tracker_perf": "configs/tracker/nvdcf_perf.yaml",
    "tracker_lite": "configs/tracker/nvdcf_perf_mmp_lite.yaml",
    "tracker_recall": "configs/tracker/nvdcf_accuracy_mmp_recall.yaml",
}


def _uri(path: str) -> str:
    if path.startswith(("file://", "rtsp://", "http://", "https://")):
        return path
    return "file://" + str(Path(path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        choices=["detector_only", *TRACKER_CONFIGS.keys()])
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--nvinfer-config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--tracker-sub-batches", default=None)
    args = parser.parse_args()

    n = len(args.sources)
    batch = max(args.batch_size, n)
    runtime_cfg = prepare_nvinfer_config(
        args.nvinfer_config, batch, args.gpu_id, force_rebuild=False)

    pipeline = psm.Pipeline("fps-ablation")
    pipeline.add("nvstreammux", "mux", {
        "batch-size": batch,
        "batched-push-timeout": 40000,
        "width": 1920,
        "height": 1080,
        "gpu-id": args.gpu_id,
    })
    for i, source in enumerate(args.sources):
        name = f"source_{i}"
        pipeline.add("nvurisrcbin", name, {"uri": _uri(source), "gpu-id": args.gpu_id})
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": runtime_cfg,
        "batch-size": batch,
        "gpu-id": args.gpu_id,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")
    pipeline.link("mux", "pgie")
    last = "pgie"

    if args.variant in TRACKER_CONFIGS:
        tracker_props = {
            "ll-lib-file": deepstream_tracker_lib_path(),
            "ll-config-file": TRACKER_CONFIGS[args.variant],
            "tracker-width": 640,
            "tracker-height": 384,
            "gpu-id": args.gpu_id,
        }
        if args.tracker_sub_batches:
            tracker_props["sub-batches"] = args.tracker_sub_batches
        pipeline.add("nvtracker", "tracker", tracker_props)
        pipeline.link("pgie", "tracker")
        last = "tracker"

    pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
    pipeline.link(last, "sink")

    try:
        pipeline.start()
        pipeline.wait()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    sys.exit(main())
