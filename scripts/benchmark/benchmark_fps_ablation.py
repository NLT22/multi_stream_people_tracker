"""
Benchmark DeepStream throughput by pipeline stage.

This isolates where FPS drops as camera count increases:

  detector_only : decode + mux + nvinfer + fakesink
  tracker_iou   : detector_only + nvtracker IoU
  tracker_perf  : detector_only + NvDCF perf tracker
  tracker_lite  : detector_only + MMP realtime-lite NvDCF tracker
  tracker_recall: detector_only + NvDCF recall/ReID tracker config
  full_main     : current src.main path with tracker + gallery probes + OSD
  full_lite     : src.main tracker-only realtime path, gallery disabled

The script reuses one video N times, runs headless/no-sync, parses the built-in
DeepStream measure_fps_probe output, and writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FPS_RE = re.compile(r"\*\*FPS:\s+([\d.]+)\s+\(([\d.]+)\)")
ERROR_RE = re.compile(
    r"ERROR|Traceback|Segmentation|GPUassert|tracker lib returned error"
    r"|streaming stopped.*error|VPI_ERROR",
    re.IGNORECASE,
)

TRACKER_CONFIGS = {
    "tracker_iou": "configs/tracker/iou.yaml",
    "tracker_perf": "configs/tracker/nvdcf_perf.yaml",
    "tracker_recall": "configs/tracker/nvdcf_accuracy_mmp_recall.yaml",
}


def _scale_sub_batches(template: str, n_cams: int) -> str:
    """Redistribute n_cams evenly across the same number of sub-batches."""
    n_batches = len(template.split(":"))
    base, rem = divmod(n_cams, n_batches)
    return ":".join(str(base + (1 if i < rem else 0)) for i in range(n_batches))


def _probe_cmd(args: argparse.Namespace, variant: str, n_cams: int) -> list[str]:
    sources = [str(Path(args.source).resolve())] * n_cams
    if variant == "full_main":
        return [
            sys.executable, "-m", "src.main",
            "--sources", *sources,
            "--no-display", "--no-sync", "--no-tiler",
            "--nvinfer-config", args.nvinfer_config,
            "--tracker-config", args.full_tracker_config,
            "--gpu-id", str(args.gpu_id),
            "--no-trajectories",
            "--disable-global-merge",
        ]
    if variant == "full_lite":
        sub_batches = (
            _scale_sub_batches(args.tracker_sub_batches, n_cams)
            if args.tracker_sub_batches else None
        )
        cmd = [
            sys.executable, "-m", "src.main",
            "--sources", *sources,
            "--no-display", "--no-sync", "--no-tiler", "--loop-video",
            "--nvinfer-config", args.nvinfer_config,
            "--tracker-config", args.lite_tracker_config,
            "--gpu-id", str(args.gpu_id),
            "--no-trajectories",
            "--disable-gallery",
            "--disable-osd",
        ]
        if sub_batches:
            cmd += ["--tracker-sub-batches", sub_batches]
        return cmd

    scaled_sub_batches = (
        _scale_sub_batches(args.tracker_sub_batches, n_cams)
        if args.tracker_sub_batches else None
    )
    cmd = [
        sys.executable, str(ROOT / "scripts" / "benchmark" / "_run_fps_ablation_variant.py"),
        "--variant", variant,
        "--sources", *sources,
        "--nvinfer-config", args.nvinfer_config,
        "--gpu-id", str(args.gpu_id),
        "--batch-size", str(n_cams),
    ]
    if scaled_sub_batches:
        cmd += ["--tracker-sub-batches", scaled_sub_batches]
    return cmd


def _run_one(args: argparse.Namespace, variant: str, n_cams: int) -> dict:
    cmd = _probe_cmd(args, variant, n_cams)
    print(f"\n[ablation] {variant} cams={n_cams}")

    start = time.monotonic()
    deadline = start + args.warmup + args.duration + 15
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    samples: list[float] = []
    errors: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if ERROR_RE.search(line):
                errors.append(line)
                print("  [!]", line)
            match = FPS_RE.search(line)
            if match:
                elapsed = time.monotonic() - start
                fps = float(match.group(1))
                phase = "warmup" if elapsed < args.warmup else "measure"
                if elapsed >= args.warmup:
                    samples.append(fps)
                print(f"  [{phase}] t={elapsed:.0f}s fps={fps:.1f} per_cam={fps / n_cams:.2f}")
            if time.monotonic() >= deadline:
                proc.terminate()
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not samples:
        return {
            "variant": variant,
            "n_cams": n_cams,
            "fps_total_mean": 0.0,
            "fps_total_min": 0.0,
            "fps_total_max": 0.0,
            "fps_per_cam": 0.0,
            "samples": 0,
            "status": "NO_DATA" if not errors else "ERROR",
        }

    total = sum(samples) / len(samples)
    return {
        "variant": variant,
        "n_cams": n_cams,
        "fps_total_mean": round(total, 2),
        "fps_total_min": round(min(samples), 2),
        "fps_total_max": round(max(samples), 2),
        "fps_per_cam": round(total / n_cams, 2),
        "samples": len(samples),
        "status": "OK",
    }


def _print_table(rows: list[dict], target_fps: float) -> None:
    print("\nvariant,cams,fps/cam,total,pass")
    for row in rows:
        ok = "PASS" if row["fps_per_cam"] >= target_fps else "FAIL"
        if row["status"] != "OK":
            ok = row["status"]
        print(
            f"{row['variant']},{row['n_cams']},"
            f"{row['fps_per_cam']:.2f},{row['fps_total_mean']:.2f},{ok}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cam-counts", nargs="+", type=int, default=[4, 8, 12, 20])
    parser.add_argument("--variants", nargs="+",
                        default=["detector_only", "tracker_iou", "tracker_perf",
                                 "tracker_lite", "tracker_recall",
                                 "full_lite", "full_main"])
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop larger camera counts for a variant once it "
                             "falls below --target-fps.")
    parser.add_argument("--nvinfer-config", default="configs/models/nvinfer_yolov11_mmp.yml")
    parser.add_argument("--full-tracker-config",
                        default="configs/tracker/nvdcf_accuracy_mmp_recall.yaml")
    parser.add_argument("--lite-tracker-config",
                        default="configs/tracker/nvdcf_perf_mmp_lite.yaml")
    parser.add_argument("--tracker-sub-batches", default="5:5:5:5")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", default="output/benchmark/fps_ablation")
    args = parser.parse_args()

    if not Path(args.source).exists():
        raise SystemExit(f"source not found: {args.source}")

    rows: list[dict] = []
    for variant in args.variants:
        for n_cams in args.cam_counts:
            row = _run_one(args, variant, n_cams)
            rows.append(row)
            if args.stop_on_fail and row["fps_per_cam"] < args.target_fps:
                print(
                    f"[ablation] {variant} cams={n_cams} "
                    f"{row['fps_per_cam']:.2f} FPS/cam < {args.target_fps}; "
                    "skipping larger camera counts for this variant."
                )
                break

    _print_table(rows, args.target_fps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fps_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ablation] saved {out}")


if __name__ == "__main__":
    main()
