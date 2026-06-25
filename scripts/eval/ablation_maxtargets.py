"""
maxTargetsPerStream ablation — FPS + VRAM on 20 cameras.

For each maxTargetsPerStream value, run the production reid0 pipeline on the
20-cam mixed source list headless, let it reach steady state, then measure:
  - per-stream FPS  (mean of the instantaneous **FPS values over the window)
  - aggregate FPS   (sum over 20 streams)
  - peak VRAM       (nvidia-smi sampled during the measure window)

No GT scoring — this is a throughput/VRAM probe only. The pipeline is launched
in the background and killed after the measure window (videos are 10 min long).

Usage:
    python scripts/eval/ablation_maxtargets.py \
        [--values 10,20,40,220] [--warmup 35] [--measure 60]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_TRACKER = REPO / "configs/tracker/nvdcf_accuracy_mmp_recall_sgie_reid0.yaml"
PIPELINE = REPO / "configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml"
SOURCES = REPO / "configs/sources/val_20cam_mixed.txt"
PYTHON = REPO / "venv/bin/python3"

FPS_RE = re.compile(r"([\d.]+)\s*\(([\d.]+)\)")


def make_config(value: int, out_dir: Path) -> Path:
    """Write a copy of the base tracker config with maxTargetsPerStream=value."""
    text = BASE_TRACKER.read_text()
    new = re.sub(r"(maxTargetsPerStream:\s*)\d+", rf"\g<1>{value}", text)
    if "maxTargetsPerStream" not in new:
        raise RuntimeError("maxTargetsPerStream not found in base config")
    out = out_dir / f"tracker_maxtgt_{value}.yaml"
    out.write_text(new)
    return out


def sample_vram(gpu_id: int = 0) -> int:
    """Return used VRAM in MiB."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu_id)],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def parse_fps(log_path: Path, last_n: int = 10) -> tuple[float, float, int]:
    """Return (mean per-stream instantaneous FPS, aggregate FPS, n_streams)
    from the last `last_n` **FPS lines."""
    lines = [ln for ln in log_path.read_text(errors="ignore").splitlines()
             if ln.startswith("**FPS:")]
    if not lines:
        return 0.0, 0.0, 0
    sample = lines[-last_n:]
    per_stream_means: list[float] = []
    n_streams = 0
    for ln in sample:
        inst = [float(m.group(1)) for m in FPS_RE.finditer(ln)]
        if not inst:
            continue
        n_streams = max(n_streams, len(inst))
        per_stream_means.append(sum(inst) / len(inst))
    if not per_stream_means:
        return 0.0, 0.0, 0
    mean_per_stream = sum(per_stream_means) / len(per_stream_means)
    return mean_per_stream, mean_per_stream * n_streams, n_streams


def run_one(value: int, work_dir: Path, warmup: float, measure: float) -> dict:
    cfg = make_config(value, work_dir)
    log_path = work_dir / f"run_maxtgt_{value}.log"
    log_f = open(log_path, "w")

    cmd = [
        str(PYTHON), "-m", "src.main",
        "--config", str(PIPELINE),
        "--sources", str(SOURCES),
        "--tracker-config", str(cfg),
        "--no-display", "--no-sync",
    ]
    print(f"\n{'='*60}\n maxTargetsPerStream = {value}\n{'='*60}")
    print(f"  launching pipeline (log: {log_path.name}) ...")
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log_f,
                            stderr=subprocess.STDOUT)

    vram_samples: list[int] = []
    try:
        # Warmup: wait for engine load + steady state
        t_end_warm = time.time() + warmup
        while time.time() < t_end_warm:
            if proc.poll() is not None:
                print(f"  ERROR: pipeline exited early (code {proc.returncode})")
                return {"value": value, "fps_per_stream": 0.0, "fps_agg": 0.0,
                        "vram_mib": -1, "n_streams": 0, "error": "early-exit"}
            time.sleep(2)
        print(f"  warmup done ({warmup:.0f}s) — measuring {measure:.0f}s ...")

        # Measure window: sample VRAM
        t_end_meas = time.time() + measure
        while time.time() < t_end_meas:
            v = sample_vram()
            if v > 0:
                vram_samples.append(v)
            time.sleep(3)
    finally:
        print("  stopping pipeline ...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_f.close()
        # cooldown so VRAM frees before next run
        time.sleep(8)

    mean_ps, agg, n = parse_fps(log_path)
    peak_vram = max(vram_samples) if vram_samples else -1
    result = {
        "value": value,
        "fps_per_stream": round(mean_ps, 2),
        "fps_agg": round(agg, 1),
        "vram_mib": peak_vram,
        "vram_gb": round(peak_vram / 1024, 2) if peak_vram > 0 else -1,
        "n_streams": n,
    }
    print(f"  → per-stream FPS={result['fps_per_stream']}  "
          f"agg={result['fps_agg']}  VRAM={result['vram_gb']} GB  "
          f"(streams={n}, vram_samples={len(vram_samples)})")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", default="10,20,40,220")
    ap.add_argument("--warmup", type=float, default=35)
    ap.add_argument("--measure", type=float, default=60)
    ap.add_argument("--work-dir",
                    default="output/eval/ablation_maxtargets")
    args = ap.parse_args()

    values = [int(v) for v in args.values.split(",")]
    work_dir = (REPO / args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results = [run_one(v, work_dir, args.warmup, args.measure) for v in values]

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" maxTargetsPerStream ablation — 20 cam, reid0 preset")
    print("=" * 60)
    print(f"{'maxTgt':>7} {'FPS/cam':>8} {'agg FPS':>9} {'VRAM(GB)':>9}")
    print("─" * 40)
    for r in results:
        print(f"{r['value']:>7} {r['fps_per_stream']:>8.2f} "
              f"{r['fps_agg']:>9.1f} {r['vram_gb']:>9.2f}")

    import json
    (work_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {work_dir}/results.json")


if __name__ == "__main__":
    main()
