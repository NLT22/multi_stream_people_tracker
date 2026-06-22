#!/usr/bin/env python3
"""Cheap post-run health summary for a long eval/soak (production_todo 3.3).

Reads the stability monitor CSV, the live-buffered CSV, the pipeline log, and the
export dir, and prints one concise health report: warmup-trimmed FPS/cam, VRAM,
RSS trend, latest GID counts, chunk cadence, and error/warning counts.

  python scripts/eval/summarize_long_run.py output/logs output/eval/long_run
  python scripts/eval/summarize_long_run.py output/logs output/eval/long_run \
      --stability long_stability.csv --buffered long_buffered.csv --pipe-log long_pipe.log
"""
from __future__ import annotations
import argparse, csv, glob, os, re, time
from pathlib import Path


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _read_csv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _trend(vals):
    """crude linear creep: (last-quartile mean) - (first-quartile mean)."""
    if len(vals) < 4:
        return 0.0
    q = max(1, len(vals) // 4)
    return sum(vals[-q:]) / q - sum(vals[:q]) / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir", type=Path)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--stability", default="long_stability.csv")
    ap.add_argument("--buffered", default="long_buffered.csv")
    ap.add_argument("--pipe-log", default="long_pipe.log")
    ap.add_argument("--warmup", type=float, default=60.0, help="seconds to trim from the start")
    args = ap.parse_args()

    stab = _read_csv(args.logs_dir / args.stability)
    buf = _read_csv(args.logs_dir / args.buffered)
    plog = args.logs_dir / args.pipe_log

    print(f"== Run summary: {args.logs_dir} / {args.export_dir} ==")

    # --- FPS / VRAM / RSS from stability monitor (warmup-trimmed) ---
    warm = [r for r in stab if r.get("elapsed_s") and float(r["elapsed_s"]) >= args.warmup] or stab
    fps = _floats(warm, "fps")
    vram = _floats(warm, "gpu_mem_mb")
    rss = _floats(warm, "rss_mb")
    util = _floats(warm, "gpu_util")
    dur = float(stab[-1]["elapsed_s"]) if stab else 0.0
    print(f"duration: {dur:.0f}s  stability rows: {len(stab)} (post-warmup {len(warm)})")
    if fps:
        print(f"FPS/cam (post-warmup): avg {sum(fps)/len(fps):.2f}  min {min(fps):.2f}  max {max(fps):.2f}")
    if util:
        print(f"GPU util: avg {sum(util)/len(util):.0f}%  max {max(util):.0f}%")
    if vram:
        print(f"VRAM: avg {sum(vram)/len(vram):.0f} MB  max {max(vram):.0f} MB")
    if rss:
        print(f"RSS: avg {sum(rss)/len(rss):.0f} MB  max {max(rss):.0f} MB  creep {_trend(rss):+.0f} MB")

    # --- GID plateau from buffered log ---
    if buf:
        last = buf[-1]
        tot = _floats(buf, "total_gids")
        print(f"GIDs: latest active {last.get('active_gids','?')} / total {last.get('total_gids','?')}"
              f"  total creep {_trend(tot):+.0f}")
        cms = _floats(buf, "cluster_ms")
        if cms:
            print(f"clustering latency: avg {sum(cms)/len(cms):.0f} ms  max {max(cms):.0f} ms")

    # --- chunk cadence from export dir ---
    chunks = sorted(glob.glob(str(args.export_dir / "det_emb_chunk_*.npz")))
    if chunks:
        last_mtime = max(os.path.getmtime(c) for c in chunks)
        age = time.time() - last_mtime
        print(f"chunks: {len(chunks)}  last written {age:.0f}s ago "
              f"({time.strftime('%H:%M:%S', time.localtime(last_mtime))})")
    else:
        print("chunks: none found")

    # --- error/warning scan of pipe log ---
    if plog.exists():
        txt = plog.read_text(errors="ignore")
        errs = len(re.findall(r"(?i)\b(error|traceback|out of memory|segfault|assert)\b", txt))
        warns = len(re.findall(r"(?i)\bwarn(?:ing)?\b", txt))
        # ignore the benign TensorRT engine-deserialize fallback noise
        benign = len(re.findall(r"(?i)deserialize|rebuild|open error|could not find", txt))
        print(f"pipe log: {errs} error-ish, {warns} warning-ish lines (~{benign} benign engine notices)")
    else:
        print("pipe log: not found")


if __name__ == "__main__":
    main()
