#!/usr/bin/env python3
"""Write a run_manifest.json for a production run (production_todo 4.3).

Captures the provenance needed to reproduce/diagnose a run: git commit, pipeline
config, source list, env map, duration, GPU name, resolved model files, and key
thresholds. Called by run_long_eval.sh; can also be run standalone.

  python scripts/eval/write_run_manifest.py --config <pipeline.yaml> \
      --sources <srcs.txt> --env-map <spec> --duration 600 --out <dir>/run_manifest.json
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _resolve(base: Path, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def _git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        return None


def _onnx(nvinfer_rel: str):
    cfg = _resolve(ROOT, nvinfer_rel)
    if not cfg.exists():
        return None
    m = re.search(r"^\s*onnx-file:\s*(\S+)", cfg.read_text(), re.M)
    return str(_resolve(cfg.parent, m.group(1))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--env-map", default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--extra", default=None, help="JSON string of extra run params")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(_resolve(ROOT, args.config).read_text()) or {}
    det = cfg.get("detection", {}) or {}
    reid = cfg.get("reid", {}) or {}

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "gpu": _gpu_name(),
        "pipeline_config": args.config,
        "sources": args.sources,
        "env_map": args.env_map,
        "duration_s": args.duration,
        "models": {
            "detector_onnx": _onnx(det["config_file"]) if det.get("config_file") else None,
            "reid_sgie_onnx": _onnx(det["reid_sgie_config"]) if det.get("reid_sgie_config") else None,
        },
        "tracker_config": (cfg.get("tracker") or {}).get("config_file"),
        "key_thresholds": {
            "reid_similarity_threshold": reid.get("similarity_threshold"),
            "global_merge_threshold": reid.get("global_merge_threshold"),
            "global_merge_margin": reid.get("global_merge_margin"),
        },
    }
    if args.extra:
        try:
            manifest["run_params"] = json.loads(args.extra)
        except json.JSONDecodeError:
            manifest["run_params_raw"] = args.extra

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
