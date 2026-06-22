#!/usr/bin/env python3
"""Production config guardrail (production_todo 4.1).

Validates a pipeline preset before a run: required files exist, model ONNX paths
resolve, the SGIE ReID config is present, the tracker has outputReidTensor:0 when
SGIE drives global IDs, and (optionally) the source count matches the env map.

  python scripts/setup/validate_config.py \
    --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
    --sources configs/sources/val_20cam_mixed.txt \
    --env-map "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
checks = []  # (ok, msg)


def chk(ok, msg):
    checks.append((bool(ok), msg))


def resolve(base: Path, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def onnx_in(nvinfer_cfg: Path):
    if not nvinfer_cfg.exists():
        return None
    m = re.search(r"^\s*onnx-file:\s*(\S+)", nvinfer_cfg.read_text(), re.M)
    return resolve(nvinfer_cfg.parent, m.group(1)) if m else None


def env_span(spec: str) -> int:
    mx = -1
    for part in spec.split(","):
        rng = part.split(":")[1]
        for seg in rng.split("+"):
            mx = max(mx, *(int(x) for x in seg.split("-")))
    return mx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--sources", type=Path)
    ap.add_argument("--env-map")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    chk(cfg_path.exists(), f"pipeline config exists: {args.config}")
    if not cfg_path.exists():
        return _finish()
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    det = cfg.get("detection", {}) or {}
    trk = cfg.get("tracker", {}) or {}

    # detector
    det_cfg = resolve(ROOT, det.get("config_file", "")) if det.get("config_file") else None
    chk(det_cfg and det_cfg.exists(), f"detector nvinfer config exists: {det.get('config_file')}")
    if det_cfg and det_cfg.exists():
        onnx = onnx_in(det_cfg)
        chk(onnx and onnx.exists(), f"detector ONNX resolves: {onnx}")

    # SGIE ReID (required for the production SGIE presets)
    sgie = det.get("reid_sgie_config")
    chk(bool(sgie), "SGIE reid config present (detection.reid_sgie_config)")
    if sgie:
        sgie_cfg = resolve(ROOT, sgie)
        chk(sgie_cfg.exists(), f"SGIE nvinfer config exists: {sgie}")
        onnx = onnx_in(sgie_cfg)
        chk(onnx and onnx.exists(), f"SGIE ReID ONNX resolves: {onnx}")

    # tracker + outputReidTensor:0 when SGIE used
    trk_cfg = resolve(ROOT, trk.get("config_file", "")) if trk.get("config_file") else None
    chk(trk_cfg and trk_cfg.exists(), f"tracker config exists: {trk.get('config_file')}")
    if trk_cfg and trk_cfg.exists() and sgie:
        txt = trk_cfg.read_text()
        m = re.search(r"^\s*outputReidTensor:\s*(\d+)", txt, re.M)
        chk(m and m.group(1) == "0",
            f"tracker outputReidTensor:0 (SGIE drives export) — found {m.group(1) if m else 'unset'}")

    # source count vs env map
    if args.sources and args.env_map:
        srcs = args.sources if args.sources.is_absolute() else (ROOT / args.sources)
        if srcs.exists():
            n = sum(1 for ln in srcs.read_text().splitlines()
                    if ln.strip() and not ln.strip().startswith("#"))
            span = env_span(args.env_map)
            chk(n == span, f"source count ({n}) matches env-map span ({span})")
        else:
            chk(False, f"sources file exists: {args.sources}")

    return _finish()


def _finish():
    ok_all = all(ok for ok, _ in checks)
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"\n{'ALL CHECKS PASSED' if ok_all else 'VALIDATION FAILED'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
