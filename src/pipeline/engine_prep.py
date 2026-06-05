"""
Dynamic nvinfer config + TensorRT engine preparation.

These helpers let the pipeline run with ANY number of streams. The detector
nvinfer engine is batch-coupled: a batch-4 engine cannot serve a batch-8 run.
Instead of hand-editing configs and deleting engines (as the config comments
tell you to), we generate a runtime config beside the original that points at
the per-batch engine name DeepStream itself uses, and clean stale engines.

NOTE: the ReID tracker engine (resnet50_market1501.etlt_b16_...) is NOT touched
here — its batchSize (crops per inference) is independent of stream count, so
the same engine is valid for any N.
"""

import os
from pathlib import Path

import yaml


def precision_token(network_mode) -> str:
    """Map nvinfer network-mode to DeepStream's engine-name precision token."""
    return {0: "fp32", 1: "int8", 2: "fp16"}.get(int(network_mode), "fp16")


def engine_name(onnx_path: Path, batch: int, gpu_id: int, prec: str) -> str:
    """Deterministic engine filename DeepStream produces for a given batch.

    Matches the names already on disk, e.g. yolov8n.onnx_b4_gpu0_fp16.engine,
    so a cached engine for this batch is reused instead of rebuilt.
    """
    return f"{onnx_path.name}_b{batch}_gpu{gpu_id}_{prec}.engine"


def prepare_nvinfer_config(config_path: str, batch: int, gpu_id: int = 0,
                           force_rebuild: bool = False) -> str:
    """Write a runtime nvinfer config for `batch` and manage its engine cache.

    Returns the path to a generated sibling config (same directory as the
    original so its relative paths still resolve) with batch-size and
    model-engine-file set for this run. Stale engines (older than the ONNX) are
    deleted so DeepStream rebuilds them; engines for other batches are kept.
    """
    cfg_path = Path(config_path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    prop = raw.get("property", {})

    onnx_rel = prop.get("onnx-file")
    if not onnx_rel:
        # No ONNX (e.g. pre-built-engine-only config): just patch batch-size.
        prop["batch-size"] = batch
        raw["property"] = prop
        return _write_runtime_config(cfg_path, raw, batch, gpu_id)

    # nvinfer paths are relative to the CONFIG FILE's directory.
    # Use the logical name from the config (preserving symlink names) for engine
    # naming so that different aliases of the same ONNX get separate engine caches.
    onnx_logical_name = Path(onnx_rel).name
    onnx_path = (cfg_path.parent / onnx_rel).resolve()
    prec = precision_token(prop.get("network-mode", 2))
    eng_file = f"{onnx_logical_name}_b{batch}_gpu{gpu_id}_{prec}.engine"
    eng_path = onnx_path.parent / eng_file

    _clean_stale_engines(onnx_path, onnx_logical_name, eng_path, force_rebuild)

    # Point model-engine-file at the per-batch engine, expressed relative to the
    # config dir so the generated sibling config keeps working paths.
    try:
        eng_rel = eng_path.relative_to(cfg_path.parent.resolve())
    except ValueError:
        eng_rel = Path(os.path.relpath(eng_path, cfg_path.parent.resolve()))
    prop["batch-size"] = batch
    prop["gpu-id"] = gpu_id
    prop["model-engine-file"] = str(eng_rel)
    raw["property"] = prop

    reuse = eng_path.exists()
    print(f"[engine] onnx={onnx_logical_name} batch={batch} prec={prec} "
          f"engine={'REUSE' if reuse else 'BUILD'} -> {eng_path}")
    return _write_runtime_config(cfg_path, raw, batch, gpu_id)


def _clean_stale_engines(onnx_path: Path, onnx_logical_name: str,
                         target_engine: Path, force_rebuild: bool) -> None:
    """Delete stale engines for this ONNX; keep valid per-batch caches."""
    model_dir = onnx_path.parent
    if not model_dir.is_dir():
        return
    onnx_mtime = onnx_path.stat().st_mtime if onnx_path.exists() else 0.0
    for eng in model_dir.glob(f"{onnx_logical_name}_b*_gpu*_*.engine"):
        # ONNX re-exported after the engine was built -> engine is stale.
        if eng.stat().st_mtime < onnx_mtime:
            print(f"[engine] removing stale engine (older than ONNX): {eng.name}")
            eng.unlink()
    if force_rebuild and target_engine.exists():
        print(f"[engine] --force-rebuild-engine: removing {target_engine.name}")
        target_engine.unlink()


def _write_runtime_config(cfg_path: Path, raw: dict, batch: int,
                          gpu_id: int) -> str:
    """Write the patched config beside the original and return its path."""
    out = cfg_path.with_name(f"{cfg_path.stem}.runtime_b{batch}_gpu{gpu_id}.yml")
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"[engine] runtime config -> {out}")
    return str(out)
