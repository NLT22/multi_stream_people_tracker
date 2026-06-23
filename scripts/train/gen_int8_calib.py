#!/usr/bin/env python3
"""Generate a TensorRT INT8 calibration cache (for DeepStream nvinfer int8-calib-file).

DeepStream's nvinfer needs an int8-calib-file to enable INT8 (it ignores QDQ-ONNX and
falls back to FP16 otherwise). ONNX Runtime's calibrator computes per-tensor ranges and
write_calibration_table() emits `calibration.cache` in the TensorRT EntropyCalibration2
text format that nvinfer accepts. Implicit quantization: pair this cache with the ORIGINAL
FP ONNX + network-mode:1.

Low-memory: MinMax calibration (running min/max, no histograms) + few frames.

  ./venv/bin/python scripts/train/gen_int8_calib.py \
      --onnx models/yolov11/yolo11n_mmp.onnx \
      --out  models/yolov11/yolo11n_mmp_int8.calib \
      --img-dir dataset/mmp_exact_yolo/images/train --n 64
"""
from __future__ import annotations
import argparse, glob, os, random, shutil, tempfile
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import create_calibrator, CalibrationMethod, write_calibration_table
from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnx

IMG = 640


def _letterbox(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s = min(IMG / w, IMG / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)
    y0, x0 = (IMG - nh) // 2, (IMG - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(bgr, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


class Reader(CalibrationDataReader):
    def __init__(self, paths, name):
        self._it = iter(paths); self._name = name

    def get_next(self):
        for p in self._it:
            a = _letterbox(p)
            if a is not None:
                return {self._name: a}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default="dataset/mmp_exact_yolo/images/train")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    name = onnx.load(args.onnx).graph.input[0].name
    paths = glob.glob(str(Path(args.img_dir) / "**/*.jpg"), recursive=True)
    random.seed(0); random.shuffle(paths); paths = paths[: args.n]
    print(f"[calib] {len(paths)} frames, input={name}, MinMax")

    workdir = Path(tempfile.mkdtemp(prefix="trtcalib_"))
    calib = create_calibrator(
        args.onnx, op_types_to_calibrate=None,
        augmented_model_path=str(workdir / "aug.onnx"),
        calibrate_method=CalibrationMethod.MinMax)
    calib.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    calib.collect_data(Reader(paths, name))
    calib.compute_data()
    cwd = os.getcwd(); os.chdir(workdir)
    try:
        write_calibration_table(calib.compute_data())   # writes calibration.cache
    finally:
        os.chdir(cwd)
    cache = workdir / "calibration.cache"
    if not cache.exists():
        raise SystemExit(f"no calibration.cache produced in {workdir}")
    shutil.copy(cache, args.out)
    print(f"[calib] wrote {args.out}")
    print("--- head ---")
    print("\n".join(Path(args.out).read_text(errors='ignore').splitlines()[:4]))
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
