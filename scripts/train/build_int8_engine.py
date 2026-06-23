#!/usr/bin/env python3
"""Build a TensorRT INT8 engine for the YOLO11 detector, for DeepStream nvinfer.

Why this exists: DeepStream's nvinfer needs either a valid TRT INT8 calibration cache
or a prebuilt INT8 engine. The ORT-exported calibration table had incomplete tensor
coverage (TRT threw `_Map_base::at`). This runs a real TRT IInt8EntropyCalibrator2 over
MMP frames -> a COMPLETE cache, builds the engine, and writes it to the exact path the
nvinfer config expects (model-engine-file), so DeepStream just loads it.

TRT-python must match DeepStream's libnvinfer (here 10.16.1.11). Uses torch for the
calibrator's device buffers (no pycuda needed).

  ./venv/bin/python scripts/train/build_int8_engine.py \
      --onnx models/yolov11/yolo11n_mmp.onnx \
      --engine models/yolov11/yolo11n_mmp.onnx_b4_gpu0_int8.engine \
      --cache  models/yolov11/yolo11n_mmp_int8.entropy.cache \
      --img-dir dataset/mmp_exact_yolo/images/train --n 256 --batch 4
"""
from __future__ import annotations

import argparse, glob, os, random
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

IMG = 640


def _letterbox(path, H=IMG, W=IMG):
    """YOLO detector preproc: letterbox to WxH, RGB, /255."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((H, W, 3), 114, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(bgr, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


# ReID preproc must match nvinfer SGIE: resize (no pad) to WxH, RGB, (px - offsets)*scale
_REID_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_REID_SCALE = 0.01735207


def _reid_preproc(path, H=256, W=128):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - _REID_MEAN) * _REID_SCALE
    return rgb.transpose(2, 0, 1)[None]


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files, batch, cache, preproc, hw):
        super().__init__()
        self.files = files
        self.batch = batch
        self.cache = cache
        self.preproc = preproc
        self.hw = hw                     # (H, W)
        self.idx = 0
        self._dbuf = None

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.idx + self.batch > len(self.files):
            return None
        arrs = []
        H, W = self.hw
        for p in self.files[self.idx:self.idx + self.batch]:
            a = self.preproc(p, H, W)
            if a is None:
                a = np.zeros((1, 3, H, W), np.float32)
            arrs.append(a)
        self.idx += self.batch
        batch = np.ascontiguousarray(np.concatenate(arrs, 0))
        self._dbuf = torch.from_numpy(batch).cuda().contiguous()  # keep alive
        if self.idx % (self.batch * 8) == 0:
            print(f"  [calib] {self.idx}/{len(self.files)}")
        return [int(self._dbuf.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            print(f"  [calib] reusing cache {self.cache}")
            return Path(self.cache).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache).write_bytes(cache)
        print(f"  [calib] wrote cache {self.cache} ({len(cache)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--img-dir", default="dataset/mmp_exact_yolo/images/train")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4, help="max/opt batch (match nvinfer batch-size)")
    ap.add_argument("--mode", default="yolo", choices=["yolo", "reid"],
                    help="yolo: letterbox 640 /255. reid: resize 256x128 + ImageNet normalize.")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--explicit", action="store_true",
                    help="explicit-precision: trust the QDQ ONNX's embedded scales (no "
                         "calibrator). Use for the Conv-only QDQ model so TRT leaves the "
                         "YOLO11 head in FP (avoids the implicit-INT8 head Myelin failure).")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[trt] {trt.__version__}  device={torch.cuda.get_device_name(0)}")
    H, W = (256, 128) if args.mode == "reid" else (IMG, IMG)
    preproc = _reid_preproc if args.mode == "reid" else _letterbox
    print(f"[trt] mode={args.mode} input HxW={H}x{W}")

    paths = glob.glob(str(Path(args.img_dir) / "**/*.jpg"), recursive=True)
    random.seed(0); random.shuffle(paths); paths = paths[: args.n]
    paths = paths[: (len(paths) // args.batch) * args.batch]   # drop remainder
    print(f"[trt] {len(paths)} calibration frames, batch={args.batch}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[onnx-err]", parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    inp = network.get_input(0)
    name = inp.name
    print(f"[trt] input '{name}' shape={inp.shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)   # FP16 fallback for layers TRT won't INT8

    profile = builder.create_optimization_profile()
    profile.set_shape(name, (1, 3, H, W), (args.batch, 3, H, W), (args.batch, 3, H, W))
    config.add_optimization_profile(profile)

    if args.explicit:
        print("[trt] explicit-precision mode: using QDQ scales embedded in the ONNX")
    else:
        calib = EntropyCalibrator(paths, args.batch, args.cache, preproc, (H, W))
        config.int8_calibrator = calib
        config.set_calibration_profile(profile)

    print("[trt] building INT8 engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    Path(args.engine).write_bytes(serialized)
    mb = serialized.nbytes / 1e6
    print(f"[trt] wrote engine {args.engine} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
