#!/usr/bin/env python3
"""Static INT8 (QDQ) quantization of the YOLO11 detector via ONNX Runtime.

Produces a QDQ ONNX whose quant scales are derived from real MMP frames, matching
the DeepStream nvinfer preprocessing (letterbox 640x640, RGB, scale 1/255).
TensorRT/DeepStream then builds an INT8 engine directly from the QDQ nodes
(network-mode:1) — no separate calibration table needed.

  ./venv/bin/python scripts/train/quantize_yolo_int8.py \
      --onnx models/yolov11/yolo11n_mmp.onnx \
      --out  models/yolov11/yolo11n_mmp_int8.onnx \
      --img-dir dataset/mmp_exact_yolo/images/train --n 256
"""
from __future__ import annotations
import argparse, glob, random
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod)

IMG = 640


def _letterbox(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s = min(IMG / w, IMG / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)          # symmetric letterbox pad
    y0, x0 = (IMG - nh) // 2, (IMG - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]                     # (1,3,640,640)


class MMPReader(CalibrationDataReader):
    def __init__(self, paths, input_name):
        self._it = iter(paths)
        self._name = input_name

    def get_next(self):
        for p in self._it:
            arr = _letterbox(p)
            if arr is not None:
                return {self._name: arr}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default="dataset/mmp_exact_yolo/images/train")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--method", default="minmax", choices=["entropy", "minmax", "percentile"])
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel weights (TRT rejects it for YOLO11 attn scalar consts; "
                         "default per-tensor)")
    ap.add_argument("--ops", default="Conv",
                    help="comma-list of op types to quantize. Default 'Conv' (conv backbone "
                         "only) — quantizing the attn/MatMul ops creates QDQ on scalar consts "
                         "that TensorRT cannot parse. Use 'all' to quantize everything.")
    args = ap.parse_args()

    # opset >= 13 needed for per-channel QDQ
    m = onnx.load(args.onnx)
    if m.opset_import[0].version < 13:
        m = version_converter.convert_version(m, 13)
        prep = str(Path(args.out).with_suffix(".op13.onnx"))
        onnx.save(m, prep)
        src = prep
    else:
        src = args.onnx
    input_name = onnx.load(src).graph.input[0].name

    paths = glob.glob(str(Path(args.img_dir) / "**/*.jpg"), recursive=True)
    random.seed(0); random.shuffle(paths); paths = paths[: args.n]
    print(f"[quant] calibrating on {len(paths)} frames, input={input_name}, method={args.method}")

    method = {"entropy": CalibrationMethod.Entropy, "minmax": CalibrationMethod.MinMax,
              "percentile": CalibrationMethod.Percentile}[args.method]
    q_kwargs = dict(
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=args.per_channel, calibrate_method=method,
        # QuantizeBias:False keeps Conv bias in FP — TensorRT cannot parse a
        # DequantizeLinear on bias (fails at node "*.conv.bias").
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False},
    )
    if args.ops != "all":
        q_kwargs["op_types_to_quantize"] = args.ops.split(",")
    quantize_static(src, args.out, MMPReader(paths, input_name), **q_kwargs)
    print(f"[quant] wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
