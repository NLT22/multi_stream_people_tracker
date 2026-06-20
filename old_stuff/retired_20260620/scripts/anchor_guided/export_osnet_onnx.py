#!/usr/bin/env python3
"""Export a fine-tuned OSNet (.pth) to ONNX for use as NvDCF in-tracker ReID.

NvDCF reidType:2 feeds preprocessed crops (NCHW, 256x128) and expects a feature
vector out. OSNet x1_0 -> 512-d. Set reidFeatureSize: 512 in the tracker config.

  python scripts/anchor_guided/export_osnet_onnx.py \
      --ckpt models/reid_osnet_mmp/osnet_mmp_retail.pth \
      --out  models/reid_osnet_mmp/osnet_mmp_retail.onnx
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    from torchreid.reid.utils import FeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    # FeatureExtractor loads the fine-tuned weights into an osnet_x1_0 module.
    ext = FeatureExtractor(model_name="osnet_x1_0", model_path=args.ckpt,
                           device="cuda", image_size=(256, 128))
    model = ext.model.eval()
    dummy = torch.randn(1, 3, 256, 128, device="cuda")
    with torch.no_grad():
        feat = model(dummy)
    print(f"[export] OSNet feature dim = {tuple(feat.shape)}  (set reidFeatureSize accordingly)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[export] wrote ONNX -> {args.out}")


if __name__ == "__main__":
    main()
