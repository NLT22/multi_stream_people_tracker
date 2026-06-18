#!/usr/bin/env python3
"""Convert MMPTracking calibration -> NvDCF Single-View-3D (SV3DT) camInfo YAMLs.

MMP gives pinhole K (Fx,Fy,Cx,Cy), R (3x3), t (mm) with p = K@(R@Pw + t).
NvDCF SV3DT wants a 3x4 world->pixel projection matrix; that is P = K @ [R|t]
(use projectionMatrix_3x4_w2p, which already includes the principal point).
World units = mm (MMP native), so modelInfo height/radius are in mm.

Writes camInfo-<src>.yml ordered by the dataset's source_id -> cam mapping, so
the list matches the tracker's source order.

  python scripts/anchor_guided/mmp_calib_to_caminfo.py \
      --short-root dataset/MMPTracking_10minute/val --scene 64pm_office_0 \
      --out-dir configs/caminfo/64pm_office_0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--height-mm", type=float, default=1700.0)
    ap.add_argument("--radius-mm", type=float, default=300.0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.dataset.mmp_tracking import MMPTrackingShortDataset
    from src.reid.geometry import GroundPlaneGeometry

    ds = MMPTrackingShortDataset(str(args.short_root), args.scene)
    cam_ids = ds.get_cam_ids()
    geo = GroundPlaneGeometry(ds.load_calibration())   # parses K,R,t per cam_id

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for src, cam in enumerate(cam_ids):
        c = geo._cams[cam]                       # {"K","R","t"}
        Rt = np.hstack([c["R"], c["t"].reshape(3, 1)])   # 3x4 [R|t]
        P = c["K"] @ Rt                          # 3x4 world(mm) -> pixel
        vals = P.flatten().tolist()              # row-major 12 values
        f = out / f"camInfo-{src:02d}.yml"
        with f.open("w") as fp:
            fp.write("%YAML:1.0\n---\n")
            fp.write("projectionMatrix_3x4_w2p:\n")
            for v in vals:
                fp.write(f"  - {v:.6f}\n")
            fp.write("modelInfo:\n")
            fp.write(f"  height: {args.height_mm}\n")
            fp.write(f"  radius: {args.radius_mm}\n")
        paths.append(str(f))
        print(f"  src{src} (cam{cam}) -> {f}")
    print("\ncameraModelFilepath list for the tracker config:")
    for p in paths:
        print(f"    - '{p}'")


if __name__ == "__main__":
    main()
