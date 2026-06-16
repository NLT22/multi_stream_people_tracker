#!/usr/bin/env python3
"""Fit the per-environment topdown-grid -> world(mm) affine and write it to
scripts/tracktacular/affines.json (consumed by mmptracking_dataset.py and
bev_compare.py).

Method: for matched person_ids, project foot points to world via the calibration
(src.reid.geometry), take a robust median across cameras, and least-squares fit
[gx, gy, 1] -> [world_x_mm, world_y_mm]. Also records the world box (grid corners
through the affine, + margin) for outlier rejection in the BEV evaluator.
"""
from __future__ import annotations
import argparse, json, sys, zipfile, io
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reid.geometry import GroundPlaneGeometry

OUT = Path(__file__).resolve().parent / "affines.json"


def _scene_parts(scene):
    session, rest = scene.split("_", 1)
    env = rest.rsplit("_", 1)[0]
    return session, env, rest


def fit(env, scene, mmp_root, split, samples_step):
    session, _env, inst = _scene_parts(scene)
    base = Path(mmp_root) / ("MMPTracking_training/train" if split == "train"
                             else "MMPTracking_validation/validation")
    calib = json.load(open(base / "calibrations" / env / "calibrations.json"))
    geo = GroundPlaneGeometry(calib)
    cams = [c["CameraId"] for c in calib["Cameras"]]

    tz = zipfile.ZipFile(base / "topdown_labels" / session / f"{inst}.zip")
    lz = zipfile.ZipFile(base / "labels" / session / f"{inst}.zip")
    lbl = {}  # frame -> cam -> {pid:bbox}
    for n in lz.namelist():
        b = n.split("/")[-1]
        if b.startswith("rgb_") and b.endswith(".json"):
            _, fr, cam = b[:-5].split("_")
            lbl.setdefault(int(fr), {})[int(cam)] = json.loads(lz.read(n))

    G, W = [], []
    td_names = sorted(n for n in tz.namelist()
                      if n.split("/")[-1].startswith("topdown_"))
    for n in td_names[::samples_step]:
        fr = int(n.split("/")[-1][len("topdown_"):-4])
        rows = [l.split(",") for l in tz.read(n).decode().splitlines() if len(l.split(",")) >= 3]
        for p in rows:
            pid, gx, gy = int(float(p[0])), float(p[1]), float(p[2])
            ws = []
            for cam in cams:
                d = lbl.get(fr, {}).get(cam, {})
                if str(pid) in d:
                    x1, y1, x2, y2 = d[str(pid)]
                    w = geo.foot_to_world(cam, (x1 + x2) / 2.0, y2)
                    if w is not None and abs(w[0]) < 1e5 and abs(w[1]) < 1e5:
                        ws.append(w)
            if len(ws) >= 2:
                ws = np.array(ws)
                med = np.median(ws, axis=0)
                # keep cameras that agree with the median (env-agnostic)
                inl = ws[np.linalg.norm(ws - med, axis=1) < 2500]
                if len(inl) >= 2:
                    G.append([gx, gy, 1.0]); W.append(inl.mean(axis=0))
    G, W = np.array(G), np.array(W)
    M, *_ = np.linalg.lstsq(G, W, rcond=None)        # (3,2): [gx,gy,1]->[x,y]
    res = np.linalg.norm(G @ M - W, axis=1)
    A = np.vstack([M.T, [0, 0, 1]])                  # 3x3 homogeneous
    # world box from grid corners (grid assumed 0..256), + 2 m margin
    import itertools
    corners = np.array([[g[0], g[1], 1] for g in itertools.product([0, 256], [0, 256])]).T
    wc = (A @ corners)[:2]
    box = [float(wc[0].min()) - 2000, float(wc[0].max()) + 2000,
           float(wc[1].min()) - 2000, float(wc[1].max()) + 2000]
    return A, box, len(G), float(np.median(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mmp-root", default="dataset/MMPTracking")
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples-step", type=int, default=60)
    args = ap.parse_args()
    A, box, n, med = fit(args.env, args.scene, args.mmp_root, args.split, args.samples_step)
    print(f"[{args.env}] samples={n} residual_median={med:.0f}mm")
    print("affine=", A.tolist())
    data = json.load(open(OUT)) if OUT.exists() else {}
    data[args.env] = {"affine": A.tolist(), "box": box}
    json.dump(data, open(OUT, "w"), indent=2)
    print(f"wrote {args.env} -> {OUT}")


if __name__ == "__main__":
    main()
