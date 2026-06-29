#!/usr/bin/env python3
"""Tune the global linker against real GT IDF1 for a warehouse. Builds tracklets + GT
ONCE (the 245 MB GT is too slow to reload), then sweeps link() params in-process.

Usage:
  mtmc_tune_linker.py --export-dir output/eval/mtmc_w020 \
    --calib .../Warehouse_020/calibration.json --gt-json .../ground_truth.json \
    --sources configs/sources/mtmc_val_w020.txt --max-frame 1799
"""
import argparse, importlib.util, itertools
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mtmc.mtmc_calib import WarehouseCalibration

spec = importlib.util.spec_from_file_location("gl", str(Path(__file__).resolve().parent / "mtmc_global_linker.py"))
GL = importlib.util.module_from_spec(spec); spec.loader.exec_module(GL)
spec2 = importlib.util.spec_from_file_location("s", str(Path(__file__).resolve().parent / "score_mtmc_idf1.py"))
S = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--max-frame", type=int, default=1799)
    args = ap.parse_args()

    src = [l.strip() for l in open(args.sources) if l.strip() and not l.strip().startswith("#")]
    cam_map = {i: int(p.split("Camera_")[-1].split(".")[0]) for i, p in enumerate(src)}
    cal = WarehouseCalibration(args.calib)
    print("[tune] building tracklets ...")
    keys, meta, raw_rows = GL.build_tracklets(Path(args.export_dir), cal, 0, 8, cam_map)
    print(f"[tune] {len(keys)} tracklets")

    print("[tune] loading GT (once) ...")
    gt = S.load_gt(args.gt_json)
    gt = {c: d[d.frame <= args.max_frame] for c, d in gt.items()}

    # pred rows once, keyed by GT cam; global_id filled per-config from tl_gid
    pred_base = {}
    for f in sorted(Path(args.export_dir).glob("cam_*_predictions.csv")):
        ec = int(f.stem.split("_")[1]); gtcam = cam_map.get(ec, ec)
        d = pd.read_csv(f).rename(columns={"frame_no_cam": "frame"})
        d = d[d.frame <= args.max_frame].copy(); d["_ec"] = ec
        pred_base[gtcam] = pd.concat([pred_base[gtcam], d], ignore_index=True) if gtcam in pred_base else d

    def score(tl_gid):
        pred = {}
        for gtcam, d in pred_base.items():
            d = d.copy()
            d["global_id"] = [tl_gid.get((int(ec), int(t)), -1) for ec, t in zip(d._ec, d.local_track_id)]
            pred[gtcam] = d[d.global_id >= 0]
        return S.global_idf1(gt, pred, iou_threshold=0.5)

    # oracle ceiling: assign each detection the GT pid of its IoU-matched GT box
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    def iom(A, B):
        ax2 = A[:, 0] + A[:, 2]; ay2 = A[:, 1] + A[:, 3]; bx2 = B[:, 0] + B[:, 2]; by2 = B[:, 1] + B[:, 3]
        x1 = np.maximum(A[:, 0][:, None], B[:, 0][None]); y1 = np.maximum(A[:, 1][:, None], B[:, 1][None])
        x2 = np.minimum(ax2[:, None], bx2[None]); y2 = np.minimum(ay2[:, None], by2[None])
        iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
        ua = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None] - inter
        return np.where(ua > 0, inter / ua, 0.0)
    po = {}
    for gtcam, d in pred_base.items():
        g = gt.get(gtcam)
        if g is None:
            continue
        d = d.copy(); d["global_id"] = -1
        for fr, dd in d.groupby("frame"):
            gg = g[g.frame == fr]
            if len(gg) == 0:
                continue
            M = iom(dd[["left", "top", "width", "height"]].values, gg[["left", "top", "width", "height"]].values)
            ri, ci = linear_sum_assignment(-M); idx = dd.index.values
            for rr, cc in zip(ri, ci):
                if M[rr, cc] >= 0.5:
                    d.loc[idx[rr], "global_id"] = int(gg.iloc[cc].person_id)
        po[gtcam] = d[d.global_id >= 0]
    ro = S.global_idf1(gt, po, iou_threshold=0.5)
    print(f"[tune] ORACLE (perfect IDs): IDF1={ro['idf1']:.4f} recall-ceiling="
          f"{ro['idtp']/(ro['idtp']+ro['idfn']):.3f}")

    base = dict(min_overlap=5, spatial_thr=1.5, conflict_thr=4.0, temporal_gap=400, pred_thr=3.5,
                reacq_gap=60, reacq_thr=1.5, temporal_weight=30.0, min_merge=0.5)
    # sweep the levers that matter for sparse 16-cam hand-offs
    grid = {
        "conflict_thr": [12.0, 18.0, 30.0],
        "spatial_thr":  [5.0, 7.0, 9.0],
    }
    keys_g = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys_g]):
        p = dict(base); p.update(dict(zip(keys_g, combo)))
        tl_gid, st = GL.link(keys, meta, SimpleNamespace(**p))
        r = score(tl_gid)
        tag = " ".join(f"{k}={p[k]}" for k in keys_g)
        results.append((r["idf1"], st["n_ids"], tag))
        print(f"  IDF1={r['idf1']:.4f} ids={st['n_ids']:4d}  {tag}")
    results.sort(reverse=True)
    print("\n[tune] BEST:")
    for idf1, nid, tag in results[:5]:
        print(f"  IDF1={idf1:.4f} ids={nid}  {tag}")


if __name__ == "__main__":
    main()
