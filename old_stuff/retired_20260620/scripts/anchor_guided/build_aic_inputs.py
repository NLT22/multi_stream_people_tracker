#!/usr/bin/env python3
"""Convert an MMP export (boxes + dense per-detection embeddings) into the EXACT
on-disk input format the AIC23 authors' aic_hungarian_cluster.py consumes:

    <work>/test_det/<scene>.txt   cam,frame,-1,x1,y1,x2,y2,score   (anchor sampling)
    <work>/test_emb/<scene>.npy   (N,512) aligned to test_det rows
    <work>/SCT/<scene>_<cam>.txt  frame,trkid,x,y,w,h,score,-1,-1,-1  (MOT, frame order)
    <work>/tracklet/<scene>_<cam>.pkl  {trkid: Tracklet(features=[emb per det, frame order])}

`cam` = source index (0-based, matches cam_<src>_predictions.csv). One detection
per (cam,frame,trkid) so features align 1:1 with SCT rows.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_types import Tracklet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="theirft_<scene> dir (dense emb + cam preds)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    pred = Path(args.pred_dir); work = Path(args.work)
    for sub in ("test_det", "test_emb", "SCT", "tracklet"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    z = np.load(pred / "detection_embeddings.npz")
    cam, frame, ltid, emb = z["cam_id"], z["frame_no"], z["local_track_id"], z["embeddings"]
    # box lookup per (cam,frame,ltid)
    box = {}
    cams = sorted(set(int(c) for c in cam))
    for src in cams:
        df = pd.read_csv(pred / f"cam_{src}_predictions.csv")
        for r in df.itertuples():
            box[(src, int(r.frame_no_cam), int(r.local_track_id))] = \
                (float(r.left), float(r.top), float(r.width), float(r.height))

    det_rows, det_emb = [], []
    for src in cams:
        m = cam == src
        cf, cl, ce = frame[m], ltid[m], emb[m]
        order = np.argsort(cf, kind="stable")
        cf, cl, ce = cf[order], cl[order], ce[order]
        sct_lines = []
        feats = defaultdict(list)
        for f, l, e in zip(cf, cl, ce):
            b = box.get((src, int(f), int(l)))
            if b is None:
                continue
            x, y, w, h = b
            sct_lines.append(f"{int(f)},{int(l)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1")
            feats[int(l)].append(e.astype(np.float32))
            det_rows.append(f"{src},{int(f)},-1,{x:.2f},{y:.2f},{x+w:.2f},{y+h:.2f},1.0")
            det_emb.append(e.astype(np.float32))
        (work / "SCT" / f"{args.scene}_{src}.txt").write_text("\n".join(sct_lines) + "\n")
        tracklets = {l: Tracklet(features=v) for l, v in feats.items()}
        with open(work / "tracklet" / f"{args.scene}_{src}.pkl", "wb") as fp:
            pickle.dump(tracklets, fp)
        print(f"  cam{src}: {len(sct_lines)} dets, {len(tracklets)} tracklets")

    (work / "test_det" / f"{args.scene}.txt").write_text("\n".join(det_rows) + "\n")
    np.save(work / "test_emb" / f"{args.scene}.npy", np.stack(det_emb))
    print(f"[build] {len(det_rows)} detections, {len(cams)} cams -> {work}")


if __name__ == "__main__":
    main()
