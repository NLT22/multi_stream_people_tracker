#!/usr/bin/env python3
"""Score a global-linker assignment for a warehouse, remapping export cam_N -> the real
GT/calibration camera number via the source list (W020/W021 cameras are non-contiguous).

Usage:
  score_mtmc_w.py --export-dir output/eval/mtmc_w020 --assign <gl.csv> \
    --gt-json .../Warehouse_020/ground_truth.json --sources configs/sources/mtmc_val_w020.txt --max-frame 1799
"""
import argparse, importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location("s", str(Path(__file__).resolve().parent / "score_mtmc_idf1.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def cam_map_from_sources(path):
    src = [l.strip() for l in open(path) if l.strip() and not l.strip().startswith("#")]
    return {i: int(p.split("Camera_")[-1].split(".")[0]) for i, p in enumerate(src)}


def load_pred_remapped(export_dir, assign_csv, cam_map, max_frame):
    a = pd.read_csv(assign_csv)
    gid = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id) for r in a.itertuples()}
    pred = {}
    for f in sorted(Path(export_dir).glob("cam_*_predictions.csv")):
        ec = int(f.stem.split("_")[1])
        gtcam = cam_map.get(ec, ec)
        d = pd.read_csv(f).rename(columns={"frame_no_cam": "frame"})
        d = d[d.frame <= max_frame].copy()
        d["global_id"] = [gid.get((ec, int(fr), int(t)), -1) for fr, t in zip(d.frame, d.local_track_id)]
        d = d[d.global_id >= 0]
        if gtcam in pred:
            pred[gtcam] = pd.concat([pred[gtcam], d], ignore_index=True)
        else:
            pred[gtcam] = d
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--max-frame", type=int, default=1799)
    args = ap.parse_args()
    cam_map = cam_map_from_sources(args.sources)
    gt = S.load_gt(args.gt_json)
    gt = {c: d[d.frame <= args.max_frame] for c, d in gt.items()}
    pred = load_pred_remapped(args.export_dir, args.assign, cam_map, args.max_frame)
    r = S.global_idf1(gt, pred, iou_threshold=0.5)
    print(f"Global IDF1 = {r['idf1']:.4f}  predIDs={r['num_pred_ids']} gtIDs={r['num_gt_ids']} "
          f"TP={r['idtp']} FP={r['idfp']} FN={r['idfn']}")


if __name__ == "__main__":
    main()
