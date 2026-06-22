#!/usr/bin/env python3
"""Diagnose which stage limits retail (production_todo 3.5): detector recall vs
tracker fragmentation vs cross-camera ReID confusion.

For each env it scores the buffered export and reports per-env:
  recall          (detector: low => missed detections)
  local IDF1      (per-camera single-cam identity = detection+tracking quality)
  switches/frags  (tracker fragmentation)
  Global IDF1     (cross-camera identity)
  local-global gap (big gap => cross-camera ReID confusion, not local tracking)

  python scripts/eval/diagnose_retail.py --export-dir <dir> \
    --map "cafe=64pm_cafe_shop_0:0-3,...,retail=64pm_retail_0:16-19"
"""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
import pandas as pd
import motmetrics as mm
from src.eval.mmp_metrics.core import _eval_scene, _eval_global_idf1


def _expand(rng):
    out = []
    for seg in rng.split("+"):
        if "-" in seg:
            a, b = seg.split("-"); out += list(range(int(a), int(b) + 1))
        elif seg:
            out.append(int(seg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True, type=Path)
    ap.add_argument("--map", required=True)
    ap.add_argument("--val-root", default="dataset/MMPTracking_10minute/val")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="cap predictions to this frame_no (use first loop of a looped export)")
    args = ap.parse_args()

    am = pd.read_csv(args.export_dir / "_eval_assign.csv")
    gid_of = {(r.group, int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
              for r in am.itertuples()}
    mh = mm.metrics.create()
    metrics = ["idf1", "num_switches", "num_fragmentations", "num_misses",
               "num_false_positives", "recall"]

    rows = []
    for part in args.map.split(","):
        grp, rest = part.split("="); scene, rng = rest.split(":")
        gcams = _expand(rng); local = {g: j for j, g in enumerate(gcams)}
        tmp = Path(tempfile.mkdtemp(prefix=f"diag_{scene}_"))
        for g in gcams:
            src = args.export_dir / f"cam_{g}_predictions.csv"
            if not src.exists():
                continue
            df = pd.read_csv(src)
            df["global_id"] = [gid_of.get((grp, g, int(f), int(t)), -1)
                               for f, t in zip(df["frame_no_cam"], df["local_track_id"])]
            df["cam_id"] = local[g]
            if args.max_frame is not None:
                df = df[df["frame_no_cam"] <= args.max_frame]
            df.to_csv(tmp / f"cam_{local[g]}_predictions.csv", index=False)
        gt_suffix = "_clean" if "retail" in scene else ""
        res = _eval_scene(scene, Path(args.val_root), tmp, 0.5, 0.0, 0.0, 0.0,
                          None, None, None, gt_suffix=gt_suffix)
        if not res or not res["per_cam_accs"]:
            continue
        summary = mh.compute_many(list(res["per_cam_accs"].values()),
                                  metrics=metrics, generate_overall=True)
        ov = summary.loc["OVERALL"]
        g = _eval_global_idf1(res["all_gt"], res["all_pred"], 0.5)
        rows.append({
            "env": grp,
            "recall": ov["recall"],
            "local_idf1": ov["idf1"],
            "switches": int(ov["num_switches"]),
            "frags": int(ov["num_fragmentations"]),
            "global_idf1": g["idf1"],
            "local_global_gap": ov["idf1"] - g["idf1"],
        })

    t = pd.DataFrame(rows).set_index("env")
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:.3f}")
    print("\n=== Per-env diagnostic ===")
    print(t)

    if "retail" in t.index:
        others = t.drop("retail")
        r = t.loc["retail"]
        print("\n=== Retail vs others (mean) ===")
        for col in ["recall", "local_idf1", "switches", "frags", "global_idf1", "local_global_gap"]:
            print(f"  {col:18s} retail={r[col]:.3f}   others_mean={others[col].mean():.3f}")
        print("\n=== Verdict ===")
        if r["recall"] < others["recall"].mean() - 0.10:
            print("  -> DETECTOR RECALL is the retail limiter (retail recall notably lower).")
        elif r["local_idf1"] < others["local_idf1"].mean() - 0.10:
            print("  -> LOCAL tracking (detector+tracker fragmentation) limits retail.")
        elif r["local_global_gap"] > others["local_global_gap"].mean() + 0.10:
            print("  -> CROSS-CAMERA ReID CONFUSION limits retail (local IDF1 ok, global collapses).")
        else:
            print("  -> retail degrades broadly; no single dominant stage by these thresholds.")


if __name__ == "__main__":
    main()
