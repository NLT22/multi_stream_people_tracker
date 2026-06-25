"""
Window-size ablation on the full 24-scene MMPTracking val set.

Sweeps window_chunks = [1, 2, 3, 4] using the existing per-scene exports
(200-frame chunks) under output/eval/full_mmp_val/64pm_*/.
No pipeline re-run needed — only live_buffered is re-run.

Usage:
    python scripts/eval/window_ablation_fullval.py \
        [--export-root output/eval/full_mmp_val] \
        [--val-root dataset/MMPTracking_10minute/val] \
        [--windows 1,2,3,4] \
        [--assign-thr 0.40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.eval.mmp_metrics.core import _eval_global_idf1
from src.mtmc.live_buffered import main as lb_main


def run_lb(export_dir: Path, assign_csv: Path, wc: int, thr: float) -> None:
    sys.argv = [
        "live_buffered",
        "--export-dir", str(export_dir),
        "--window-chunks", str(wc),
        "--assign-thr", str(thr),
        "--assign-csv", str(assign_csv),
        "--once",
    ]
    lb_main()


def score(scene_dir: Path, val_root: Path, scene: str, assign_csv: Path) -> float | None:
    assign = pd.read_csv(assign_csv)
    gid_map = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
               for r in assign.itertuples()}
    cam_files  = sorted(scene_dir.glob("cam_*_predictions.csv"))
    source_ids = [int(p.stem.split("_")[1]) for p in cam_files]
    scene_data = val_root / scene
    gt_cam_ids = sorted(int(p.stem[3:]) for p in scene_data.glob("cam*.mp4"))
    all_gt, all_pred = {}, {}
    for src_id, gt_cam_id in zip(source_ids, gt_cam_ids):
        pred_path = scene_dir / f"cam_{src_id}_predictions.csv"
        gt_path   = scene_data / f"gt_cam{gt_cam_id}_clean.csv"
        if not gt_path.exists(): gt_path = scene_data / f"gt_cam{gt_cam_id}.csv"
        if not pred_path.exists() or not gt_path.exists(): continue
        pred = pd.read_csv(pred_path).copy()
        pred["global_id"] = [gid_map.get((src_id, int(f), int(t)), -1)
                             for f, t in zip(pred["frame_no_cam"], pred["local_track_id"])]
        pred = pred[pred["global_id"] >= 0].rename(columns={"frame_no_cam": "frame"})
        all_gt[gt_cam_id]  = pd.read_csv(gt_path)
        all_pred[gt_cam_id] = pred
    if not all_gt: return None
    r = _eval_global_idf1(all_gt, all_pred, iou_threshold=0.5)
    return r.get("idf1") or r.get("global_idf1") or r.get("mean_idf1")


def env_of(scene: str) -> str:
    return scene.removeprefix("64pm_").rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", default="output/eval/full_mmp_val")
    ap.add_argument("--val-root",    default="dataset/MMPTracking_10minute/val")
    ap.add_argument("--windows",     default="1,2,3,4",
                    help="comma-separated window_chunks values to test")
    ap.add_argument("--chunk-frames", type=int, default=100,
                    help="frames per chunk (used only for display labels)")
    ap.add_argument("--assign-thr",  type=float, default=0.40)
    args = ap.parse_args()

    export_root = Path(args.export_root)
    val_root    = Path(args.val_root)
    windows      = [int(w) for w in args.windows.split(",")]
    chunk_frames = args.chunk_frames
    wlabel = {wc: f"{wc * chunk_frames}f" for wc in windows}

    scene_dirs = sorted(d for d in export_root.glob("64pm_*") if d.is_dir()
                        and list(d.glob("cam_*_predictions.csv")))
    if not scene_dirs:
        print(f"No scenes found under {export_root}"); sys.exit(1)

    # results[scene][wc] = idf1
    results: dict[str, dict[int, float | None]] = {}

    for scene_dir in scene_dirs:
        scene = scene_dir.name
        results[scene] = {}
        print(f"\n── {scene} " + "─" * 30)
        for wc in windows:
            assign_csv = scene_dir / f"_ablation_wc{wc}.csv"
            if not assign_csv.exists():
                run_lb(scene_dir, assign_csv, wc, args.assign_thr)
            idf1 = score(scene_dir, val_root, scene, assign_csv)
            results[scene][wc] = idf1
            print(f"  wc={wc}: {idf1:.4f}" if idf1 is not None else f"  wc={wc}: N/A")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Scene':<34}", end="")
    for wc in windows: print(f" {wlabel[wc]:>6}", end="")
    print("  best")
    print("─" * 70)

    env_bests: dict[str, list[float]] = {}
    overall_best: dict[int, list[float]] = {wc: [] for wc in windows}

    for scene in sorted(results):
        row = results[scene]
        best_wc  = max((wc for wc in windows if row.get(wc) is not None),
                       key=lambda wc: row[wc], default=None)
        best_val = row[best_wc] if best_wc else None
        print(f"{scene:<34}", end="")
        for wc in windows:
            v = row.get(wc)
            tag = "*" if wc == best_wc else " "
            print(f" {v:.3f}{tag}" if v else "  N/A ", end="")
        print(f"  {wlabel[best_wc]}" if best_wc else "")
        env = env_of(scene)
        env_bests.setdefault(env, []).append(best_val or 0.0)
        for wc in windows:
            if row.get(wc) is not None:
                overall_best[wc].append(row[wc])

    print("\n── Per-environment means ─────────────────────────────────────────────")
    print(f"{'Env':<26}", end="")
    for wc in windows: print(f"  {wlabel[wc]:>6}", end="")
    print()

    env_rows: dict[str, dict[int, list[float]]] = {}
    for scene in results:
        env = env_of(scene)
        env_rows.setdefault(env, {wc: [] for wc in windows})
        for wc in windows:
            if results[scene].get(wc) is not None:
                env_rows[env][wc].append(results[scene][wc])

    for env in sorted(env_rows):
        print(f"{env:<26}", end="")
        best_wc_env = None; best_mean = -1
        for wc in windows:
            vals = env_rows[env][wc]
            m = sum(vals)/len(vals) if vals else 0.0
            if m > best_mean: best_mean = m; best_wc_env = wc
        for wc in windows:
            vals = env_rows[env][wc]
            m = sum(vals)/len(vals) if vals else 0.0
            tag = "*" if wc == best_wc_env else " "
            print(f"  {m:.4f}{tag}", end="")
        print()

    print("\n── Overall means ─────────────────────────────────────────────────────")
    print(f"{'MEAN (all scenes)':<26}", end="")
    best_wc_all = None; best_all = -1
    for wc in windows:
        vals = overall_best[wc]
        m = sum(vals)/len(vals) if vals else 0.0
        if m > best_all: best_all = m; best_wc_all = wc
    for wc in windows:
        vals = overall_best[wc]
        m = sum(vals)/len(vals) if vals else 0.0
        tag = "*" if wc == best_wc_all else " "
        print(f"  {m:.4f}{tag}", end="")
    print(f"  → best overall: wc={best_wc_all} ({best_all:.4f})")

    # Save JSON summary
    out = {"windows": windows, "assign_thr": args.assign_thr, "results": {
        s: {str(wc): v for wc, v in row.items()} for s, row in results.items()
    }}
    (export_root / "window_ablation_fullval.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {export_root}/window_ablation_fullval.json")


if __name__ == "__main__":
    main()
