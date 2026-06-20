"""Sweep nearline merge parameters for MMPTracking_short exports.

This script assumes DeepStream predictions already exist, e.g.

    output/eval/mmp_lobby_0_nvdcf_realtime_baseline/

It runs nearline remap + metrics_mmp for each parameter set, then ranks the
sets by micro-averaged Global IDF1 across scenes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
import sys
from pathlib import Path


GLOBAL_RE = re.compile(
    r"Global IDF1:\s*([0-9.]+)\s*"
    r"\(IDTP=(\d+)\s+IDFP=(\d+)\s+IDFN=(\d+).*?Pred IDs=(\d+)\)"
)


def _log(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


def _float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep nearline_merge params and rank by Global IDF1")
    p.add_argument("--short-root", default="dataset/MMPTracking_short")
    p.add_argument("--pred-root", default="output/eval")
    p.add_argument("--out-root", default="output/eval/nearline_sweep")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "lobby_0",
            "industry_safety_0",
            "office_0",
            "cafe_shop_0",
            "lobby_3",
            "industry_safety_4",
            "office_2",
            "cafe_shop_3",
        ],
        help="Scenes to sweep. Keep retail separate until occlusion policy is settled.",
    )
    p.add_argument("--pred-suffix", default="nvdcf_realtime_baseline")
    p.add_argument("--thresholds", default="0.62,0.65,0.68,0.70")
    p.add_argument("--margins", default="0.02,0.03,0.04,0.05")
    p.add_argument("--geo-weights", default="0.15,0.25,0.35")
    p.add_argument("--geo-min-overlaps", default="4,8,12")
    p.add_argument("--window-frames", default="125")
    p.add_argument("--delay-frames", type=int, default=50)
    p.add_argument("--min-gid-embeddings", type=int, default=6)
    p.add_argument("--min-tracklet-detections", type=int, default=10)
    p.add_argument("--geo-sample-step", type=int, default=5)
    p.add_argument("--limit", type=int, default=0,
                   help="Run only first N parameter sets, useful for smoke tests.")
    p.add_argument("--keep-outputs", action="store_true",
                   help="Keep each remapped prediction directory. Default removes it.")
    return p.parse_args()


def _pred_dir(pred_root: Path, scene: str, suffix: str) -> Path:
    return pred_root / f"mmp_{scene}_{suffix}"


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}"
        )
    return proc.stdout


def _eval_global(
    short_root: Path,
    scene: str,
    pred_dir: Path,
) -> tuple[float, int, int, int, int, str]:
    output = _run([
        sys.executable,
        "-m",
        "src.eval.metrics_mmp",
        "--short-root",
        str(short_root),
        "--scene",
        scene,
        "--pred-dir",
        str(pred_dir),
    ])
    matches = GLOBAL_RE.findall(output)
    if not matches:
        raise RuntimeError(f"Could not parse Global IDF1 for {scene}\n{output}")
    idf1, idtp, idfp, idfn, pred_ids = matches[-1]
    return (
        float(idf1),
        int(idtp),
        int(idfp),
        int(idfn),
        int(pred_ids),
        output,
    )


def main() -> None:
    args = _parse_args()
    short_root = Path(args.short_root)
    pred_root = Path(args.pred_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = _float_list(args.thresholds)
    margins = _float_list(args.margins)
    geo_weights = _float_list(args.geo_weights)
    geo_min_overlaps = _int_list(args.geo_min_overlaps)
    window_frames = _int_list(args.window_frames)

    combos = list(itertools.product(
        thresholds,
        margins,
        geo_weights,
        geo_min_overlaps,
        window_frames,
    ))
    if args.limit > 0:
        combos = combos[:args.limit]

    rows: list[dict] = []
    _log(f"[sweep] scenes={args.scenes}")
    _log(f"[sweep] param_sets={len(combos)} out_root={out_root}")

    for idx, (thr, margin, geo_w, geo_ov, window) in enumerate(combos, start=1):
        tag = (
            f"t{thr:.2f}_m{margin:.2f}_g{geo_w:.2f}_"
            f"ov{geo_ov}_w{window}"
        ).replace(".", "p")
        total_idtp = total_idfp = total_idfn = 0
        scene_scores: dict[str, float] = {}
        scene_pred_ids: dict[str, int] = {}
        failed = False

        _log(f"\n[sweep] {idx}/{len(combos)} {tag}")
        for scene in args.scenes:
            pred_dir = _pred_dir(pred_root, scene, args.pred_suffix)
            if not (pred_dir / "tracklets.csv").exists():
                _log(f"  [skip] {scene}: missing {pred_dir}/tracklets.csv")
                failed = True
                break

            out_dir = out_root / tag / scene
            if out_dir.exists():
                shutil.rmtree(out_dir)

            try:
                _run([
                    sys.executable,
                    "-m",
                    "src.eval.nearline_merge",
                    "--pred-dir",
                    str(pred_dir),
                    "--out-dir",
                    str(out_dir),
                    "--threshold",
                    str(thr),
                    "--margin",
                    str(margin),
                    "--min-gid-embeddings",
                    str(args.min_gid_embeddings),
                    "--min-tracklet-detections",
                    str(args.min_tracklet_detections),
                    "--mmp-short-root",
                    str(short_root),
                    "--scene",
                    scene,
                    "--geo-weight",
                    str(geo_w),
                    "--geo-sample-step",
                    str(args.geo_sample_step),
                    "--geo-min-overlaps",
                    str(geo_ov),
                    "--window-frames",
                    str(window),
                    "--delay-frames",
                    str(args.delay_frames),
                ])
                idf1, idtp, idfp, idfn, pred_ids, _ = _eval_global(
                    short_root, scene, out_dir)
            except Exception as exc:
                _log(f"  [fail] {scene}: {exc}")
                failed = True
                break

            scene_scores[scene] = idf1
            scene_pred_ids[scene] = pred_ids
            total_idtp += idtp
            total_idfp += idfp
            total_idfn += idfn
            _log(f"  {scene:18s} global={idf1:.4f} pred_ids={pred_ids}")

            if not args.keep_outputs:
                shutil.rmtree(out_dir, ignore_errors=True)

        if failed:
            continue

        micro = (2 * total_idtp) / max(1, 2 * total_idtp + total_idfp + total_idfn)
        mean = sum(scene_scores.values()) / max(1, len(scene_scores))
        row = {
            "tag": tag,
            "threshold": thr,
            "margin": margin,
            "geo_weight": geo_w,
            "geo_min_overlaps": geo_ov,
            "window_frames": window,
            "micro_global_idf1": micro,
            "mean_scene_global_idf1": mean,
            "total_idtp": total_idtp,
            "total_idfp": total_idfp,
            "total_idfn": total_idfn,
        }
        for scene, score in scene_scores.items():
            row[f"{scene}_global"] = score
            row[f"{scene}_pred_ids"] = scene_pred_ids[scene]
        rows.append(row)
        _log(f"  => micro={micro:.4f} mean={mean:.4f}")

    rows.sort(key=lambda r: r["micro_global_idf1"], reverse=True)
    csv_path = out_root / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _log("\n[sweep] TOP")
    for row in rows[:10]:
        _log(
            f"  {row['micro_global_idf1']:.4f} mean={row['mean_scene_global_idf1']:.4f} "
            f"tag={row['tag']}"
        )
    _log(f"[sweep] wrote {csv_path}")


if __name__ == "__main__":
    main()
