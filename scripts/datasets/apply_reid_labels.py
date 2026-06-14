"""Apply manual reid_labels/*.json -> a cleanly-labeled training crop cache.

Turns the verified manual groupings into ReID training identities so the triplet
loss sees correct supervision (fixing the scene-local over-fragmentation that the
197-label cache caused). Identity = (environment, manual group); 'JUNK' tracks
dropped. Writes a drop-in cache that reuses the original crops (rel_path
back-pointed), so train with --crop-cache-root <out-dir>.

  train: relabeled from reid_labels/  (the manually-verified identities)
  val:   kept as-is (held-out monitor; not manually labeled)

Run:
    python scripts/datasets/apply_reid_labels.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter


def env_of(scene: str) -> str:
    return "_".join(scene.split("_")[1:-1])


def identity(track_key: str, group: str) -> str:
    scene = track_key.split("|")[0]
    return f"{env_of(scene)}::{group}"   # namespaced per env (groups are per-env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="reid_labels")
    ap.add_argument("--cache-root", default="dataset/MMPTracking_10minute_reid_cache")
    ap.add_argument("--out-dir", default="dataset/MMPTracking_10minute_reid_cache_labeled")
    args = ap.parse_args()

    labels: dict[str, str] = {}
    for f in sorted(glob.glob(f"{args.labels_dir}/labels_*.json")):
        labels.update(json.load(open(f)))
    ids = sorted({identity(k, v) for k, v in labels.items() if v != "JUNK"})
    id2pid = {ident: i for i, ident in enumerate(ids)}
    per_env = Counter(i.split("::")[0] for i in ids)
    print(f"[apply] {len(labels)} labeled tracks -> {len(ids)} identities "
          f"(JUNK dropped); per env: {dict(per_env)}")

    rel_prefix = f"../{os.path.basename(args.cache_root)}/"
    fields = ["rel_path", "pid", "cam_id", "scene", "frame"]

    # train: relabel by manual identity (drop JUNK / unlabeled)
    os.makedirs(f"{args.out_dir}/train", exist_ok=True)
    kept = dropped = 0
    with open(f"{args.cache_root}/train/manifest.csv") as fin, \
         open(f"{args.out_dir}/train/manifest.csv", "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()
        for r in csv.DictReader(fin):
            grp = labels.get(f"{r['scene']}|{r['pid']}")
            if grp is None or grp == "JUNK":
                dropped += 1
                continue
            w.writerow({"rel_path": rel_prefix + r["rel_path"],
                        "pid": id2pid[identity(f"{r['scene']}|{r['pid']}", grp)],
                        "cam_id": r["cam_id"], "scene": r["scene"], "frame": r["frame"]})
            kept += 1
    print(f"[apply] train crops: {kept} kept, {dropped} dropped (JUNK/unlabeled)")

    # val: kept as-is (held-out monitor, not manually labeled). Skip if the
    # source cache was built train-only (e.g. --splits train) — fine when the
    # trainer uses --crop-cache-val-from-train.
    val_src = f"{args.cache_root}/val/manifest.csv"
    if not os.path.exists(val_src):
        print(f"[apply] val: skipped (no {val_src}; use --crop-cache-val-from-train)")
    else:
        os.makedirs(f"{args.out_dir}/val", exist_ok=True)
        nval = 0
        with open(val_src) as fin, \
             open(f"{args.out_dir}/val/manifest.csv", "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()
            for r in csv.DictReader(fin):
                w.writerow({"rel_path": rel_prefix + r["rel_path"], "pid": r["pid"],
                            "cam_id": r["cam_id"], "scene": r["scene"], "frame": r["frame"]})
                nval += 1
        print(f"[apply] val crops: {nval} (scene-local, held-out monitor)")
    print(f"[apply] labeled cache -> {args.out_dir}")
    print("[apply] retrain with:")
    print(f"  python scripts/train/finetune_reid_mmp.py --resume output/reid_mmp_all/best.pth \\")
    print(f"    --crop-cache-root {args.out_dir} --output output/reid_10min_labeled \\")
    print(f"    --epochs 80 --early-stop 12 --pk-p 16 --pk-k 4 --accum-steps 2 --batches-per-epoch 400")


if __name__ == "__main__":
    main()
