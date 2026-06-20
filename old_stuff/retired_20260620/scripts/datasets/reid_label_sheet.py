"""Render per-environment ReID label sheets for assisted manual identity merge.

Each scene-track (a `(scene, person_id)` group, cross-camera-consistent within
one scene) is shown as ONE clear crop with a big index number. Tracks are
ordered by the auto-clustering proposal so look-alikes are adjacent and a colored
border marks the proposed group. You then tell me the true groupings by index,
e.g.   lobby A: 1 2 5   B: 3 7 9 ...

Inputs : the consolidated manifest (proposal) + the crop cache.
Outputs: output/reid_labelsheet/sheet_<env>.png   (one numbered crop per track)
         output/reid_labelsheet/index.csv          (idx -> scene, orig_id, env)

Run:
    python scripts/datasets/reid_label_sheet.py --env lobby
    python scripts/datasets/reid_label_sheet.py --all
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

CACHE = "dataset/MMPTracking_10minute_reid_cache"
PROPOSAL = "output/reid_consolidation/train_consolidated_manifest.csv"
OUT = "output/reid_labelsheet"


def env_of(scene: str) -> str:
    return "_".join(scene.split("_")[1:-1])


def top_crops(rel_paths: list[str], n: int = 3) -> list[np.ndarray]:
    """Pick n clear crops spread across time: split frames into n chunks, take
    the largest-area crop from each chunk (avoids n near-duplicate frames and
    biases away from tiny/occluded crops)."""
    paths = sorted(rel_paths)  # filename encodes frame -> temporal order
    chunks = np.array_split(paths, n) if len(paths) >= n else [paths]
    out = []
    for ch in chunks:
        best, ba = None, -1
        # sample up to ~10 per chunk for speed
        sub = list(ch)[:: max(1, len(ch) // 10)] or list(ch)
        for p in sub:
            im = cv2.imread(os.path.join(CACHE, p))
            if im is not None and im.shape[0] * im.shape[1] > ba:
                ba, best = im.shape[0] * im.shape[1], im
        if best is not None:
            out.append(best)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="Single environment, e.g. lobby")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cw", type=int, default=58)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--per-track", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = list(csv.DictReader(open(PROPOSAL)))
    # group crops by scene-track (scene, orig_pid); keep proposed gid for ordering
    track_paths: dict[tuple, list[str]] = defaultdict(list)
    track_gid: dict[tuple, int] = {}
    for r in rows:
        key = (r["scene"], r["orig_pid"])
        track_paths[key].append(r["rel_path"])
        track_gid[key] = int(r["gid"])

    envs = sorted({env_of(s) for s, _ in track_paths}) if args.all else [args.env]
    index_rows = []
    for env in envs:
        tracks = sorted([k for k in track_paths if env_of(k[0]) == env],
                        key=lambda k: (track_gid[k], k[0], int(k[1])))
        CW, CH, COLS = args.cw, args.ch, args.cols
        cells = []
        # color per proposed gid
        gids = sorted({track_gid[k] for k in tracks})
        palette = {g: tuple(int(c) for c in np.random.RandomState(g * 13 + 1).randint(70, 255, 3))
                   for g in gids}
        for idx, k in enumerate(tracks, 1):
            scene, op = k
            imgs = top_crops(track_paths[k], args.per_track)
            while len(imgs) < args.per_track:
                imgs.append(np.zeros((CH, CW, 3), np.uint8))
            strip = np.hstack([cv2.resize(im, (CW, CH)) for im in imgs[:args.per_track]])
            bc = palette[track_gid[k]]
            # header (index) + footer (scene/id) bars, then color border
            header = np.zeros((20, strip.shape[1], 3), np.uint8)
            cv2.putText(header, f"#{idx}", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(header, f"{scene.split('_')[0]} id{op}", (44, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
            cell = np.vstack([header, strip])
            cell = cv2.copyMakeBorder(cell, 3, 5, 4, 4, cv2.BORDER_CONSTANT, value=bc)
            cells.append(cell)
            index_rows.append({"env": env, "idx": idx, "scene": scene, "orig_id": op,
                               "proposed_gid": track_gid[k]})
        # tile
        ch, cw = cells[0].shape[:2]
        nrow = (len(cells) + COLS - 1) // COLS
        sheet = np.full((nrow * ch, COLS * cw, 3), 30, np.uint8)
        for i, c in enumerate(cells):
            r, cc = divmod(i, COLS)
            sheet[r * ch:(r + 1) * ch, cc * cw:(cc + 1) * cw] = c
        out = f"{OUT}/sheet_{env}.png"
        cv2.imwrite(out, sheet)
        print(f"[{env}] {len(tracks)} scene-tracks (proposed {len(gids)} groups, color-bordered) -> {out}")

    with open(f"{OUT}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "idx", "scene", "orig_id", "proposed_gid"])
        w.writeheader(); w.writerows(index_rows)
    print(f"[done] index -> {OUT}/index.csv")


if __name__ == "__main__":
    main()
