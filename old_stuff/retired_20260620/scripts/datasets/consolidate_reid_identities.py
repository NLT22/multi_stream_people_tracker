"""Recover env-global ReID identities from scene-local person_id labels.

MMPTracking `person_id` is scene-local (re-numbered every scene), so the crop
cache keys identity on `(scene, person_id)` and shatters each real person into
one label per scene they appear in (~196 labels for ~50 real people). This
poisons triplet supervision (same person pushed apart across scenes).

This tool re-links scene-tracks into env-global identities:
  1. embed each scene-track (mean deployed-ReID embedding over its cached crops)
  2. cluster within each environment via constrained agglomerative clustering
     with a hard CANNOT-LINK constraint (two ids from the SAME scene are always
     different people)
  3. write a consolidated manifest (adds a `gid` column) + optional montages.

Run:
    python scripts/datasets/consolidate_reid_identities.py \
        --cache-root dataset/MMPTracking_10minute_reid_cache --split train \
        --reid-onnx models/reid/swin_tiny_mmp_reid_all.onnx \
        --threshold 0.45 --make-montages
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def env_of(scene: str) -> str:
    """'63am_lobby_0' -> 'lobby'; '63am_industry_safety_0' -> 'industry_safety'."""
    parts = scene.split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else scene


def load_manifest(cache_root: Path, split: str) -> list[dict]:
    rows = []
    with (cache_root / split / "manifest.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def embed_tracks(cache_root: Path, rows: list[dict], onnx: str,
                 crops_per_track: int, batch: int) -> dict[int, np.ndarray]:
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx, providers=providers)
    inp = sess.get_inputs()[0].name
    print(f"[reid] {onnx}  providers={sess.get_providers()}")
    H, W = 256, 128
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    by_track: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        by_track[int(r["pid"])].append(r["rel_path"])

    # subsample crops per track (evenly across its frames)
    paths, owner = [], []
    for pid, ps in by_track.items():
        step = max(1, len(ps) // crops_per_track)
        for p in ps[::step][:crops_per_track]:
            paths.append(cache_root / p)
            owner.append(pid)

    feats = np.zeros((len(paths), 256), np.float32)
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        arr = np.zeros((len(chunk), 3, H, W), np.float32)
        for j, p in enumerate(chunk):
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr[j] = ((im - mean) / std).transpose(2, 0, 1)
        f = sess.run(None, {inp: arr})[0]
        feats[i:i + len(chunk)] = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        print(f"  embedded {min(i + batch, len(paths))}/{len(paths)} crops", end="\r")
    print()

    track_emb: dict[int, np.ndarray] = {}
    owner = np.array(owner)
    for pid in by_track:
        m = feats[owner == pid].mean(0)
        track_emb[pid] = m / (np.linalg.norm(m) + 1e-9)
    return track_emb


def constrained_agglomerative(tracks: list[int], emb: dict[int, np.ndarray],
                              track_scene: dict[int, str],
                              threshold: float, mutual: bool = False) -> dict[int, int]:
    """Average-link agglomerative with a same-scene cannot-link constraint.
    If mutual=True, only merge a pair that are each other's best valid partner
    (mutual nearest neighbour) — structurally prevents a look-alike 'hub' from
    absorbing several distinct people. Returns {track_id: cluster_id}."""
    clu = {t: {"tracks": [t], "scenes": {track_scene[t]}, "emb": emb[t].copy()}
           for t in tracks}
    nxt = max(tracks) + 1 if tracks else 0

    def best_partner(a):
        bp, bs = None, threshold
        for c in clu:
            if c == a or (clu[a]["scenes"] & clu[c]["scenes"]):
                continue
            s = float(clu[a]["emb"] @ clu[c]["emb"])
            if s > bs:
                bs, bp = s, c
        return bp, bs

    while True:
        best, best_s = None, threshold
        ids = list(clu)
        for a, b in combinations(ids, 2):
            if clu[a]["scenes"] & clu[b]["scenes"]:
                continue  # cannot-link: share a scene
            s = float(clu[a]["emb"] @ clu[b]["emb"])
            if s > best_s:
                best_s, best = s, (a, b)
        if best is None:
            break
        if mutual:
            a, b = best
            # require a<->b to be each other's best valid partner
            if best_partner(a)[0] != b or best_partner(b)[0] != a:
                # demote this pair: temporarily block by scanning next-best overall
                # (simple approach: rebuild best among mutual pairs)
                mbest, mbs = None, threshold
                for x, y in combinations(ids, 2):
                    if clu[x]["scenes"] & clu[y]["scenes"]:
                        continue
                    s = float(clu[x]["emb"] @ clu[y]["emb"])
                    if s > mbs and best_partner(x)[0] == y and best_partner(y)[0] == x:
                        mbs, mbest = s, (x, y)
                if mbest is None:
                    break
                best = mbest
        a, b = best
        na, nb = len(clu[a]["tracks"]), len(clu[b]["tracks"])
        m = (clu[a]["emb"] * na + clu[b]["emb"] * nb) / (na + nb)
        clu[nxt] = {"tracks": clu[a]["tracks"] + clu[b]["tracks"],
                    "scenes": clu[a]["scenes"] | clu[b]["scenes"],
                    "emb": m / (np.linalg.norm(m) + 1e-9)}
        del clu[a], clu[b]
        nxt += 1
    out = {}
    for cid, c in clu.items():
        for t in c["tracks"]:
            out[t] = cid
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path("dataset/MMPTracking_10minute_reid_cache"))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--reid-onnx", default="models/reid/swin_tiny_mmp_reid_all.onnx")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--crops-per-track", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("output/reid_consolidation"))
    ap.add_argument("--mutual", action="store_true",
                    help="Only merge mutual nearest-neighbour pairs (safer; fewer over-merges).")
    ap.add_argument("--reembed", action="store_true",
                    help="Force re-running the ReID instead of loading cached embeddings.")
    ap.add_argument("--make-montages", action="store_true")
    ap.add_argument("--montage-envs", nargs="+", default=["lobby", "office", "cafe_shop"])
    ap.add_argument("--envs", nargs="+", default=None,
                    help="Only (re)cluster these environments (e.g. 'retail'); embed "
                         "just their tracks and MERGE into the existing proposal, "
                         "preserving other envs' rows. Omit to cluster everything.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.cache_root, args.split)
    track_scene, track_env = {}, {}
    for r in rows:
        pid = int(r["pid"])
        track_scene[pid] = r["scene"]
        track_env[pid] = env_of(r["scene"])
    print(f"[{args.split}] {len(rows)} crops, {len(track_scene)} scene-tracks, "
          f"{len(set(track_env.values()))} environments")

    target_envs = set(args.envs) if args.envs else None
    out_manifest = args.out_dir / f"{args.split}_consolidated_manifest.csv"

    # When restricting to a subset of envs, embed only those tracks (fast) and
    # carry forward the existing proposal's rows for all other envs.
    embed_rows = rows if target_envs is None else \
        [r for r in rows if track_env[int(r["pid"])] in target_envs]

    # cache track embeddings so threshold sweeps don't re-run the ReID
    emb_cache = args.out_dir / f"track_emb_{args.split}.npz"
    if target_envs is None and emb_cache.exists() and not args.reembed:
        z = np.load(emb_cache)
        track_emb = {int(k): z[k] for k in z.files}
        print(f"[reid] loaded cached embeddings: {emb_cache} ({len(track_emb)} tracks)")
    else:
        track_emb = embed_tracks(args.cache_root, embed_rows, args.reid_onnx,
                                 args.crops_per_track, args.batch)
        if target_envs is None:
            np.savez(emb_cache, **{str(k): v for k, v in track_emb.items()})
            print(f"[reid] cached embeddings -> {emb_cache}")
        else:
            print(f"[reid] embedded {len(track_emb)} tracks for envs={sorted(target_envs)}")

    # cluster per environment; assign globally-unique consolidated gids
    env_tracks: dict[str, list[int]] = defaultdict(list)
    for t, e in track_env.items():
        if target_envs is None or e in target_envs:
            env_tracks[e].append(t)

    # merge mode: keep existing proposal rows for envs we're NOT reclustering,
    # and start new gids after the highest gid already used by those rows.
    kept_rows: list[dict] = []
    gid_base = 0
    if target_envs is not None and out_manifest.exists():
        with out_manifest.open() as f:
            for r in csv.DictReader(f):
                if env_of(r["scene"]) in target_envs:
                    continue
                kept_rows.append(r)
                gid_base = max(gid_base, int(r["gid"]) + 1)
        print(f"[merge] keeping {len(kept_rows)} rows from {gid_base} existing identities")

    track_to_gid: dict[int, int] = {}
    gid_start = gid_base
    print(f"\n=== consolidation (threshold={args.threshold}) ===")
    for env in sorted(env_tracks):
        ts = env_tracks[env]
        local = constrained_agglomerative(ts, track_emb, track_scene, args.threshold,
                                          mutual=args.mutual)
        clusters = sorted(set(local.values()))
        remap = {c: gid_base + i for i, c in enumerate(clusters)}
        for t in ts:
            track_to_gid[t] = remap[local[t]]
        sizes = defaultdict(int)
        for t in ts:
            sizes[local[t]] += 1
        merged = sum(1 for c in clusters if sizes[c] > 1)
        print(f"  {env:18s}: {len(ts):3d} scene-tracks -> {len(clusters):3d} identities "
              f"({merged} are multi-scene merges)")
        gid_base += len(clusters)
    print(f"  TOTAL: {len(track_to_gid)} scene-tracks -> "
          f"{gid_base - gid_start} new identities (gids {gid_start}..{gid_base - 1})")

    # write consolidated manifest (kept rows + freshly-clustered target rows)
    fields = ["rel_path", "gid", "orig_pid", "cam_id", "scene", "frame"]
    with out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept_rows:
            w.writerow({k: r[k] for k in fields})
        for r in embed_rows:
            w.writerow({"rel_path": r["rel_path"], "gid": track_to_gid[int(r["pid"])],
                        "orig_pid": r["pid"], "cam_id": r["cam_id"],
                        "scene": r["scene"], "frame": r["frame"]})
    print(f"[done] consolidated manifest -> {out_manifest}")

    if args.make_montages:
        render_montages(args, embed_rows, track_to_gid, track_scene, track_env)


def render_montages(args, rows, track_to_gid, track_scene, track_env) -> None:
    by_track_paths = defaultdict(list)
    for r in rows:
        by_track_paths[int(r["pid"])].append(r["rel_path"])
    gid_tracks = defaultdict(list)
    for t, g in track_to_gid.items():
        gid_tracks[g].append(t)
    CW, CH = 80, 160
    for env in args.montage_envs:
        # pick multi-track clusters in this env
        clusters = [(g, ts) for g, ts in gid_tracks.items()
                    if track_env.get(ts[0]) == env and len(ts) >= 2]
        clusters = sorted(clusters, key=lambda x: -len(x[1]))[:7]
        if not clusters:
            continue
        ncol = min(6, max(len(ts) for _, ts in clusters))
        rowimgs = []
        for g, ts in clusters:
            cells = []
            for t in ts[:ncol]:
                p = args.cache_root / by_track_paths[t][len(by_track_paths[t]) // 2]
                im = cv2.imread(str(p))
                im = cv2.resize(im, (CW, CH)) if im is not None else np.zeros((CH, CW, 3), np.uint8)
                cv2.putText(im, f"{track_scene[t].replace('63am_','').replace('64am_','')}", (1, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.putText(im, f"id{int(by_track_paths[t][0].split('cam')[1].split('_')[0]) if False else ''}", (1, CH-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255),1)
                cells.append(im)
            while len(cells) < ncol:
                cells.append(np.zeros((CH, CW, 3), np.uint8))
            strip = np.hstack(cells)
            cv2.putText(strip, f"GID{g}", (1, CH//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            rowimgs.append(strip)
        out = args.out_dir / f"{args.split}_montage_{env}.png"
        cv2.imwrite(str(out), np.vstack(rowimgs))
        print(f"[montage] {out}  ({len(clusters)} multi-scene identities; each ROW should be ONE person across scenes)")


if __name__ == "__main__":
    main()
