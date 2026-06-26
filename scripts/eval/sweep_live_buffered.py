"""Parameter sweep for the live_buffered cross-camera identity stage.

live_buffered (src/mtmc/live_buffered.py) is the module that ACTUALLY produces
the evaluated cross-camera global IDs (score_full_mmp_val.py runs it, not the
live CrossCameraGalleryProbe). This script tunes its real knobs on the FULL
24-scene MMP val set, operating directly on the already-exported det_emb_chunk
.npz files (CPU only — no GPU, no re-run, no training):

  * window_chunks   — how many recent chunks form one clustering window
  * assign_thr      — max (1-cosine) to stitch a window cluster to a global id
  * anchor_window   — sliding-window majority-vote length inside assign_per_frame
  * fixed_k         — fixed #identities (default: per-window concurrency floor)

It ALSO optionally applies a geometry/STCRA position post-pass (the mentor's
"position before appearance" idea), reconstructing per-detection foot world-XY
from the prediction bbox + MMP calibration and reassigning outlier detections to
the nearest identity's per-frame world centroid (src/eval/offline_anchor_faithful.stcra).

Baseline (matches score_full_mmp_val defaults): window_chunks=1, assign_thr=0.40,
anchor_window=15, no geometry  →  24-scene mean IDF1 ~0.774.

Usage:
    python scripts/eval/sweep_live_buffered.py --config baseline
    python scripts/eval/sweep_live_buffered.py --config assign_thr=0.45
    python scripts/eval/sweep_live_buffered.py --config "assign_thr=0.45,geo=1,geo_passes=1500:1000:750,geo_conf=0.65"
    python scripts/eval/sweep_live_buffered.py --grid     # run the built-in grid
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

CALIB_BASE = REPO / "dataset/MMPTracking/MMPTracking_validation/validation/calibrations"


# --------------------------------------------------------------------- config
@dataclass
class SweepCfg:
    name: str = "baseline"
    window_chunks: int = 1
    assign_thr: float = 0.40
    anchor_window: int = 15
    fixed_k: int | None = None
    # geometry / STCRA post-pass (REJECTED on MMP — reassignment over-merges)
    geo: bool = False
    geo_passes: tuple[float, ...] = (1500.0, 1000.0, 750.0)
    geo_conf: float = 0.65
    # geo-merge link-prior post-pass: merge two global ids that are consistently
    # co-located across cameras AND never co-present in one camera (overlapping-FOV
    # mutual exclusion). Fixes appearance fragmentation without merging distinct people.
    geomerge: bool = False
    gm_dist: float = 500.0      # mm: two views within this are "same world point"
    gm_overlaps: int = 12       # min co-observed frames to consider a pair
    gm_frac: float = 0.70       # fraction of co-observed frames that must be co-located
    # geo-split: split a global id that contains a PROVEN conflict (two tracklets
    # co-observed at the same frame but consistently far apart = two distinct people).
    # Targets retail's dominant error (similar-looking people falsely merged).
    geosplit: bool = False
    gs_far: float = 1200.0      # mm: co-observed but farther than this => conflict
    gs_near: float = 600.0      # mm: within this over co-observed frames => same person
    gs_overlaps: int = 10       # min co-observed frames for a conflict/same edge
    gs_frac: float = 0.60       # fraction of co-observed frames the relation must hold
    # static false-positive filter: drop local tracks that barely move and live long
    # (mannequins / posters / shelf clutter). Applied BEFORE clustering so it also
    # cleans the identity space. Retail FPs are static (motion~4px) vs people (~30px).
    fp_filter: bool = False
    fp_motion: float = 5.0      # px: center-position std below this = static
    fp_minframes: int = 100     # only drop static tracks living at least this long

    @classmethod
    def parse(cls, spec: str) -> "SweepCfg":
        c = cls()
        if spec in ("", "baseline"):
            c.name = "baseline"
            return c
        parts = [p for p in spec.split(",") if p]
        labels = []
        for p in parts:
            k, _, v = p.partition("=")
            k = k.strip()
            if k == "window_chunks":
                c.window_chunks = int(v)
            elif k == "assign_thr":
                c.assign_thr = float(v)
            elif k == "anchor_window":
                c.anchor_window = int(v)
            elif k == "fixed_k":
                c.fixed_k = int(v)
            elif k == "geo":
                c.geo = bool(int(v))
            elif k == "geo_passes":
                c.geo_passes = tuple(float(x) for x in v.split(":"))
            elif k == "geo_conf":
                c.geo_conf = float(v)
            elif k == "geomerge":
                c.geomerge = bool(int(v))
            elif k == "gm_dist":
                c.gm_dist = float(v)
            elif k == "gm_overlaps":
                c.gm_overlaps = int(v)
            elif k == "gm_frac":
                c.gm_frac = float(v)
            elif k == "geosplit":
                c.geosplit = bool(int(v))
            elif k == "gs_far":
                c.gs_far = float(v)
            elif k == "gs_near":
                c.gs_near = float(v)
            elif k == "gs_overlaps":
                c.gs_overlaps = int(v)
            elif k == "gs_frac":
                c.gs_frac = float(v)
            elif k == "fp_filter":
                c.fp_filter = bool(int(v))
            elif k == "fp_motion":
                c.fp_motion = float(v)
            elif k == "fp_minframes":
                c.fp_minframes = int(v)
            else:
                raise ValueError(f"unknown sweep key {k!r}")
            labels.append(p)
        c.name = ",".join(labels)
        return c


def _static_track_dropset(scene_dir: Path, cfg: SweepCfg) -> set[tuple[int, int]]:
    """Local tracks (src_cam, ltid) that barely move and live long = static FPs.

    src_cam here is the pipeline source_id (the cam_id used in the chunks and the
    pred CSV filename), so the drop-set keys match both clustering and scoring.
    """
    if not cfg.fp_filter:
        return set()
    drop: set[tuple[int, int]] = set()
    for p in sorted(scene_dir.glob("cam_*_predictions.csv")):
        src = int(p.stem.split("_")[1])
        df = pd.read_csv(p)
        cx = df["left"] + df["width"] / 2.0
        cy = df["top"] + df["height"] / 2.0
        g = pd.DataFrame({"ltid": df["local_track_id"], "cx": cx, "cy": cy})
        for ltid, grp in g.groupby("ltid"):
            if len(grp) < cfg.fp_minframes:
                continue
            motion = float(np.hypot(grp["cx"].std(), grp["cy"].std()))
            if motion < cfg.fp_motion:
                drop.add((src, int(ltid)))
    return drop


# --------------------------------------------------------- clustering (no I/O)
def _cluster_scene(scene_dir: Path, cfg: SweepCfg,
                   drop: set[tuple[int, int]] | None = None
                   ) -> list[tuple[int, int, int, int]]:
    """Replicates live_buffered.main(--once) accumulation WITHOUT file I/O.
    Returns list of (cam_id, frame_no, local_track_id, global_id).
    `drop` removes static-FP local tracks from the clustering input."""
    from src.mtmc.live_buffered import LiveBufferedMTMC, _load_chunk

    chunks = sorted(scene_dir.glob("det_emb_chunk_*.npz"))
    if not chunks:
        return []
    mtmc = LiveBufferedMTMC(cfg.window_chunks, cfg.assign_thr,
                            cfg.fixed_k, cfg.anchor_window)
    assignments: list[tuple[int, int, int, int]] = []
    for path in chunks:
        try:
            cam, frame, ltid, emb = _load_chunk(path)
        except Exception:
            continue
        if drop:
            keep = np.array([(int(c), int(t)) not in drop
                             for c, t in zip(cam, ltid)], dtype=bool)
            cam, frame, ltid, emb = cam[keep], frame[keep], ltid[keep], emb[keep]
            if len(cam) == 0:
                continue
        st = mtmc.process_chunk(cam, frame, ltid, emb)
        assignments.extend(st.get("assignments", []))
    return assignments


# ------------------------------------------------------- geometry post-pass
def _scene_env(scene: str) -> str:
    env = scene.removeprefix("64pm_").rsplit("_", 1)[0]
    return env


def _build_world(scene_dir: Path, scene: str,
                 assignments: list[tuple[int, int, int, int]]):
    """Reconstruct per-detection foot world-XY from pred bbox + calibration.

    Returns (world dict keyed by (cam,frame,ltid) -> np.array([X,Y]), valid_frac).
    cam in the assign rows is the pipeline source_id (0-based); calibration is
    keyed by 1-based CameraId == gt_cam_id, paired with source_id in sorted order
    exactly as score_full_mmp_val does.
    """
    from src.reid.geometry import GroundPlaneGeometry

    env = _scene_env(scene)
    cal_path = CALIB_BASE / env / "calibrations.json"
    if not cal_path.exists():
        return {}, 0.0
    geo = GroundPlaneGeometry(json.loads(cal_path.read_text()))

    # source_id -> gt_cam_id mapping (same sorted-zip as score_full)
    pred_files = sorted(scene_dir.glob("cam_*_predictions.csv"))
    source_ids = [int(p.stem.split("_")[1]) for p in pred_files]
    val_scene = REPO / "dataset/MMPTracking_10minute/val" / scene
    gt_cam_ids = sorted(int(p.stem[3:]) for p in val_scene.glob("cam*.mp4"))
    src2cam = dict(zip(source_ids, gt_cam_ids))

    # bbox lookup per (src, frame, ltid)
    bbox: dict[tuple[int, int, int], tuple[float, float, float, float]] = {}
    for p, src in zip(pred_files, source_ids):
        df = pd.read_csv(p)
        for r in df.itertuples():
            bbox[(src, int(r.frame_no_cam), int(r.local_track_id))] = (
                float(r.left), float(r.top), float(r.width), float(r.height))

    world: dict[tuple[int, int, int], np.ndarray] = {}
    n_total = 0
    for (cam, frame, ltid, _gid) in assignments:
        key = (cam, frame, ltid)
        b = bbox.get(key)
        if b is None:
            continue
        n_total += 1
        gt_cam = src2cam.get(cam)
        if gt_cam is None:
            continue
        fw = geo.bbox_foot(gt_cam, b[0], b[1], b[2], b[3])
        if fw is not None:
            world[key] = np.array([fw[0], fw[1]], dtype=np.float64)
    valid_frac = len(world) / max(1, n_total)
    return world, valid_frac


# ----------------------------------------------- geo-merge link-prior (c)
def geo_merge(det_gid: dict, world: dict, d_merge: float,
              min_overlaps: int, frac: float) -> dict:
    """Merge global-id pairs that are consistently co-located across cameras and
    never co-present in a single camera (overlapping-FOV mutual exclusion).

    This is a LINK prior, not a reassignment: it only unions ids that appearance
    clustering split but geometry says are one person seen from different cameras.
    """
    from collections import defaultdict

    # gid -> frame -> {cam: xy}
    gid_fc: dict = defaultdict(lambda: defaultdict(dict))
    gid_frames: dict = defaultdict(set)
    for (cam, frame, ltid), g in det_gid.items():
        xy = world.get((cam, frame, ltid))
        if xy is None:
            continue
        gid_fc[g][frame][cam] = xy
        gid_frames[g].add(frame)

    gids = sorted(gid_fc)
    parent = {g: g for g in gids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, g1 in enumerate(gids):
        f1 = gid_frames[g1]
        for g2 in gids[i + 1:]:
            common = f1 & gid_frames[g2]
            if len(common) < min_overlaps:
                continue
            blocked = False
            close = 0
            for f in common:
                c1, c2 = gid_fc[g1][f], gid_fc[g2][f]
                if set(c1) & set(c2):       # same camera sees both -> distinct people
                    blocked = True
                    break
                mind = min(float(np.linalg.norm(a - b))
                           for a in c1.values() for b in c2.values())
                if mind < d_merge:
                    close += 1
            if blocked:
                continue
            if close / len(common) >= frac:
                union(g1, g2)

    return {k: find(g) for k, g in det_gid.items()}


def geo_split(det_gid: dict, world: dict, far: float, near: float,
              min_overlaps: int, frac: float) -> dict:
    """Split a global id that contains a proven geometric conflict.

    Atomic unit = tracklet (cam, ltid). Two tracklets under one gid that are
    co-observed (same frames) but consistently FAR apart in world space are two
    distinct people. Only such conflicted gids are repartitioned (by single-linkage
    on co-location); gids with no conflict are left untouched, so sequential
    same-person tracklets that never co-observe are never fragmented.
    """
    from collections import defaultdict

    # tracklet (cam,ltid) -> gid, and tracklet -> {frame: xy}
    tl_gid: dict = {}
    tl_xy: dict = defaultdict(dict)
    for (cam, frame, ltid), g in det_gid.items():
        tl = (cam, ltid)
        tl_gid[tl] = g          # gid stable per tracklet (last wins; consistent in practice)
        xy = world.get((cam, frame, ltid))
        if xy is not None:
            tl_xy[tl][frame] = xy

    gid_tls: dict = defaultdict(list)
    for tl, g in tl_gid.items():
        gid_tls[g].append(tl)

    next_gid = max(det_gid.values()) + 1 if det_gid else 1
    remap: dict = {}     # tracklet -> new gid

    def relation(t1, t2):
        """Return ('same'|'conflict'|None) over co-observed frames."""
        a, b = tl_xy.get(t1, {}), tl_xy.get(t2, {})
        common = a.keys() & b.keys()
        if len(common) < min_overlaps:
            return None
        nf = ndist = 0
        for f in common:
            d = float(np.linalg.norm(a[f] - b[f]))
            if d <= near:
                nf += 1
            if d >= far:
                ndist += 1
        n = len(common)
        if ndist / n >= frac:
            return "conflict"
        if nf / n >= frac:
            return "same"
        return None

    for g, tls in gid_tls.items():
        if len(tls) < 2:
            for tl in tls:
                remap[tl] = g
            continue
        # detect any conflict within this gid
        conflict = False
        same_edges = []
        for i in range(len(tls)):
            for j in range(i + 1, len(tls)):
                rel = relation(tls[i], tls[j])
                if rel == "conflict":
                    conflict = True
                elif rel == "same":
                    same_edges.append((i, j))
        if not conflict:
            for tl in tls:
                remap[tl] = g          # untouched
            continue
        # repartition: single-linkage union on 'same' edges
        parent = list(range(len(tls)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in same_edges:
            parent[find(i)] = find(j)
        comp_gid: dict = {}
        nonlocal_next = next_gid
        for idx, tl in enumerate(tls):
            r = find(idx)
            if r not in comp_gid:
                # keep original gid for the first component, mint new for others
                comp_gid[r] = g if len(comp_gid) == 0 else nonlocal_next
                if len(comp_gid) > 1:
                    nonlocal_next += 1
            remap[tl] = comp_gid[r]
        next_gid = nonlocal_next

    return {(cam, frame, ltid): remap.get((cam, ltid), g)
            for (cam, frame, ltid), g in det_gid.items()}


# ----------------------------------------------------------------- scoring
def _build_eval_frames(scene_dir: Path, scene: str,
                       gid_map: dict[tuple[int, int, int], int]):
    pred_files = sorted(scene_dir.glob("cam_*_predictions.csv"))
    source_ids = [int(p.stem.split("_")[1]) for p in pred_files]
    val_scene = REPO / "dataset/MMPTracking_10minute/val" / scene
    gt_cam_ids = sorted(int(p.stem[3:]) for p in val_scene.glob("cam*.mp4"))

    all_gt, all_pred = {}, {}
    for src_id, gt_cam_id in zip(source_ids, gt_cam_ids):
        pred_path = scene_dir / f"cam_{src_id}_predictions.csv"
        gt_path = val_scene / f"gt_cam{gt_cam_id}_clean.csv"
        if not gt_path.exists():
            gt_path = val_scene / f"gt_cam{gt_cam_id}.csv"
        if not pred_path.exists() or not gt_path.exists():
            continue
        pred = pd.read_csv(pred_path).copy()
        gt = pd.read_csv(gt_path)
        pred["global_id"] = [
            gid_map.get((src_id, int(f), int(t)), -1)
            for f, t in zip(pred["frame_no_cam"], pred["local_track_id"])
        ]
        pred = pred[pred["global_id"] >= 0].copy()
        pred = pred.rename(columns={"frame_no_cam": "frame"})
        all_gt[gt_cam_id] = gt
        all_pred[gt_cam_id] = pred
    return all_gt, all_pred


def _score(scene_dir: Path, scene: str,
           gid_map: dict[tuple[int, int, int], int]) -> float | None:
    from src.eval.mmp_metrics.core import _eval_global_idf1
    all_gt, all_pred = _build_eval_frames(scene_dir, scene, gid_map)
    if not all_gt:
        return None
    result = _eval_global_idf1(all_gt, all_pred, iou_threshold=0.5)
    return result.get("idf1") or result.get("global_idf1") or result.get("mean_idf1")


def _per_camera(scene_dir: Path, scene: str,
                gid_map: dict[tuple[int, int, int], int],
                with_hota: bool) -> dict | None:
    """Env-aggregatable per-camera tracking metrics for this scene."""
    from src.eval.mmp_metrics.core import compute_per_camera_metrics
    all_gt, all_pred = _build_eval_frames(scene_dir, scene, gid_map)
    if not all_gt:
        return None
    rows = compute_per_camera_metrics(all_gt, all_pred, iou_threshold=0.5,
                                      pred_id_col="global_id", with_hota=with_hota)
    if not rows:
        return None
    keys = [k for k in rows[0] if k != "camera"]
    return {k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in keys}


def score_scene(scene_dir_str: str, scene: str, cfg: SweepCfg,
                with_mota: bool = False, with_hota: bool = False) -> dict:
    scene_dir = Path(scene_dir_str)
    drop = _static_track_dropset(scene_dir, cfg)
    assignments = _cluster_scene(scene_dir, cfg, drop=drop)
    if not assignments:
        return {"scene": scene, "idf1": None, "valid_frac": None}

    gid_map = {(c, f, t): g for (c, f, t, g) in assignments}

    valid_frac = None
    if cfg.geo or cfg.geomerge or cfg.geosplit:
        world, valid_frac = _build_world(scene_dir, scene, assignments)
        if world:
            det_gid = dict(gid_map)
            if cfg.geo:   # STCRA reassignment (rejected on MMP; kept for ablation)
                from src.eval.offline_anchor_faithful import stcra
                det_gid = stcra(det_gid, world, list(cfg.geo_passes),
                                cfg.geo_conf, min_overlap=8)
            if cfg.geosplit:   # split gids with a proven geometric conflict
                det_gid = geo_split(det_gid, world, cfg.gs_far, cfg.gs_near,
                                    cfg.gs_overlaps, cfg.gs_frac)
            if cfg.geomerge:   # link-prior co-location merge
                det_gid = geo_merge(det_gid, world, cfg.gm_dist,
                                    cfg.gm_overlaps, cfg.gm_frac)
            gid_map = det_gid

    out = {"scene": scene, "idf1": _score(scene_dir, scene, gid_map),
           "valid_frac": valid_frac}
    if with_mota:
        out["per_camera_mean"] = _per_camera(scene_dir, scene, gid_map, with_hota)
    return out


# -------------------------------------------------------------------- driver
def run_config(cfg: SweepCfg, export_root: Path, workers: int,
               scenes: list[str] | None,
               with_mota: bool = False, with_hota: bool = False) -> dict:
    scene_dirs = sorted(d for d in export_root.glob("64pm_*") if d.is_dir())
    if scenes:
        scene_dirs = [d for d in scene_dirs if d.name in scenes]

    results: dict[str, dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(score_scene, str(d), d.name, cfg,
                              with_mota, with_hota): d.name
                    for d in scene_dirs}
            for fut in as_completed(futs):
                r = fut.result()
                results[r["scene"]] = r
                v = r["idf1"]
                vf = r["valid_frac"]
                tag = f" geo_valid={vf:.2f}" if vf is not None else ""
                print(f"  [{r['scene']:34}] IDF1={v:.4f}{tag}" if v is not None
                      else f"  [{r['scene']:34}] IDF1=NA")
    else:
        for d in scene_dirs:
            r = score_scene(str(d), d.name, cfg, with_mota, with_hota)
            results[r["scene"]] = r
            v = r["idf1"]
            print(f"  [{d.name:34}] IDF1={v:.4f}" if v is not None
                  else f"  [{d.name:34}] IDF1=NA")

    # aggregate
    env_vals: dict[str, list[float]] = {}
    valid: list[float] = []
    for scene, r in results.items():
        if r["idf1"] is None:
            continue
        env = _scene_env(scene)
        env_vals.setdefault(env, []).append(r["idf1"])
        valid.append(r["idf1"])
    env_means = {e: round(sum(v) / len(v), 4) for e, v in env_vals.items()}
    overall = round(sum(valid) / len(valid), 4) if valid else None
    nonretail = [v for e, vs in env_vals.items() if e != "retail" for v in vs]
    nonretail_mean = round(sum(nonretail) / len(nonretail), 4) if nonretail else None

    summary = {
        "config": cfg.name,
        "params": {
            "window_chunks": cfg.window_chunks, "assign_thr": cfg.assign_thr,
            "anchor_window": cfg.anchor_window, "fixed_k": cfg.fixed_k,
            "geo": cfg.geo, "geo_passes": list(cfg.geo_passes),
            "geo_conf": cfg.geo_conf, "geomerge": cfg.geomerge,
            "gm_dist": cfg.gm_dist, "gm_overlaps": cfg.gm_overlaps,
            "gm_frac": cfg.gm_frac, "geosplit": cfg.geosplit,
            "gs_far": cfg.gs_far, "gs_near": cfg.gs_near,
            "gs_overlaps": cfg.gs_overlaps, "gs_frac": cfg.gs_frac,
            "fp_filter": cfg.fp_filter, "fp_motion": cfg.fp_motion,
            "fp_minframes": cfg.fp_minframes,
        },
        "overall_idf1": overall,
        "nonretail_idf1": nonretail_mean,
        "env_means": env_means,
        "per_scene": {s: r["idf1"] for s, r in sorted(results.items())},
    }

    # aggregate per-camera tracking metrics by env (mota/hota/switches/...)
    if with_mota:
        pc_keys = ["mota", "motp", "idf1", "num_switches", "num_fragmentations",
                   "precision", "recall", "hota", "deta", "assa"]
        env_pc: dict[str, dict[str, list[float]]] = {}
        for scene, r in results.items():
            pcm = r.get("per_camera_mean")
            if not pcm:
                continue
            env = _scene_env(scene)
            for k in pc_keys:
                if k in pcm:
                    env_pc.setdefault(env, {}).setdefault(k, []).append(pcm[k])
        summary["per_camera_by_env"] = {
            e: {k: round(sum(vs) / len(vs), 4) for k, vs in d.items()}
            for e, d in env_pc.items()
        }
        # overall mean across all scenes
        all_pc: dict[str, list[float]] = {}
        for r in results.values():
            pcm = r.get("per_camera_mean")
            if not pcm:
                continue
            for k in pc_keys:
                if k in pcm:
                    all_pc.setdefault(k, []).append(pcm[k])
        summary["per_camera_overall"] = {
            k: round(sum(vs) / len(vs), 4) for k, vs in all_pc.items()}

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", default="output/eval/full_mmp_val")
    ap.add_argument("--config", default="baseline",
                    help="comma sep key=val, e.g. 'assign_thr=0.45,anchor_window=25'")
    ap.add_argument("--grid", action="store_true", help="run built-in grid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenes", default=None,
                    help="comma sep scene names (subset) for fast iteration")
    ap.add_argument("--mota", action="store_true",
                    help="also compute per-camera MOTA/IDS/precision/recall, aggregated by env")
    ap.add_argument("--hota", action="store_true",
                    help="also compute per-camera HOTA/DetA/AssA (slow; implies --mota)")
    ap.add_argument("--out", default="output/eval/sweep_live_buffered/results.jsonl")
    args = ap.parse_args()
    with_hota = args.hota
    with_mota = args.mota or args.hota

    export_root = REPO / args.export_root
    scenes = args.scenes.split(",") if args.scenes else None
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.grid:
        specs = [
            "baseline",
            "assign_thr=0.30", "assign_thr=0.35", "assign_thr=0.45", "assign_thr=0.55",
            "anchor_window=9", "anchor_window=25",
        ]
    else:
        specs = [args.config]

    all_summaries = []
    for spec in specs:
        cfg = SweepCfg.parse(spec)
        print(f"\n{'='*64}\n CONFIG: {cfg.name}\n{'='*64}")
        summary = run_config(cfg, export_root, args.workers, scenes,
                             with_mota=with_mota, with_hota=with_hota)
        all_summaries.append(summary)
        print(f"\n  >> overall IDF1 = {summary['overall_idf1']}  "
              f"(non-retail {summary['nonretail_idf1']})")
        print(f"  >> env: {summary['env_means']}")
        if "per_camera_overall" in summary:
            print(f"  >> per-cam overall: {summary['per_camera_overall']}")
            print(f"  >> per-cam by env: {summary['per_camera_by_env']}")
        with open(out_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    print(f"\nAppended {len(all_summaries)} config(s) → {out_path}")


if __name__ == "__main__":
    main()
