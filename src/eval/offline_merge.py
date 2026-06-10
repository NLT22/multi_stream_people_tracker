"""Offline / nearline global-ID merge for exported MTMC predictions.

The realtime gallery intentionally stays conservative. This script runs after
prediction export and merges duplicate global IDs using full-tracklet evidence:

    python -m src.eval.offline_merge \
        --pred-dir output/eval/mta_swin_mta_no_merge_thr068 \
        --out-dir output/eval/mta_swin_mta_thr068_offline_merge

It requires files produced by PredictionExporter:

    tracklets.csv
    tracklet_embeddings.npz
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("[offline merge] numpy not found. Install: pip install numpy")


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {v: v for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        keep = min(ra, rb)
        drop = max(ra, rb)
        self.parent[drop] = keep
        return keep


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge duplicate global IDs after prediction export")
    p.add_argument("--pred-dir", required=True,
                   help="Prediction directory containing cam_* CSVs, "
                        "tracklets.csv, and tracklet_embeddings.npz")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for remapped predictions")
    p.add_argument("--threshold", type=float, default=0.82,
                   help="Minimum cosine similarity between global-ID embeddings")
    p.add_argument("--margin", type=float, default=0.05,
                   help="Best merge candidate must beat runner-up by this much")
    p.add_argument("--min-gid-embeddings", type=int, default=12,
                   help="Minimum sampled embeddings for each global ID")
    p.add_argument("--min-tracklet-detections", type=int, default=20,
                   help="Ignore very short tracklet segments for embedding means")
    p.add_argument("--max-candidates-per-gid", type=int, default=5,
                   help="Bound candidate pairs considered per global ID")
    p.add_argument("--temporal-tolerance", type=int, default=0,
                   help="Same-camera frame-overlap tolerance before IDs conflict")
    p.add_argument("--mmp-short-root", default=None,
                   help="Enable geometry-assisted merge using MMPTracking_short root")
    p.add_argument("--scene", default=None,
                   help="MMPTracking_short scene name for geometry-assisted merge")
    p.add_argument("--geo-weight", type=float, default=0.0,
                   help="Blend weight for ground-plane geometry score. "
                        "0 = embedding only, e.g. 0.25 for MMP lobby.")
    p.add_argument("--geo-sample-step", type=int, default=5,
                   help="Sample every N frames when computing geometry score")
    p.add_argument("--geo-min-overlaps", type=int, default=20,
                   help="Minimum cross-camera overlapping samples for geometry score")
    p.add_argument("--dry-run", action="store_true",
                   help="Print merge summary without writing remapped CSVs")
    return p.parse_args()


def _load_tracklets(pred_dir: Path) -> tuple[list[dict], dict[int, np.ndarray]]:
    tracklet_path = pred_dir / "tracklets.csv"
    embedding_path = pred_dir / "tracklet_embeddings.npz"
    if not tracklet_path.exists():
        raise FileNotFoundError(
            f"Missing {tracklet_path}. Re-run the pipeline after the "
            "tracklet exporter change.")
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing {embedding_path}. Re-run the pipeline with ReID tensors "
            "available so tracklet embeddings can be exported.")

    with open(tracklet_path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                "tracklet_id": int(row["tracklet_id"]),
                "cam_id": int(row["cam_id"]),
                "local_track_id": int(row["local_track_id"]),
                "global_id": int(row["global_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "num_detections": int(row["num_detections"]),
                "num_embeddings": int(row["num_embeddings"]),
            })

    data = np.load(embedding_path)
    tracklet_ids = data["tracklet_ids"].astype(np.int64)
    embeddings = data["embeddings"].astype(np.float32)
    emb_by_tracklet = {
        int(tracklet_id): embeddings[i]
        for i, tracklet_id in enumerate(tracklet_ids)
    }
    return rows, emb_by_tracklet


def _build_gid_summaries(
    tracklets: list[dict],
    emb_by_tracklet: dict[int, np.ndarray],
    min_gid_embeddings: int,
    min_tracklet_detections: int,
) -> tuple[list[int], np.ndarray, dict[int, list[tuple[int, int, int]]]]:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    intervals: dict[int, list[tuple[int, int, int]]] = {}

    for row in tracklets:
        gid = row["global_id"]
        if gid < 0:
            continue
        intervals.setdefault(gid, []).append(
            (row["cam_id"], row["start_frame"], row["end_frame"]))

        if row["num_detections"] < min_tracklet_detections:
            continue
        emb = emb_by_tracklet.get(row["tracklet_id"])
        if emb is None:
            continue
        weight = max(1, row["num_embeddings"])
        sums[gid] = sums.get(gid, np.zeros_like(emb)) + emb * weight
        counts[gid] = counts.get(gid, 0) + weight

    gids = []
    vectors = []
    for gid, emb_sum in sums.items():
        if counts.get(gid, 0) < min_gid_embeddings:
            continue
        norm = np.linalg.norm(emb_sum)
        if norm == 0.0:
            continue
        gids.append(gid)
        vectors.append((emb_sum / norm).astype(np.float32))

    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float32), intervals
    order = np.argsort(np.asarray(gids))
    gids = [gids[i] for i in order]
    vectors = np.stack([vectors[i] for i in order]).astype(np.float32)
    return gids, vectors, intervals


def _intervals_conflict(
    intervals_a: list[tuple[int, int, int]],
    intervals_b: list[tuple[int, int, int]],
    tolerance: int,
) -> bool:
    by_cam_b: dict[int, list[tuple[int, int]]] = {}
    for cam, start, end in intervals_b:
        by_cam_b.setdefault(cam, []).append((start, end))

    for cam, start_a, end_a in intervals_a:
        for start_b, end_b in by_cam_b.get(cam, []):
            if max(start_a, start_b) <= min(end_a, end_b) + tolerance:
                return True
    return False


def _load_geometry_points(
    pred_dir: Path,
    short_root: str | None,
    scene: str | None,
    sample_step: int,
) -> dict[int, dict[int, list[tuple[int, float, float]]]]:
    """Return gid -> frame -> [(cam_id, world_x, world_y), ...]."""
    exported = _load_exported_bev_points(pred_dir, sample_step)
    if exported:
        return exported

    if not short_root or not scene:
        return {}

    try:
        from src.dataset.mmp_tracking import MMPTrackingShortDataset
        from src.reid.geometry import GroundPlaneGeometry
    except ImportError as e:
        print(f"[offline merge] geometry disabled: {e}")
        return {}

    ds = MMPTrackingShortDataset(short_root, scene)
    geometry = GroundPlaneGeometry(ds.load_calibration())
    points: dict[int, dict[int, list[tuple[int, float, float]]]] = {}
    step = max(1, sample_step)

    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        cam_id = int(path.stem.split("_")[1])
        real_cam_id = cam_id + 1
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                gid = int(float(row["global_id"]))
                if gid < 0:
                    continue
                frame = int(float(row["frame_no_cam"]))
                if frame % step != 0:
                    continue
                foot = geometry.bbox_foot(
                    real_cam_id,
                    float(row["left"]),
                    float(row["top"]),
                    float(row["width"]),
                    float(row["height"]),
                )
                if foot is None:
                    continue
                points.setdefault(gid, {}).setdefault(frame, []).append(
                    (cam_id, foot[0], foot[1]))
    return points


def _load_exported_bev_points(
    pred_dir: Path,
    sample_step: int,
) -> dict[int, dict[int, list[tuple[int, float, float]]]]:
    """Load BEV/world trajectory samples exported by PredictionExporter.

    This is preferred over reconstructing foot points from cam_* CSVs because it
    preserves the live pipeline's actual calibrated foot projection. Older
    prediction directories do not have this file, so callers fall back to
    calibration-based reconstruction.
    """
    path = pred_dir / "tracklet_bev.csv"
    if not path.exists():
        return {}

    step = max(1, sample_step)
    points: dict[int, dict[int, list[tuple[int, float, float]]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = int(float(row["global_id"]))
            if gid < 0:
                continue
            frame = int(float(row["frame_no_cam"]))
            if frame % step != 0:
                continue
            cam_id = int(float(row["cam_id"]))
            points.setdefault(gid, {}).setdefault(frame, []).append((
                cam_id,
                float(row["world_x"]),
                float(row["world_y"]),
            ))
    return points


def _geometry_pair_score(
    gid_a: int,
    gid_b: int,
    points: dict[int, dict[int, list[tuple[int, float, float]]]],
    min_overlaps: int,
) -> tuple[float, int]:
    frames = set(points.get(gid_a, {})) & set(points.get(gid_b, {}))
    scores = []
    for frame in frames:
        for cam_a, xa, ya in points[gid_a][frame]:
            for cam_b, xb, yb in points[gid_b][frame]:
                if cam_a == cam_b:
                    continue
                dist = float(np.hypot(xa - xb, ya - yb))
                t = max(0.0, (dist - 300.0) / 1700.0)
                scores.append(float(np.exp(-3.0 * t * t)))
    if len(scores) < min_overlaps:
        return 0.0, len(scores)
    return float(np.median(scores)), len(scores)


# --- Ground-plane TRAJECTORY matching -----------------------------------------
# Beyond instantaneous co-location, compare the whole floor path of two Global
# IDs. Two signals:
#   1. Co-visible overlap + VELOCITY consistency — when both IDs are seen at the
#      same frames (different cameras), require not only that their feet coincide
#      but that they MOVE the same way. This rejects two different people who
#      momentarily stand near each other (same spot, different motion).
#   2. Hand-off prediction — when the two IDs are temporally adjacent (one leaves
#      a camera, the other appears shortly after), use constant-velocity
#      prediction on the floor to test whether one is the continuation of the
#      other. This links the same person across NON-overlapping camera views,
#      which the per-frame co-location score (which needs shared frames) cannot.
_GEO_CLOSE_MM = 300.0
_GEO_FAR_MM = 2000.0


def _decay(dist: float) -> float:
    t = max(0.0, (dist - _GEO_CLOSE_MM) / (_GEO_FAR_MM - _GEO_CLOSE_MM))
    return float(np.exp(-3.0 * t * t))


def _build_gid_trajectories(
    points: dict[int, dict[int, list[tuple[int, float, float]]]],
) -> dict[int, list[tuple[int, float, float]]]:
    """gid -> sorted [(frame, x, y)] using one averaged world position per frame."""
    traj: dict[int, list[tuple[int, float, float]]] = {}
    for gid, frames in points.items():
        seq = []
        for frame, obs in frames.items():
            xs = [o[1] for o in obs]
            ys = [o[2] for o in obs]
            seq.append((int(frame), float(np.mean(xs)), float(np.mean(ys))))
        seq.sort(key=lambda r: r[0])
        traj[gid] = seq
    return traj


def _seq_velocity(seq: list[tuple[int, float, float]], window: int,
                  at_end: bool) -> tuple[float, float]:
    if len(seq) < 2:
        return 0.0, 0.0
    seg = seq[-window:] if at_end else seq[:window]
    f0, x0, y0 = seg[0]
    f1, x1, y1 = seg[-1]
    df = f1 - f0
    if df <= 0:
        return 0.0, 0.0
    return (x1 - x0) / df, (y1 - y0) / df


def _trajectory_pair_score(
    gid_a: int,
    gid_b: int,
    traj: dict[int, list[tuple[int, float, float]]],
    min_overlaps: int,
    max_gap: int = 125,
    vel_window: int = 8,
) -> float:
    a = traj.get(gid_a, [])
    b = traj.get(gid_b, [])
    if len(a) < 2 or len(b) < 2:
        return 0.0

    # 1. Co-visible overlap + velocity consistency.
    b_at = {f: (x, y) for f, x, y in b}
    co = [
        _decay(float(np.hypot(x - b_at[f][0], y - b_at[f][1])))
        for f, x, y in a if f in b_at
    ]
    if len(co) >= min_overlaps:
        co_score = float(np.median(co))
        va = _seq_velocity(a, vel_window, at_end=True)
        vb = _seq_velocity(b, vel_window, at_end=True)
        na = float(np.hypot(*va))
        nb = float(np.hypot(*vb))
        if na > 1e-3 and nb > 1e-3:
            cosv = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
            vel_factor = 0.5 + 0.5 * max(0.0, cosv)   # 0.5 (opposed) .. 1.0 (aligned)
        else:
            vel_factor = 1.0                          # near-stationary: don't penalize
        return co_score * vel_factor

    # 2. Hand-off across a temporal gap (non-overlapping cameras).
    if a[-1][0] <= b[0][0]:
        first, second = a, b
    elif b[-1][0] <= a[0][0]:
        first, second = b, a
    else:
        return 0.0   # overlapping in time but not co-located -> different people
    gap = second[0][0] - first[-1][0]
    if gap < 0 or gap > max_gap:
        return 0.0
    vx, vy = _seq_velocity(first, vel_window, at_end=True)
    px = first[-1][1] + vx * gap
    py = first[-1][2] + vy * gap
    d = float(np.hypot(px - second[0][1], py - second[0][2]))
    return _decay(d) * 0.85   # discount a predicted match vs an observed one


def _component_conflict(
    root_a: int,
    root_b: int,
    members: dict[int, set[int]],
    intervals: dict[int, list[tuple[int, int, int]]],
    tolerance: int,
) -> bool:
    for gid_a in members[root_a]:
        for gid_b in members[root_b]:
            if _intervals_conflict(
                intervals.get(gid_a, []), intervals.get(gid_b, []), tolerance):
                return True
    return False


def _candidate_pairs(
    gids: list[int],
    vectors: np.ndarray,
    threshold: float,
    margin: float,
    max_candidates_per_gid: int,
    intervals: dict[int, list[tuple[int, int, int]]] | None = None,
    geometry_points: dict[int, dict[int, list[tuple[int, float, float]]]] | None = None,
    geo_weight: float = 0.0,
    geo_min_overlaps: int = 20,
    geo_mode: str = "cooccur",
    gid_trajectories: dict[int, list[tuple[int, float, float]]] | None = None,
) -> list[tuple[float, int, int]]:
    if len(gids) < 2:
        return []

    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1.0)
    use_geo = bool(geometry_points) and geo_weight > 0.0
    use_traj = use_geo and geo_mode == "trajectory"
    if use_traj and gid_trajectories is None:
        gid_trajectories = _build_gid_trajectories(geometry_points)
    pairs = {}
    for i, gid in enumerate(gids):
        scores = sim[i].copy()
        if use_geo:
            for j, other_gid in enumerate(gids):
                if i == j:
                    continue
                if intervals and _intervals_conflict(
                    intervals.get(gid, []), intervals.get(other_gid, []), 0
                ):
                    continue
                if use_traj:
                    geo_score = _trajectory_pair_score(
                        gid, other_gid, gid_trajectories, geo_min_overlaps)
                else:
                    geo_score, _ = _geometry_pair_score(
                        gid, other_gid, geometry_points, geo_min_overlaps)
                scores[j] = (1.0 - geo_weight) * scores[j] + geo_weight * geo_score
        order = np.argsort(scores)[::-1]
        top = [
            j for j in order[:max(2, max_candidates_per_gid + 1)]
            if scores[j] >= threshold
        ]
        if not top:
            continue
        best = top[0]
        runner_up_score = scores[top[1]] if len(top) > 1 else -1.0
        if runner_up_score >= 0.0 and scores[best] < runner_up_score + margin:
            continue
        for j in top[:max_candidates_per_gid]:
            a, b = sorted((gid, gids[j]))
            pairs[(a, b)] = max(float(scores[j]), pairs.get((a, b), -1.0))

    return sorted(
        [(score, a, b) for (a, b), score in pairs.items()],
        key=lambda item: item[0],
        reverse=True,
    )


def _merge_map(
    gids: list[int],
    pairs: list[tuple[float, int, int]],
    intervals: dict[int, list[tuple[int, int, int]]],
    temporal_tolerance: int,
) -> tuple[dict[int, int], list[tuple[int, int, float]]]:
    uf = UnionFind(gids)
    members = {gid: {gid} for gid in gids}
    accepted = []

    for score, gid_a, gid_b in pairs:
        root_a = uf.find(gid_a)
        root_b = uf.find(gid_b)
        if root_a == root_b:
            continue
        if _component_conflict(
            root_a, root_b, members, intervals, temporal_tolerance):
            continue
        new_root = uf.union(root_a, root_b)
        old_root = root_b if new_root == root_a else root_a
        members[new_root] = members.pop(root_a, {root_a}) | members.pop(root_b, {root_b})
        members.pop(old_root, None)
        accepted.append((max(root_a, root_b), min(root_a, root_b), score))

    return {gid: uf.find(gid) for gid in gids}, accepted


def _write_remapped_predictions(
    pred_dir: Path,
    out_dir: Path,
    remap: dict[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pred_dir.glob("cam_*_predictions.csv")):
        out_path = out_dir / path.name
        with open(path, newline="") as src, open(out_path, "w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                gid = int(float(row["global_id"]))
                if gid >= 0:
                    row["global_id"] = remap.get(gid, gid)
                writer.writerow(row)

    for name in ("tracklets.csv", "tracklet_embeddings.npz", "tracklet_bev.csv"):
        src = pred_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _write_merge_map(
    out_dir: Path,
    remap: dict[int, int],
    accepted: list[tuple[int, int, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "merge_map.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_global_id", "target_global_id", "score"])
        writer.writeheader()
        for source_gid, target_gid, score in accepted:
            writer.writerow({
                "source_global_id": source_gid,
                "target_global_id": target_gid,
                "score": round(score, 6),
            })

    with open(out_dir / "global_id_remap.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["global_id", "remapped_global_id"])
        writer.writeheader()
        for gid in sorted(remap):
            writer.writerow({
                "global_id": gid,
                "remapped_global_id": remap[gid],
            })


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    tracklets, emb_by_tracklet = _load_tracklets(pred_dir)
    gids, vectors, intervals = _build_gid_summaries(
        tracklets,
        emb_by_tracklet,
        min_gid_embeddings=args.min_gid_embeddings,
        min_tracklet_detections=args.min_tracklet_detections,
    )
    pairs = _candidate_pairs(
        gids,
        vectors,
        threshold=args.threshold,
        margin=args.margin,
        max_candidates_per_gid=args.max_candidates_per_gid,
        intervals=intervals,
        geometry_points=_load_geometry_points(
            pred_dir,
            args.mmp_short_root,
            args.scene,
            args.geo_sample_step,
        ),
        geo_weight=max(0.0, min(1.0, args.geo_weight)),
        geo_min_overlaps=max(1, args.geo_min_overlaps),
    )
    remap, accepted = _merge_map(
        gids, pairs, intervals, temporal_tolerance=args.temporal_tolerance)

    print(f"[offline merge] pred_dir={pred_dir}")
    print(f"[offline merge] eligible_gids={len(gids)} candidate_pairs={len(pairs)}")
    print(f"[offline merge] accepted_merges={len(accepted)}")
    print(f"[offline merge] threshold={args.threshold} margin={args.margin}")
    if args.geo_weight > 0.0:
        print(f"[offline merge] geo_weight={args.geo_weight} "
              f"scene={args.scene} sample_step={args.geo_sample_step}")

    if args.dry_run:
        for source_gid, target_gid, score in accepted[:20]:
            print(
                f"  G{source_gid} -> G{target_gid} score={score:.3f}")
        return

    _write_remapped_predictions(pred_dir, out_dir, remap)
    _write_merge_map(out_dir, remap, accepted)
    print(f"[offline merge] wrote {out_dir}")


if __name__ == "__main__":
    main()
