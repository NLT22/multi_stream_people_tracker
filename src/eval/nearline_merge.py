"""Nearline-style global ID remap from exported MTMC predictions.

This module simulates the production design we want for realtime MTMC:
the gallery emits temporary online IDs, and a delayed association service emits
remap events once enough tracklet evidence is available.

It consumes the same export files as offline_merge.py and writes:

    remap_events.csv       delayed source_gid -> target_gid events
    global_id_remap.csv    final gid remap table after all events
    cam_*_predictions.csv  predictions with all remaps applied

Unlike offline_merge.py, candidate evidence is restricted to tracklets whose
end_frame is inside the current nearline window. This makes it a closer proxy
for a delayed service than a full future-looking offline pass.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from src.eval import offline_merge


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit delayed global-ID remap events from exported tracklets")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--min-gid-embeddings", type=int, default=6)
    p.add_argument("--min-tracklet-detections", type=int, default=10)
    p.add_argument("--max-candidates-per-gid", type=int, default=5)
    p.add_argument("--temporal-tolerance", type=int, default=0)
    p.add_argument("--window-frames", type=int, default=125,
                   help="Nearline decision window, e.g. 125 = 5s at 25 FPS")
    p.add_argument("--delay-frames", type=int, default=50,
                   help="Decision delay after window end, e.g. 50 = 2s at 25 FPS")
    p.add_argument("--mmp-short-root", default=None)
    p.add_argument("--scene", default=None)
    p.add_argument("--geo-weight", type=float, default=0.25)
    p.add_argument("--geo-sample-step", type=int, default=5)
    p.add_argument("--geo-min-overlaps", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _window_tracklets(tracklets: list[dict], end_frame: int) -> list[dict]:
    return [row for row in tracklets if row["end_frame"] <= end_frame]


def _compress_remap(remap: dict[int, int]) -> dict[int, int]:
    def find(gid: int) -> int:
        nxt = remap.get(gid, gid)
        if nxt != gid:
            remap[gid] = find(nxt)
        return remap.get(gid, gid)

    for gid in list(remap):
        find(gid)
    return remap


def _write_remap_events(
    out_dir: Path,
    events: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "remap_events.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_frame",
            "window_end_frame",
            "source_global_id",
            "target_global_id",
            "score",
        ])
        writer.writeheader()
        writer.writerows(events)


def _event_remap(events: list[dict]) -> dict[int, int]:
    remap: dict[int, int] = {}
    for event in events:
        source = int(event["source_global_id"])
        target = int(event["target_global_id"])
        remap[source] = min(target, remap.get(source, target))
        _compress_remap(remap)
    return _compress_remap(remap)


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    tracklets, emb_by_tracklet = offline_merge._load_tracklets(pred_dir)
    if not tracklets:
        raise SystemExit("[nearline merge] no tracklets found")

    max_end = max(row["end_frame"] for row in tracklets)
    window = max(1, args.window_frames)
    delay = max(0, args.delay_frames)

    geometry_points = offline_merge._load_geometry_points(
        pred_dir,
        args.mmp_short_root,
        args.scene,
        max(1, args.geo_sample_step),
    )

    accepted_seen: set[tuple[int, int]] = set()
    events: list[dict] = []
    cumulative_remap: dict[int, int] = {}

    for window_end in range(window, max_end + window, window):
        visible_tracklets = _window_tracklets(tracklets, min(window_end, max_end))
        gids, vectors, intervals = offline_merge._build_gid_summaries(
            visible_tracklets,
            emb_by_tracklet,
            min_gid_embeddings=args.min_gid_embeddings,
            min_tracklet_detections=args.min_tracklet_detections,
        )
        pairs = offline_merge._candidate_pairs(
            gids,
            vectors,
            threshold=args.threshold,
            margin=args.margin,
            max_candidates_per_gid=args.max_candidates_per_gid,
            intervals=intervals,
            geometry_points=geometry_points,
            geo_weight=max(0.0, min(1.0, args.geo_weight)),
            geo_min_overlaps=max(1, args.geo_min_overlaps),
        )
        _, accepted = offline_merge._merge_map(
            gids,
            pairs,
            intervals,
            temporal_tolerance=args.temporal_tolerance,
        )

        for source_gid, target_gid, score in accepted:
            source_gid = cumulative_remap.get(source_gid, source_gid)
            target_gid = cumulative_remap.get(target_gid, target_gid)
            if source_gid == target_gid:
                continue
            source_gid, target_gid = max(source_gid, target_gid), min(source_gid, target_gid)
            key = (source_gid, target_gid)
            if key in accepted_seen:
                continue
            accepted_seen.add(key)
            cumulative_remap[source_gid] = target_gid
            _compress_remap(cumulative_remap)
            events.append({
                "event_frame": min(window_end, max_end) + delay,
                "window_end_frame": min(window_end, max_end),
                "source_global_id": source_gid,
                "target_global_id": target_gid,
                "score": round(float(score), 6),
            })

        if window_end >= max_end:
            break

    final_remap = _event_remap(events)

    print(f"[nearline merge] pred_dir={pred_dir}")
    print(f"[nearline merge] windows={int(np.ceil(max_end / window))} "
          f"window_frames={window} delay_frames={delay}")
    print(f"[nearline merge] remap_events={len(events)} "
          f"final_remaps={sum(1 for gid, to_gid in final_remap.items() if gid != to_gid)}")
    if args.geo_weight > 0.0:
        print(f"[nearline merge] geo_weight={args.geo_weight} "
              f"scene={args.scene} sample_step={args.geo_sample_step}")

    if args.dry_run:
        for event in events[:20]:
            print(
                f"  frame {event['event_frame']}: "
                f"G{event['source_global_id']} -> G{event['target_global_id']} "
                f"score={event['score']}"
            )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    offline_merge._write_remapped_predictions(pred_dir, out_dir, final_remap)
    offline_merge._write_merge_map(
        out_dir,
        final_remap,
        [
            (int(e["source_global_id"]), int(e["target_global_id"]), float(e["score"]))
            for e in events
        ],
    )
    _write_remap_events(out_dir, events)

    for name in ("tracklets.csv", "tracklet_embeddings.npz"):
        src = pred_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    print(f"[nearline merge] wrote {out_dir}")


if __name__ == "__main__":
    main()
