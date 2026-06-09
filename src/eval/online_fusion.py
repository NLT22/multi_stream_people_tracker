"""Drive the streaming MicroBatchFusion engine over exported predictions.

This is the production-correct replacement for per-frame online gallery matching:
it replays exported tracklets through `MicroBatchFusion` in streaming order and
writes remapped predictions, so it plugs into the exact same eval harness as
`nearline_merge` and can be compared on IDF1.

    python -m src.eval.online_fusion \
        --pred-dir output/eval/mmp_lobby0 \
        --out-dir  output/eval/mmp_lobby0_online \
        --interval-frames 125 --threshold 0.55 --geo-weight 0.25 \
        --mmp-short-root dataset/MMPTracking_short --scene lobby_0

Unlike nearline_merge (a batch loop over the whole file), the engine here ingests
tracklets incrementally and fuses on a fixed micro-batch cadence — the same shape
the live pipeline would call. Output files match nearline_merge:

    remap_events.csv        micro-batch source_gid -> target_gid events
    global_id_remap.csv     final gid remap table
    merge_map.csv           accepted merges with scores
    cam_*_predictions.csv   predictions with all remaps applied
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from src.eval import offline_merge
from src.reid.micro_batch_fusion import MicroBatchFusion


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming micro-batch cross-camera fusion over exports")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--interval-frames", type=int, default=125,
                   help="Micro-batch decision cadence, e.g. 125 = 5s at 25 FPS")
    p.add_argument("--delay-frames", type=int, default=50,
                   help="Latency added to event timestamps (fusion compute model)")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--min-gid-embeddings", type=int, default=4)
    p.add_argument("--min-tracklet-detections", type=int, default=6)
    p.add_argument("--max-candidates-per-gid", type=int, default=5)
    p.add_argument("--temporal-tolerance", type=int, default=0)
    p.add_argument("--geo-weight", type=float, default=0.25)
    p.add_argument("--geo-sample-step", type=int, default=5)
    p.add_argument("--geo-min-overlaps", type=int, default=8)
    p.add_argument("--mmp-short-root", default=None)
    p.add_argument("--scene", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _write_remap_events(out_dir: Path, events: list[dict]) -> None:
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


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    tracklets, emb_by_tracklet = offline_merge._load_tracklets(pred_dir)
    if not tracklets:
        raise SystemExit("[online fusion] no tracklets found")

    geometry_points = offline_merge._load_geometry_points(
        pred_dir,
        args.mmp_short_root,
        args.scene,
        max(1, args.geo_sample_step),
    )

    fusion = MicroBatchFusion(
        interval_frames=args.interval_frames,
        delay_frames=args.delay_frames,
        threshold=args.threshold,
        margin=args.margin,
        min_gid_embeddings=args.min_gid_embeddings,
        min_tracklet_detections=args.min_tracklet_detections,
        max_candidates_per_gid=args.max_candidates_per_gid,
        temporal_tolerance=args.temporal_tolerance,
        geo_weight=args.geo_weight,
        geo_min_overlaps=args.geo_min_overlaps,
        geometry_points=geometry_points,
    )

    # Stream tracklets in the order perception would complete them (by end_frame).
    # A tracklet's aggregated evidence becomes available once it ends, so we
    # ingest it at end_frame and advance the engine clock to that frame.
    ordered = sorted(tracklets, key=lambda row: row["end_frame"])
    max_end = ordered[-1]["end_frame"]

    for row in ordered:
        fusion.ingest_tracklet(
            tracklet_id=row["tracklet_id"],
            cam_id=row["cam_id"],
            local_track_id=row["local_track_id"],
            global_id=row["global_id"],
            start_frame=row["start_frame"],
            end_frame=row["end_frame"],
            num_detections=row["num_detections"],
            num_embeddings=row["num_embeddings"],
            embedding=emb_by_tracklet.get(row["tracklet_id"]),
        )
        fusion.step(row["end_frame"])

    fusion.flush(max_end)

    events = fusion.events
    final_remap = fusion.final_remap()
    n_merges = sum(1 for gid, to_gid in final_remap.items() if gid != to_gid)
    n_windows = max_end // fusion.interval_frames + 1

    print(f"[online fusion] pred_dir={pred_dir}")
    print(f"[online fusion] micro_batches={n_windows} "
          f"interval_frames={fusion.interval_frames} "
          f"delay_frames={fusion.delay_frames}")
    print(f"[online fusion] remap_events={len(events)} final_remaps={n_merges}")
    if args.geo_weight > 0.0:
        print(f"[online fusion] geo_weight={args.geo_weight} "
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

    print(f"[online fusion] wrote {out_dir}")


if __name__ == "__main__":
    main()
