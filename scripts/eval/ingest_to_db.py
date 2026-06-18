#!/usr/bin/env python3
"""Ingest a scene's global tracks (anchor/pipeline output) into the SQLite store
(production_todo §3). Loads cam_*_predictions.csv + tracklet_bev.csv (world foot),
optionally assigns zones and emits zone-enter events. Offline stand-in for the live
gallery/MTMC -> DB sink; lets the analytics/dashboard query one DB. No model/GPU.

  python scripts/eval/ingest_to_db.py \
      --pred-dir output/eval/heldout_64pm_office_0_anchor \
      --db output/db/64pm_office_0.db --fps 15 [--zones configs/zones/<scene>.json]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.storage import TrackDBSink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--db", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--zones", help="zones JSON; if set, also writes zone-enter events")
    ap.add_argument("--min-dwell", type=int, default=15)
    args = ap.parse_args()

    # world foot per (cam, frame, ltid)
    world = {}
    bev = args.pred_dir / "tracklet_bev.csv"
    if bev.exists():
        for r in pd.read_csv(bev).itertuples():
            world[(int(r.cam_id), int(r.frame_no_cam), int(r.local_track_id))] = (r.world_x, r.world_y)

    zones = None
    if args.zones:
        from src.analytics.zones import load_zones, assign_zone
        zones = load_zones(args.zones)

    n = 0
    with TrackDBSink(args.db, fresh=True) as sink:
        for csv in sorted(args.pred_dir.glob("cam_*_predictions.csv")):
            df = pd.read_csv(csv)
            for r in df.itertuples():
                wx, wy = world.get((int(r.cam_id), int(r.frame_no_cam), int(r.local_track_id)), (None, None))
                sink.add_track(ts=r.frame_no_cam / args.fps, cam_id=int(r.cam_id),
                               frame=int(r.frame_no_cam), local_id=int(r.local_track_id),
                               global_id=int(r.global_id), left=float(r.left), top=float(r.top),
                               w=float(r.width), h=float(r.height),
                               conf=float(getattr(r, "conf", -1)), world_x=wx, world_y=wy)
                n += 1
        # zone-enter events (debounced) from world foot, per global_id
        if zones is not None and world:
            recs = []
            for csv in sorted(args.pred_dir.glob("cam_*_predictions.csv")):
                for r in pd.read_csv(csv).itertuples():
                    w = world.get((int(r.cam_id), int(r.frame_no_cam), int(r.local_track_id)))
                    if w and int(r.global_id) >= 0:
                        recs.append((int(r.global_id), int(r.frame_no_cam), assign_zone(w[0], w[1], zones)))
            recs.sort(key=lambda r: (r[0], r[1]))   # by (global_id, frame); zone may be None
            prev = {}
            for gid, fr, z in recs:
                if isinstance(z, str) and prev.get(gid) != z:
                    sink.add_zone_event(ts=fr / args.fps, global_id=gid, zone=z, event="enter")
                    prev[gid] = z
    print(f"[ingest] {n} track rows -> {args.db}")


if __name__ == "__main__":
    main()
