#!/usr/bin/env python3
"""Spatial analytics on multi-camera global tracks (production_todo §4):
  (a) common movement routes between zones,
  (b) most-used entry/exit points,
  (c) people count per zone over time.

Consumes the world ground-plane foot points (tracklet_bev.csv: global_id, world_x,
world_y, frame_no_cam) produced by the pipeline/anchor stage. Zones come from
configs/zones/<scene>.json (semantic named polygons) or an auto grid fallback.
No model / GPU needed.

  python scripts/eval/zone_analytics.py \
      --pred-dir output/eval/heldout_64pm_office_0_anchor \
      --out-dir  output/analytics/64pm_office_0 \
      [--zones configs/zones/64pm_office_0.json] [--fps 15 --bucket-sec 60]
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.zones import load_zones, save_zones, assign_zone, auto_grid_zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", help="dir containing tracklet_bev.csv (or use --bev)")
    ap.add_argument("--bev", help="explicit tracklet_bev.csv path")
    ap.add_argument("--zones", help="zones JSON (default: auto grid)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--bucket-sec", type=float, default=60.0)
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--top-routes", type=int, default=15)
    ap.add_argument("--min-dwell", type=int, default=15,
                    help="frames a zone must be held to count as a real visit "
                         "(debounces ground-plane boundary jitter; 15 ~ 1s @15fps)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bev_path = Path(args.bev) if args.bev else Path(args.pred_dir) / "tracklet_bev.csv"
    df = pd.read_csv(bev_path)
    df = df[df["global_id"] >= 0]
    # one world point per (global_id, frame): mean across cameras seeing them
    pts = (df.groupby(["global_id", "frame_no_cam"])[["world_x", "world_y"]]
             .mean().reset_index())

    if args.zones:
        zones = load_zones(args.zones)
        print(f"[zones] loaded {len(zones)} named zones from {args.zones}")
    else:
        zones = auto_grid_zones(pts.world_x.to_numpy(), pts.world_y.to_numpy(),
                                args.nx, args.ny)
        zpath = args.out_dir / "zones_auto.json"
        save_zones(zones, zpath)
        print(f"[zones] no --zones given; auto {args.nx}x{args.ny} grid -> {zpath}")

    pts["zone"] = [assign_zone(x, y, zones) for x, y in zip(pts.world_x, pts.world_y)]
    zone_tags = {z.name: z.tags for z in zones}
    bucket_frames = max(1, int(args.fps * args.bucket_sec))
    pts["bucket"] = (pts["frame_no_cam"] // bucket_frames).astype(int)

    # ---- (a) routes between zones ----
    transitions = Counter()       # (A,B) directed
    full_paths = Counter()
    entry_z, exit_z = Counter(), Counter()
    def _debounced(zlist: list, flist: list) -> list[tuple[str, int]]:
        """Run-length-encode per-frame zones, keep runs held >= min_dwell (kills
        boundary jitter), collapse consecutive dupes. Returns (zone, start_frame)."""
        runs: list[list] = []           # [zone, count, start_frame]
        for z, f in zip(zlist, flist):
            if runs and runs[-1][0] == z:
                runs[-1][1] += 1
            else:
                runs.append([z, 1, int(f)])
        stable = [(z, sf) for z, n, sf in runs if isinstance(z, str) and n >= args.min_dwell]
        out = []
        for z, sf in stable:
            if not out or out[-1][0] != z:
                out.append((z, sf))
        return out

    enter_rows = []   # debounced zone-entry events (for throughput)
    for gid, g in pts.sort_values("frame_no_cam").groupby("global_id"):
        gd = g.sort_values("frame_no_cam")
        path = _debounced(gd["zone"].tolist(), gd["frame_no_cam"].tolist())
        if not path:
            continue
        collapsed = [z for z, _ in path]
        entry_z[collapsed[0]] += 1
        exit_z[collapsed[-1]] += 1
        for a, b in zip(collapsed, collapsed[1:]):
            transitions[(a, b)] += 1
        if len(collapsed) >= 2:
            full_paths["→".join(collapsed)] += 1
        for z, sf in path:
            enter_rows.append({"bucket": int(sf // bucket_frames), "zone": z})

    pd.DataFrame([{"from": a, "to": b, "count": c}
                  for (a, b), c in transitions.most_common()]
                 ).to_csv(args.out_dir / "routes_transitions.csv", index=False)
    pd.DataFrame([{"path": p, "count": c}
                  for p, c in full_paths.most_common(args.top_routes)]
                 ).to_csv(args.out_dir / "routes_top.csv", index=False)

    # ---- (b) entry/exit ranking ----
    allz = sorted(set(entry_z) | set(exit_z))
    pd.DataFrame([{"zone": z, "tags": "|".join(zone_tags.get(z, [])),
                   "entries": entry_z.get(z, 0), "exits": exit_z.get(z, 0)}
                  for z in allz]
                 ).sort_values("entries", ascending=False
                 ).to_csv(args.out_dir / "entry_exit_ranking.csv", index=False)

    # ---- (c) per-zone occupancy + throughput over time ----
    occ = (pts[pts.zone.notna()].groupby(["bucket", "zone"])["global_id"]
           .nunique().reset_index().rename(columns={"global_id": "n_unique"}))
    # throughput: debounced zone-entry events collected above (enter_rows)
    ent = (pd.DataFrame(enter_rows).groupby(["bucket", "zone"]).size()
           .reset_index(name="n_enter") if enter_rows else
           pd.DataFrame(columns=["bucket", "zone", "n_enter"]))
    occ = occ.merge(ent, on=["bucket", "zone"], how="left").fillna({"n_enter": 0})
    occ["n_enter"] = occ["n_enter"].astype(int)
    occ["t_sec"] = occ["bucket"] * args.bucket_sec
    occ.sort_values(["bucket", "zone"]).to_csv(
        args.out_dir / "zone_occupancy_timeseries.csv", index=False)

    # ---- flow map (zone centroids + directed edges) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        cen = {z.name: z.centroid for z in zones}
        for z in zones:
            p = np.asarray(z.polygon + [z.polygon[0]])
            ax.plot(p[:, 0], p[:, 1], "0.7", lw=0.8)
            ax.text(*cen[z.name], z.name, ha="center", va="center", fontsize=8, color="navy")
        mx = max(transitions.values()) if transitions else 1
        for (a, b), c in transitions.items():
            if a not in cen or b not in cen or a == b:
                continue
            x0, y0 = cen[a]; x1, y1 = cen[b]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", lw=0.5 + 4 * c / mx,
                                        color="crimson", alpha=0.6))
        ax.set_title("Zone flow (edge width ∝ #transitions)")
        ax.set_aspect("equal"); ax.invert_yaxis()
        fig.savefig(args.out_dir / "flow_map.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[flow-map] skipped: {e}")

    # ---- summary ----
    print(f"[zone-analytics] {pts.global_id.nunique()} global ids, "
          f"{len(zones)} zones, {pts.bucket.max()+1} time buckets ({args.bucket_sec:g}s)")
    print("  top routes:")
    for p, c in full_paths.most_common(5):
        print(f"    {p}  ({c})")
    print("  busiest entry zones:", dict(entry_z.most_common(3)))
    print("  busiest exit zones :", dict(exit_z.most_common(3)))
    print(f"  -> {args.out_dir}/ (routes_top.csv, routes_transitions.csv, "
          f"entry_exit_ranking.csv, zone_occupancy_timeseries.csv, flow_map.png)")


if __name__ == "__main__":
    main()
