"""RAG Phase A — build the queryable SQLite store from a scene export.

Reconstructs everything the Q&A layer needs from existing export artifacts
(`cam_*_predictions.csv`, `_eval_assign.csv`, `tracklet_embeddings.npz`,
`tracklets.csv`) + the env calibration + the gst-nvdsanalytics ROI zones:

  runs              one row per ingested run/scene (fps, epoch, env, n_cams)
  detections        per-detection: ts, buffered global_id, foot point, world XY, zone, bbox
  presence          merged per-(gid,cam,zone) visit intervals (t_start..t_end)
  dwell             per-(gid,zone) total seconds
  zone_timeseries   per-(zone,time_bucket) occupancy seconds + unique-gid footfall
  gid_embeddings    per-gid mean L2 embedding (BLOB) for image search

Buffered (anchor-guided) global IDs are used as the identity — the same IDs the
report scores. Wall-clock ts = run epoch + frame_no / fps (MMP has no capture
time; epoch is configurable so "today/10am/this week" resolve).
"""
from __future__ import annotations

import json
import sqlite3
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CALIB_BASE = REPO / "dataset/MMPTracking/MMPTracking_validation/validation/calibrations"
ANALYTICS_DIR = REPO / "configs/analytics"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, scene TEXT, env TEXT, fps REAL, epoch_iso TEXT,
  n_cams INT, n_dets INT, n_gids INT, bucket_s REAL);
CREATE TABLE IF NOT EXISTS detections (
  run_id TEXT, cam_id INT, frame_no INT, ts REAL, global_id INT, local_track_id INT,
  foot_x REAL, foot_y REAL, world_x REAL, world_y REAL, zone TEXT,
  left REAL, top REAL, width REAL, height REAL);
CREATE TABLE IF NOT EXISTS presence (
  run_id TEXT, global_id INT, cam_id INT, zone TEXT,
  t_start REAL, t_end REAL, frame_start INT, frame_end INT, n_frames INT, seconds REAL);
CREATE TABLE IF NOT EXISTS dwell (
  run_id TEXT, global_id INT, zone TEXT, seconds REAL, visits INT);
CREATE TABLE IF NOT EXISTS zone_timeseries (
  run_id TEXT, zone TEXT, bucket INT, t_bucket REAL, occupancy_s REAL, footfall INT);
CREATE TABLE IF NOT EXISTS gid_embeddings (
  run_id TEXT, global_id INT, dim INT, vec BLOB);
CREATE INDEX IF NOT EXISTS ix_det ON detections(run_id, global_id);
CREATE INDEX IF NOT EXISTS ix_det_zone ON detections(run_id, zone);
CREATE INDEX IF NOT EXISTS ix_pres ON presence(run_id, global_id);
CREATE INDEX IF NOT EXISTS ix_zts ON zone_timeseries(run_id, zone);
"""


def _scene_env(scene: str) -> str:
    return scene.removeprefix("64pm_").removeprefix("63am_").rsplit("_", 1)[0]


def _load_geometry(env: str):
    cal = CALIB_BASE / env / "calibrations.json"
    if not cal.exists():
        return None
    from src.reid.geometry import GroundPlaneGeometry
    return GroundPlaneGeometry(json.loads(cal.read_text()))


def _merge_intervals(frames: list[int], fps: float, epoch: float,
                     gap_frames: int) -> list[tuple]:
    """Merge a sorted frame list into visit intervals (gap <= gap_frames joins)."""
    if not frames:
        return []
    frames = sorted(frames)
    out, fs, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > gap_frames:
            out.append((fs, prev))
            fs = f
        prev = f
    out.append((fs, prev))
    return [(epoch + a / fps, epoch + b / fps, a, b, b - a + 1) for a, b in out]


def ingest_scene(export_dir: str | Path, scene: str, db_path: str | Path,
                 env: str | None = None, fps: float = 15.0,
                 epoch_iso: str = "2026-06-26T09:00:00", bucket_s: float = 30.0,
                 gap_s: float = 2.0, run_id: str | None = None,
                 analytics_path: str | Path | None = None) -> dict:
    export_dir = Path(export_dir)
    env = env or _scene_env(scene)
    run_id = run_id or scene
    epoch = datetime.fromisoformat(epoch_iso).timestamp()
    gap_frames = int(gap_s * fps)

    # zones + geometry
    ana = Path(analytics_path) if analytics_path else ANALYTICS_DIR / f"nvdsanalytics_{env}.txt"
    if not ana.exists():
        ana = ANALYTICS_DIR / "nvdsanalytics_mmp.txt"
    from src.rag.zones import ZoneRegistry
    zones = ZoneRegistry.from_analytics(ana) if ana.exists() else ZoneRegistry()
    geo = _load_geometry(env)

    # buffered global-id map (cam, frame, ltid) -> gid
    gid_map: dict = {}
    assign = export_dir / "_eval_assign.csv"
    if assign.exists():
        a = pd.read_csv(assign)
        gid_map = {(int(r.cam_id), int(r.frame_no), int(r.local_track_id)): int(r.global_id)
                   for r in a.itertuples()}

    # source_id -> gt_cam_id (1-based CameraId) for calibration projection
    pred_files = sorted(export_dir.glob("cam_*_predictions.csv"))
    source_ids = [int(p.stem.split("_")[1]) for p in pred_files]
    val_scene = REPO / "dataset/MMPTracking_10minute/val" / scene
    gt_cam_ids = sorted(int(p.stem[3:]) for p in val_scene.glob("cam*.mp4")) or [s + 1 for s in source_ids]
    src2cam = dict(zip(source_ids, gt_cam_ids))

    rows = []
    for p, cam in zip(pred_files, source_ids):
        df = pd.read_csv(p)
        for r in df.itertuples():
            gid = gid_map.get((cam, int(r.frame_no_cam), int(r.local_track_id)), -1)
            if gid < 0:
                continue
            fx, fy = float(r.left) + float(r.width) / 2.0, float(r.top) + float(r.height)
            wx = wy = None
            if geo is not None:
                fw = geo.bbox_foot(src2cam.get(cam, cam + 1), float(r.left), float(r.top),
                                   float(r.width), float(r.height))
                if fw is not None:
                    wx, wy = fw
            rows.append((run_id, cam, int(r.frame_no_cam), epoch + int(r.frame_no_cam) / fps,
                         gid, int(r.local_track_id), fx, fy, wx, wy,
                         zones.resolve(cam, fx, fy),
                         float(r.left), float(r.top), float(r.width), float(r.height)))

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    for t in ("runs", "detections", "presence", "dwell", "zone_timeseries", "gid_embeddings"):
        conn.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
    conn.executemany("INSERT INTO detections VALUES (" + ",".join(["?"] * 15) + ")", rows)

    det = pd.DataFrame(rows, columns=["run_id", "cam_id", "frame_no", "ts", "global_id",
                                      "local_track_id", "foot_x", "foot_y", "world_x", "world_y",
                                      "zone", "left", "top", "width", "height"])
    # presence + dwell
    pres_rows, dwell_acc = [], {}
    if len(det):
        for (gid, cam, zone), g in det.groupby(["global_id", "cam_id", "zone"]):
            for t0, t1, f0, f1, nf in _merge_intervals(g["frame_no"].tolist(), fps, epoch, gap_frames):
                pres_rows.append((run_id, int(gid), int(cam), zone, t0, t1, int(f0), int(f1),
                                  int(nf), round(t1 - t0, 3)))
                k = (int(gid), zone)
                s, v = dwell_acc.get(k, (0.0, 0))
                dwell_acc[k] = (s + (t1 - t0), v + 1)
    conn.executemany("INSERT INTO presence VALUES (" + ",".join(["?"] * 10) + ")", pres_rows)
    conn.executemany("INSERT INTO dwell VALUES (?,?,?,?,?)",
                     [(run_id, gid, z, round(s, 3), v) for (gid, z), (s, v) in dwell_acc.items()])

    # zone timeseries (occupancy seconds + unique-gid footfall per bucket)
    zts_rows = []
    if len(det):
        det = det.copy()
        det["bucket"] = ((det["frame_no"] / fps) // bucket_s).astype(int)
        for (zone, b), g in det.groupby(["zone", "bucket"]):
            zts_rows.append((run_id, zone, int(b), epoch + b * bucket_s,
                             round(len(g) / fps, 3), int(g["global_id"].nunique())))
    conn.executemany("INSERT INTO zone_timeseries VALUES (?,?,?,?,?,?)", zts_rows)

    # per-gid mean embedding (from tracklet embeddings, grouped by gid)
    n_emb = 0
    tnpz, tcsv = export_dir / "tracklet_embeddings.npz", export_dir / "tracklets.csv"
    if tnpz.exists() and tcsv.exists():
        z = np.load(tnpz)
        tid2vec = dict(zip(z["tracklet_ids"].astype(int), z["embeddings"].astype(np.float32)))
        tk = pd.read_csv(tcsv)
        # tracklets carry RAW global_id; remap to buffered gid via majority over its detections
        tl2buf: dict = {}
        for r in tk.itertuples():
            key_gids = det[(det.cam_id == r.cam_id) & (det.local_track_id == r.local_track_id)]["global_id"]
            if len(key_gids):
                tl2buf[int(r.tracklet_id)] = int(key_gids.mode().iloc[0])
        gid_vecs: dict = {}
        for tid, gid in tl2buf.items():
            if tid in tid2vec:
                gid_vecs.setdefault(gid, []).append(tid2vec[tid])
        for gid, vecs in gid_vecs.items():
            m = np.mean(vecs, axis=0)
            n = np.linalg.norm(m)
            m = (m / n) if n > 0 else m
            conn.execute("INSERT INTO gid_embeddings VALUES (?,?,?,?)",
                         (run_id, int(gid), int(m.shape[0]), m.astype(np.float32).tobytes()))
            n_emb += 1

    n_gids = int(det["global_id"].nunique()) if len(det) else 0
    conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                 (run_id, scene, env, fps, epoch_iso, len(source_ids), len(rows), n_gids, bucket_s))
    conn.commit()
    conn.close()
    return {"run_id": run_id, "env": env, "n_cams": len(source_ids), "n_dets": len(rows),
            "n_gids": n_gids, "n_embeddings": n_emb, "zones": len(zones.zone_names()),
            "db": str(db_path)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the RAG SQLite store from a scene export.")
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--db", default="output/rag/rag.sqlite")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--epoch-iso", default="2026-06-26T09:00:00")
    ap.add_argument("--bucket-s", type=float, default=30.0)
    args = ap.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(ingest_scene(args.export_dir, args.scene, args.db,
                                  fps=args.fps, epoch_iso=args.epoch_iso,
                                  bucket_s=args.bucket_s), indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    main()
