#!/usr/bin/env python3
"""Append-only SQLite sink for a production run (production_todo 3.4).

Ingests the existing export/log schema (no new pipeline coupling) into one SQLite
DB so runs are queryable without parsing CSV/NPZ by hand. Idempotent per run_id:
re-ingesting a run is a no-op unless --replace. This is a deliberately small sink
built around the current files — NOT the archived analytics/storage prototype.

Ingests:
  per-detection rows   <export>/cam_*_predictions.csv
  global assignments   <export>/_eval_assign.csv
  chunk metadata       <export>/det_emb_chunk_*.npz
  run health metrics   <logs>/long_stability.csv, <logs>/long_buffered.csv
  run provenance       <export>/run_manifest.json

  python scripts/eval/persist_run.py --run-dir output/runs/20260622_085207_sgie_reid0
  python scripts/eval/persist_run.py --export-dir output/eval/long_run --logs-dir output/logs
  python scripts/eval/persist_run.py --run-dir <dir> --db output/runs.sqlite --replace
"""
from __future__ import annotations
import argparse, glob, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created TEXT, git_commit TEXT, gpu TEXT,
  pipeline_config TEXT, preset TEXT, sources TEXT, env_map TEXT,
  duration_s REAL, detector_onnx TEXT, reid_onnx TEXT, manifest_json TEXT);
CREATE TABLE IF NOT EXISTS detections (
  run_id TEXT, cam_id INT, frame_no INT, local_track_id INT, global_id INT,
  left REAL, top REAL, width REAL, height REAL);
CREATE TABLE IF NOT EXISTS assignments (
  run_id TEXT, grp TEXT, cam_id INT, frame_no INT, local_track_id INT, global_id INT);
CREATE TABLE IF NOT EXISTS health (
  run_id TEXT, ts TEXT, elapsed_s REAL, gpu_util REAL, gpu_mem_mb REAL,
  rss_mb REAL, fps REAL, n_gids INT);
CREATE TABLE IF NOT EXISTS buffered (
  run_id TEXT, ts TEXT, elapsed_s REAL, chunk INT, grp TEXT, n_dets INT, k INT,
  n_clusters INT, active_gids INT, total_gids INT, cluster_ms REAL);
CREATE TABLE IF NOT EXISTS chunks (
  run_id TEXT, file TEXT, n_rows INT, frame_min INT, frame_max INT, dim INT);
CREATE INDEX IF NOT EXISTS ix_det_run  ON detections(run_id, cam_id);
CREATE INDEX IF NOT EXISTS ix_asg_run  ON assignments(run_id, global_id);
"""


def _append(conn, table, df, run_id):
    if df is None or df.empty:
        return 0
    df = df.copy()
    df.insert(0, "run_id", run_id)
    df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, help="run dir (export/ + log CSVs + manifest)")
    ap.add_argument("--export-dir", type=Path, help="override: dir with cam_*/chunks/_eval_assign")
    ap.add_argument("--logs-dir", type=Path, help="override: dir with long_stability/long_buffered")
    ap.add_argument("--db", type=Path, default=Path("output/runs.sqlite"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    if args.run_dir:
        export = args.export_dir or (args.run_dir / "export" if (args.run_dir / "export").exists()
                                     else args.run_dir)
        logs = args.logs_dir or args.run_dir
    else:
        if not args.export_dir:
            ap.error("need --run-dir or --export-dir")
        export = args.export_dir
        logs = args.logs_dir or Path("output/logs")

    manifest = {}
    mpath = export / "run_manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    run_id = args.run_id or (
        f"{manifest.get('created','?')}_{manifest.get('run_params',{}).get('preset','run')}"
        if manifest else (args.run_dir.name if args.run_dir else export.name))

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    exists = conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if exists and not args.replace:
        print(f"[persist] run '{run_id}' already ingested — skip (use --replace).")
        conn.close()
        return
    if exists:
        for t in ("runs", "detections", "assignments", "health", "buffered", "chunks"):
            conn.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))

    m = manifest
    conn.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, m.get("created"), m.get("git_commit"), m.get("gpu"),
         m.get("pipeline_config"), (m.get("run_params") or {}).get("preset"),
         m.get("sources"), m.get("env_map"), m.get("duration_s"),
         (m.get("models") or {}).get("detector_onnx"),
         (m.get("models") or {}).get("reid_sgie_onnx"), json.dumps(m)))

    counts = {}
    # detections
    n = 0
    for f in sorted(glob.glob(str(export / "cam_*_predictions.csv"))):
        df = pd.read_csv(f).rename(columns={"frame_no_cam": "frame_no"})
        keep = ["cam_id", "frame_no", "local_track_id", "global_id",
                "left", "top", "width", "height"]
        n += _append(conn, "detections", df[[c for c in keep if c in df.columns]], run_id)
    counts["detections"] = n
    # assignments
    ap_csv = export / "_eval_assign.csv"
    if ap_csv.exists():
        df = pd.read_csv(ap_csv).rename(columns={"group": "grp"})
        counts["assignments"] = _append(conn, "assignments", df, run_id)
    # health + buffered
    s = logs / "long_stability.csv"
    if s.exists():
        counts["health"] = _append(conn, "health", pd.read_csv(s), run_id)
    b = logs / "long_buffered.csv"
    if b.exists():
        counts["buffered"] = _append(conn, "buffered",
                                     pd.read_csv(b).rename(columns={"group": "grp"}), run_id)
    # chunk metadata
    rows = []
    for f in sorted(glob.glob(str(export / "det_emb_chunk_*.npz"))):
        d = np.load(f, allow_pickle=True)
        fr = np.asarray(d["frame_no"]) if "frame_no" in d else np.array([])
        emb = d["embeddings"]
        rows.append({"file": Path(f).name, "n_rows": int(emb.shape[0]),
                     "frame_min": int(fr.min()) if fr.size else -1,
                     "frame_max": int(fr.max()) if fr.size else -1,
                     "dim": int(emb.shape[1]) if emb.ndim > 1 else 0})
    if rows:
        counts["chunks"] = _append(conn, "chunks", pd.DataFrame(rows), run_id)

    conn.commit()
    print(f"[persist] run '{run_id}' -> {args.db}")
    for t, c in counts.items():
        print(f"    {t:12s} {c}")
    conn.close()


if __name__ == "__main__":
    main()
