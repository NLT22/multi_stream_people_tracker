"""RAG Phase B — deterministic query functions over the SQLite store.

These are the *correctness gate*: pure, testable retrieval functions with no LLM.
The agent layer (Phase C) only orchestrates these and writes prose; it is never
the thing under test. Each function takes an optional `time_range=(t0,t1)` in
epoch seconds (None = whole run).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class RagStore:
    def __init__(self, db_path: str | Path, run_id: str | None = None):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        if run_id is None:
            row = self.conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
            run_id = row["run_id"] if row else None
        self.run_id = run_id

    # ---- helpers ----
    def _tr(self, col: str, time_range):
        if not time_range:
            return "", []
        return f" AND {col} BETWEEN ? AND ?", [time_range[0], time_range[1]]

    def list_runs(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM runs").fetchall()]

    def run_info(self) -> dict:
        r = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
        return dict(r) if r else {}

    def zones(self) -> list[str]:
        return [r["zone"] for r in self.conn.execute(
            "SELECT DISTINCT zone FROM detections WHERE run_id=? ORDER BY zone",
            (self.run_id,)).fetchall()]

    def list_persons(self) -> list[int]:
        return [r["global_id"] for r in self.conn.execute(
            "SELECT DISTINCT global_id FROM detections WHERE run_id=? ORDER BY global_id",
            (self.run_id,)).fetchall()]

    # ---- Route A: person-centric ----
    def person_timeline(self, gid: int, time_range=None) -> list[dict]:
        """Per-camera appearance intervals for a person (from presence)."""
        w, p = self._tr("t_start", time_range)
        rows = self.conn.execute(
            "SELECT cam_id, zone, t_start, t_end, seconds, n_frames FROM presence "
            "WHERE run_id=? AND global_id=?" + w + " ORDER BY t_start",
            [self.run_id, gid] + p).fetchall()
        return [dict(r) for r in rows]

    def person_trajectory_bev(self, gid: int, time_range=None, step: int = 1) -> list[dict]:
        """BEV foot-point path (world XY) over time for a person."""
        w, p = self._tr("ts", time_range)
        rows = self.conn.execute(
            "SELECT ts, cam_id, world_x, world_y, foot_x, foot_y, zone FROM detections "
            "WHERE run_id=? AND global_id=? AND world_x IS NOT NULL" + w + " ORDER BY ts",
            [self.run_id, gid] + p).fetchall()
        pts = [dict(r) for r in rows]
        return pts[::step] if step > 1 else pts

    def person_dwell(self, gid: int, time_range=None) -> list[dict]:
        """Per-zone dwell seconds for a person."""
        if time_range:
            w, p = self._tr("t_start", time_range)
            rows = self.conn.execute(
                "SELECT zone, ROUND(SUM(seconds),2) seconds, COUNT(*) visits FROM presence "
                "WHERE run_id=? AND global_id=?" + w + " GROUP BY zone ORDER BY seconds DESC",
                [self.run_id, gid] + p).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT zone, seconds, visits FROM dwell WHERE run_id=? AND global_id=? "
                "ORDER BY seconds DESC", (self.run_id, gid)).fetchall()
        return [dict(r) for r in rows]

    # ---- Route A: aggregate ----
    def top_zones(self, time_range=None, metric: str = "footfall", k: int = 5) -> list[dict]:
        """Ranked zones by footfall (unique people) or occupancy (person-seconds)."""
        agg = "SUM(footfall)" if metric == "footfall" else "ROUND(SUM(occupancy_s),2)"
        w, p = self._tr("t_bucket", time_range)
        rows = self.conn.execute(
            f"SELECT zone, {agg} AS value FROM zone_timeseries WHERE run_id=?" + w +
            " GROUP BY zone ORDER BY value DESC LIMIT ?",
            [self.run_id] + p + [k]).fetchall()
        return [{"zone": r["zone"], "metric": metric, "value": r["value"]} for r in rows]

    def zone_occupancy(self, zone: str, time_range=None) -> list[dict]:
        """Occupancy/footfall timeseries for a zone."""
        w, p = self._tr("t_bucket", time_range)
        rows = self.conn.execute(
            "SELECT bucket, t_bucket, occupancy_s, footfall FROM zone_timeseries "
            "WHERE run_id=? AND zone=?" + w + " ORDER BY bucket",
            [self.run_id, zone] + p).fetchall()
        return [dict(r) for r in rows]

    # ---- Route B: image search ----
    def _gallery(self):
        rows = self.conn.execute(
            "SELECT global_id, dim, vec FROM gid_embeddings WHERE run_id=?",
            (self.run_id,)).fetchall()
        gids = [r["global_id"] for r in rows]
        if not gids:
            return [], None
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return gids, mat

    def search_person_by_embedding(self, vec, k: int = 5) -> list[dict]:
        """Cosine-rank a query embedding against the per-gid gallery."""
        gids, mat = self._gallery()
        if mat is None:
            return []
        q = np.asarray(vec, dtype=np.float32).ravel()
        n = np.linalg.norm(q)
        q = q / n if n > 0 else q
        sims = mat @ q  # gallery already L2-normalised
        order = np.argsort(-sims)[:k]
        return [{"global_id": int(gids[i]), "score": round(float(sims[i]), 4)} for i in order]

    def search_person_by_image(self, image_path: str, k: int = 5) -> list[dict]:
        """Embed a person crop with the deployed Swin ONNX, then cosine-rank."""
        from src.rag.embed import embed_crop
        return self.search_person_by_embedding(embed_crop(image_path), k=k)

    def gid_embedding(self, gid: int):
        """Return a stored gid embedding (used for tests / self-match checks)."""
        r = self.conn.execute(
            "SELECT vec FROM gid_embeddings WHERE run_id=? AND global_id=?",
            (self.run_id, gid)).fetchone()
        return np.frombuffer(r["vec"], dtype=np.float32) if r else None

    def close(self):
        self.conn.close()
