"""Batched SQLite sink for the track stream + zone events (production_todo §3).

Buffers rows and flushes every `batch` (NOT per-detection — at 20 cam × 10 fps ×
~10 people ≈ 2000 rows/s, per-row insert bottlenecks before the GPU). Indices are
built at close() so bulk ingest stays fast. The schema maps 1:1 to a TimescaleDB
hypertable (tracks on ts) when moving off single-host.

Usage (live, in the gallery probe / MTMC service):
    sink = TrackDBSink("output/run.db")
    sink.add_track(ts=.., cam_id=.., frame=.., local_id=.., global_id=..,
                   left=.., top=.., w=.., h=.., conf=.., world_x=.., world_y=..)
    sink.add_zone_event(ts=.., global_id=.., zone="checkout", event="enter")
    sink.close()
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_TRACK_COLS = ("ts", "cam_id", "frame", "local_id", "global_id",
               "left", "top", "w", "h", "conf", "world_x", "world_y")
_ZONE_COLS = ("ts", "global_id", "zone", "event")


class TrackDBSink:
    def __init__(self, path: str | Path, batch: int = 5000,
                 wal: bool = True, fresh: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and path.exists():
            path.unlink()
        self.con = sqlite3.connect(str(path))
        if wal:
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
        self.batch = batch
        self._tracks: list[tuple] = []
        self._zones: list[tuple] = []
        self._create_schema()

    def _create_schema(self) -> None:
        self.con.execute(f"""CREATE TABLE IF NOT EXISTS tracks (
            {', '.join(c + (' INTEGER' if c in ('cam_id','frame','local_id','global_id')
                            else ' REAL') for c in _TRACK_COLS)})""")
        self.con.execute("""CREATE TABLE IF NOT EXISTS zone_events (
            ts REAL, global_id INTEGER, zone TEXT, event TEXT)""")
        self.con.commit()

    # ------------------------------------------------------------- writes
    def add_track(self, **kw) -> None:
        self._tracks.append(tuple(kw.get(c) for c in _TRACK_COLS))
        if len(self._tracks) >= self.batch:
            self._flush_tracks()

    def add_zone_event(self, ts: float, global_id: int, zone: str, event: str) -> None:
        self._zones.append((ts, global_id, zone, event))
        if len(self._zones) >= self.batch:
            self._flush_zones()

    def _flush_tracks(self) -> None:
        if self._tracks:
            self.con.executemany(
                f"INSERT INTO tracks ({','.join(_TRACK_COLS)}) "
                f"VALUES ({','.join('?' * len(_TRACK_COLS))})", self._tracks)
            self._tracks.clear()

    def _flush_zones(self) -> None:
        if self._zones:
            self.con.executemany(
                f"INSERT INTO zone_events ({','.join(_ZONE_COLS)}) "
                f"VALUES ({','.join('?' * len(_ZONE_COLS))})", self._zones)
            self._zones.clear()

    def flush(self) -> None:
        self._flush_tracks(); self._flush_zones(); self.con.commit()

    def build_indices(self) -> None:
        """Build query indices (call once after bulk ingest; slow to maintain live)."""
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_cam_ts ON tracks(cam_id, ts)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_gid_ts ON tracks(global_id, ts)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_zone_ts ON zone_events(ts, zone)")
        self.con.commit()

    def close(self) -> None:
        self.flush(); self.build_indices(); self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
