"""Storage layer for the tracker (production_todo §3): a batched SQLite sink for
the high-rate track stream + zone events. SQLite is the single-host start; the
same schema maps to TimescaleDB hypertables later.
"""
from .db_sink import TrackDBSink
__all__ = ["TrackDBSink"]
