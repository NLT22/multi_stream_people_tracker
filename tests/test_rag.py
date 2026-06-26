"""RAG Phase B correctness gate — deterministic tests (no LLM).

Validates the zone resolver, the SQLite ingest, every query function, the
image-search self-match invariant, and the agent's pure-Python tool dispatch.
The LLM layer is NOT under test here (by design — see production_todo §5.4).

Run: PYTHONPATH=. venv/bin/python3 -m pytest tests/test_rag.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rag.zones import ZoneRegistry, _point_in_poly
from src.rag.ingest import ingest_scene
from src.rag.queries import RagStore

EXPORT = REPO / "output/eval/full_mmp_val/64pm_cafe_shop_1"
SCENE = "64pm_cafe_shop_1"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    if not EXPORT.exists():
        pytest.skip(f"export not found: {EXPORT}")
    p = tmp_path_factory.mktemp("rag") / "rag.sqlite"
    ingest_scene(EXPORT, SCENE, p, fps=15.0, epoch_iso="2026-06-26T09:00:00")
    return str(p)


@pytest.fixture(scope="module")
def store(db):
    s = RagStore(db); yield s; s.close()


# ---- zones (pure, no DB) ----
def test_point_in_poly_square():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _point_in_poly(0.5, 0.5, sq)
    assert not _point_in_poly(1.5, 0.5, sq)


def test_zone_parse():
    reg = ZoneRegistry.from_analytics(REPO / "configs/analytics/nvdsanalytics_cafe_shop.txt")
    assert reg.zones, "should parse at least one camera's ROI"
    # a foot point inside a parsed ROI resolves to that named zone, not :other
    cam = next(iter(reg.zones))
    name, poly = reg.zones[cam][0]
    cx = sum(x for x, _ in poly) / len(poly) * 640
    cy = sum(y for _, y in poly) / len(poly) * 360
    assert reg.resolve(cam, cx, cy).endswith(name)
    assert reg.resolve(cam, -10, -10) == f"cam{cam}:other"


# ---- ingest ----
def test_ingest_tables(store):
    info = store.run_info()
    assert info["n_dets"] > 0 and info["n_gids"] > 0
    assert len(store.list_persons()) == info["n_gids"]
    assert len(store.zones()) > 0


# ---- aggregate queries ----
@pytest.mark.parametrize("metric", ["footfall", "occupancy"])
def test_top_zones_ranked(store, metric):
    top = store.top_zones(metric=metric, k=3)
    assert 1 <= len(top) <= 3
    vals = [r["value"] for r in top]
    assert vals == sorted(vals, reverse=True)        # descending
    assert all(r["metric"] == metric for r in top)


def test_zone_occupancy_series(store):
    z = store.top_zones(metric="footfall", k=1)[0]["zone"]
    series = store.zone_occupancy(z)
    assert series and all("occupancy_s" in r and "footfall" in r for r in series)


# ---- person queries ----
def test_person_timeline(store):
    gid = store.list_persons()[0]
    tl = store.person_timeline(gid)
    assert tl and all(r["t_end"] >= r["t_start"] for r in tl)


def test_person_dwell_sorted(store):
    gid = store.list_persons()[0]
    d = store.person_dwell(gid)
    secs = [r["seconds"] for r in d]
    assert secs == sorted(secs, reverse=True)


def test_trajectory_world_xy(store):
    gid = store.list_persons()[0]
    traj = store.person_trajectory_bev(gid, step=50)
    assert traj and all(p["world_x"] is not None for p in traj)
    ts = [p["ts"] for p in traj]
    assert ts == sorted(ts)                          # time-ordered


def test_time_range_filter(store):
    """A narrow window must return no more than the full run."""
    gid = store.list_persons()[0]
    full = store.person_trajectory_bev(gid)
    info = store.run_info()
    from datetime import datetime
    e = datetime.fromisoformat(info["epoch_iso"]).timestamp()
    windowed = store.person_trajectory_bev(gid, time_range=(e, e + 5))
    assert len(windowed) <= len(full)


# ---- image search: the key invariant ----
def test_image_self_match(store):
    """Every persisted gid embedding must rank ITSELF first."""
    gids = [r["global_id"] for r in store.conn.execute(
        "SELECT global_id FROM gid_embeddings WHERE run_id=?", (store.run_id,)).fetchall()]
    assert gids, "no gid embeddings persisted"
    for gid in gids:
        emb = store.gid_embedding(gid)
        res = store.search_person_by_embedding(emb, k=1)
        assert res[0]["global_id"] == gid
        assert res[0]["score"] > 0.99


# ---- agent dispatch (pure, no LLM) ----
def test_agent_dispatch(db):
    from src.rag.agent import RagAgent
    ag = RagAgent(db)
    top = ag.dispatch("top_zones", {"metric": "footfall", "k": 2})
    assert isinstance(top, list) and len(top) <= 2
    gid = ag.store.list_persons()[0]
    tl = ag.dispatch("person_timeline", {"global_id": gid})
    assert isinstance(tl, list) and tl
    assert "error" in ag.dispatch("search_person_by_image", {})  # no image supplied
    ag.close()
