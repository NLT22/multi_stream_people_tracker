"""RAG Phase B — FastAPI service exposing the deterministic query tools.

This is the backend the webUI "Ask" view (Phase D) and the LLM agent (Phase C)
both call. Every endpoint is a thin wrapper over src.rag.queries.RagStore — no
LLM here, so it is independently testable. Time ranges are ISO strings (the
agent resolves "10am today" -> ISO before calling).

Run:
  RAG_DB=output/rag/rag.sqlite venv/bin/python3 -m uvicorn src.rag.api:app --port 8077
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Query, HTTPException

from src.rag.queries import RagStore

DB = os.environ.get("RAG_DB", "output/rag/rag.sqlite")
app = FastAPI(title="MTMC RAG query API", version="1.0")


def _store(run_id: str | None) -> RagStore:
    try:
        return RagStore(DB, run_id)
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, f"cannot open RAG db {DB}: {e}")


def _range(t0: str | None, t1: str | None):
    if not t0 or not t1:
        return None
    return (datetime.fromisoformat(t0).timestamp(), datetime.fromisoformat(t1).timestamp())


@app.get("/runs")
def runs():
    s = _store(None); out = s.list_runs(); s.close(); return out


@app.get("/zones")
def zones(run_id: str | None = None):
    s = _store(run_id); out = {"run": s.run_id, "zones": s.zones(),
                               "persons": s.list_persons()}; s.close(); return out


@app.get("/person/{gid}/timeline")
def person_timeline(gid: int, run_id: str | None = None,
                    t_start: str | None = None, t_end: str | None = None):
    s = _store(run_id); out = s.person_timeline(gid, _range(t_start, t_end)); s.close()
    return {"global_id": gid, "intervals": out}


@app.get("/person/{gid}/trajectory")
def person_trajectory(gid: int, run_id: str | None = None, step: int = 1,
                      t_start: str | None = None, t_end: str | None = None):
    s = _store(run_id); out = s.person_trajectory_bev(gid, _range(t_start, t_end), step); s.close()
    return {"global_id": gid, "points": out}


@app.get("/person/{gid}/dwell")
def person_dwell(gid: int, run_id: str | None = None,
                 t_start: str | None = None, t_end: str | None = None):
    s = _store(run_id); out = s.person_dwell(gid, _range(t_start, t_end)); s.close()
    return {"global_id": gid, "dwell": out}


@app.get("/zones/top")
def top_zones(run_id: str | None = None, metric: str = "footfall", k: int = 5,
              t_start: str | None = None, t_end: str | None = None):
    s = _store(run_id); out = s.top_zones(_range(t_start, t_end), metric, k); s.close()
    return {"metric": metric, "top": out}


@app.get("/zones/{zone}/occupancy")
def zone_occupancy(zone: str, run_id: str | None = None,
                   t_start: str | None = None, t_end: str | None = None):
    s = _store(run_id); out = s.zone_occupancy(zone, _range(t_start, t_end)); s.close()
    return {"zone": zone, "series": out}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), run_id: str | None = Query(None), k: int = 5):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data); tmp = f.name
    try:
        s = _store(run_id); out = s.search_person_by_image(tmp, k=k); s.close()
    finally:
        os.unlink(tmp)
    return {"candidates": out}
