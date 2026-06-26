# Natural-Language Q&A / RAG layer

Structured-metadata RAG over the MTMC tracking outputs (see `docs/production_todo.md` §5).
A tool-using LLM router calls **deterministic** retrieval functions over a SQLite store
built from existing export artifacts — no video re-captioning (our CV pipeline already
emits structured identity + trajectory metadata).

```
ingest.py   Phase A  build the SQLite store (timestamps, named zones, presence/dwell/
                     zone-timeseries, per-gid embeddings) from a scene export
queries.py  Phase B  deterministic query functions (RagStore) — the correctness gate
api.py      Phase B  FastAPI service exposing the tools
embed.py             person-crop -> Swin ONNX embedding (image search)
agent.py    Phase C  Anthropic tool-use router that composes prose answers
```

## Build the store (Phase A)

```bash
PYTHONPATH=. venv/bin/python3 -m src.rag.ingest \
  --export-dir output/eval/full_mmp_val/64pm_cafe_shop_1 --scene 64pm_cafe_shop_1 \
  --db output/rag/rag.sqlite --fps 15 --epoch-iso 2026-06-26T09:00:00
```

Identities are the **buffered** global IDs (from `_eval_assign.csv`); zones come from
`configs/analytics/nvdsanalytics_<env>.txt` (the ROI editor's vocabulary); wall-clock
`ts = epoch + frame_no/fps` (MMP has no capture time — set `--epoch-iso` to place the run).

## Query API (Phase B)

```bash
RAG_DB=output/rag/rag.sqlite PYTHONPATH=. \
  venv/bin/python3 -m uvicorn src.rag.api:app --port 8077
# GET /zones/top?metric=footfall&k=5      busiest areas
# GET /person/{gid}/timeline|dwell|trajectory
# GET /zones/{zone}/occupancy             timeseries
# POST /search/image  (multipart file)    crop -> candidate global_ids
```

## Ask (Phase C — needs ANTHROPIC_API_KEY)

```python
from src.rag.agent import RagAgent
RagAgent("output/rag/rag.sqlite").ask("which area got the most attention?")
RagAgent("output/rag/rag.sqlite").ask("when did this person appear?", image_path="crop.jpg")
```

The LLM only orchestrates tools and writes prose; all retrieval is the tested `RagStore`.

## Tests (correctness gate)

```bash
PYTHONPATH=. venv/bin/python3 -m pytest tests/test_rag.py -v   # 12 tests, no LLM/network
```

## Status / not yet built
- Phase D (webUI "Ask" view) — frontend; the FastAPI endpoints are the integration seam.
- Image search reliability is lowest in **retail** (env IDF1 0.661).
- For richer "which shelf" answers, define finer ROIs per environment in the ROI editor;
  the resolver picks them up automatically.
