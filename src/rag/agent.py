"""RAG Phase C — tool-using LLM agent (Anthropic Messages API tool-use).

The LLM is a *router*: it picks one of the deterministic query tools (Phase B),
fills params (resolving "10am today" -> ISO range, an uploaded crop -> global_id
via image search), then composes a prose answer. It never touches embeddings or
SQL directly — all retrieval is the tested RagStore.

Usage:
  from src.rag.agent import RagAgent
  ans = RagAgent("output/rag/rag.sqlite").ask("which area was busiest?")
  ans = RagAgent(db).ask("when did this person appear?", image_path="crop.jpg")

Requires ANTHROPIC_API_KEY. The TOOLS schema + dispatch are pure-Python and
unit-testable without the API (see tests).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.rag.queries import RagStore

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {"name": "top_zones",
     "description": "Rank areas/zones by footfall (unique people) or occupancy "
                    "(person-seconds). Use for 'busiest area', 'most attention'.",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": ["footfall", "occupancy"]},
         "k": {"type": "integer"},
         "t_start": {"type": "string", "description": "ISO datetime, optional"},
         "t_end": {"type": "string", "description": "ISO datetime, optional"}},
         "required": ["metric"]}},
    {"name": "zone_occupancy",
     "description": "Occupancy/footfall timeseries for one named zone.",
     "input_schema": {"type": "object", "properties": {
         "zone": {"type": "string"}, "t_start": {"type": "string"}, "t_end": {"type": "string"}},
         "required": ["zone"]}},
    {"name": "person_timeline",
     "description": "When/where a specific person (global_id) appeared: per-camera intervals.",
     "input_schema": {"type": "object", "properties": {
         "global_id": {"type": "integer"}, "t_start": {"type": "string"}, "t_end": {"type": "string"}},
         "required": ["global_id"]}},
    {"name": "person_dwell",
     "description": "Per-zone dwell seconds for a person (global_id).",
     "input_schema": {"type": "object", "properties": {
         "global_id": {"type": "integer"}, "t_start": {"type": "string"}, "t_end": {"type": "string"}},
         "required": ["global_id"]}},
    {"name": "person_trajectory_bev",
     "description": "BEV (top-down) world-XY path over time for a person (global_id).",
     "input_schema": {"type": "object", "properties": {
         "global_id": {"type": "integer"}, "step": {"type": "integer"}},
         "required": ["global_id"]}},
    {"name": "search_person_by_image",
     "description": "Identify the global_id(s) matching the uploaded person crop. "
                    "Call FIRST when the user asks about 'this person' with an image.",
     "input_schema": {"type": "object", "properties": {"k": {"type": "integer"}}}},
]


def _iso(t):
    return (datetime.fromisoformat(t).timestamp() if isinstance(t, str) and t else None)


def _trange(args):
    a, b = args.get("t_start"), args.get("t_end")
    return (_iso(a), _iso(b)) if a and b else None


class RagAgent:
    def __init__(self, db_path: str | Path, run_id: str | None = None,
                 image_path: str | None = None, model: str = MODEL):
        self.store = RagStore(db_path, run_id)
        self.image_path = image_path
        self.model = model

    # ---- deterministic tool dispatch (testable without the LLM) ----
    def dispatch(self, name: str, args: dict):
        tr = _trange(args)
        if name == "top_zones":
            return self.store.top_zones(tr, args.get("metric", "footfall"), args.get("k", 5))
        if name == "zone_occupancy":
            return self.store.zone_occupancy(args["zone"], tr)
        if name == "person_timeline":
            return self.store.person_timeline(args["global_id"], tr)
        if name == "person_dwell":
            return self.store.person_dwell(args["global_id"], tr)
        if name == "person_trajectory_bev":
            return self.store.person_trajectory_bev(args["global_id"], tr, args.get("step", 1))
        if name == "search_person_by_image":
            if not self.image_path:
                return {"error": "no image supplied with the question"}
            return self.store.search_person_by_image(self.image_path, args.get("k", 5))
        return {"error": f"unknown tool {name}"}

    def _system(self) -> str:
        info = self.store.run_info()
        return ("You answer questions about multi-camera people-tracking data by calling "
                "the provided tools, then writing a concise prose answer with the concrete "
                "numbers. Identities are integer global_id; areas are named zones "
                f"(e.g. cam0:ZONE). Run context: {json.dumps(info)}. Available zones: "
                f"{self.store.zones()}. If the user references 'this person' and an image was "
                "provided, call search_person_by_image first, then use the top global_id. "
                "Resolve relative times to ISO datetimes within the run's epoch.")

    def ask(self, question: str, image_path: str | None = None, max_steps: int = 6) -> dict:
        """Run the tool-use loop and return {'answer': str, 'tool_calls': [...]}"""
        import anthropic
        if image_path:
            self.image_path = image_path
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": question}]
        calls = []
        for _ in range(max_steps):
            resp = client.messages.create(model=self.model, max_tokens=1024,
                                          system=self._system(), tools=TOOLS, messages=messages)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return {"answer": text, "tool_calls": calls}
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = self.dispatch(b.name, b.input or {})
                    calls.append({"tool": b.name, "args": b.input, "result_preview": str(out)[:200]})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, default=str)})
            messages.append({"role": "user", "content": results})
        return {"answer": "(stopped: max tool steps reached)", "tool_calls": calls}

    def close(self):
        self.store.close()
