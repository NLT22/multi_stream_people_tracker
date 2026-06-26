"""Natural-language Q&A / RAG layer over MTMC tracking metadata (production_todo §5).

Phase A  ingest.py   build the SQLite store (timestamps, zones, derived tables, gid embeddings)
Phase B  queries.py  deterministic query functions (the correctness gate)
         api.py       FastAPI service exposing them
         embed.py     crop -> Swin ONNX embedding for image search
Phase C  agent.py     Anthropic tool-use router that composes prose answers
"""
