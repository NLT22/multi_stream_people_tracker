# Production Readiness & Roadmap

Live roadmap for the narrow production system: YOLO11 detector, NvDCF tracker,
SGIE Swin ReID, live-buffered MTMC. Archived research/training/prototype files
live under `old_stuff/retired_20260620/` — do not restore unless they re-enter
the production system.

> Section 5 (Natural-Language Q&A / RAG Layer) is the active new workstream
> (mentor requirement, 2026-06-26). Sections 0–4 are the existing system + open
> hardening items. This file was condensed on 2026-06-26 — verbose experiment
> dumps were summarized to their commands + verdicts; nothing actionable was dropped.

---

## 0. Current Production System

Target: 20 cameras, ≥10 FPS/cam, mean IDF1 ≥ 0.8 on the 640×360 mixed validation set,
production-style buffered/global IDs (not offline-only scoring).

Architecture:

```text
video files or RTSP
  -> DeepStream / pyservicemaker
  -> YOLO11 PGIE detector  -> NvDCF tracker  -> SGIE Swin-Tiny ReID on person crops
  -> PredictionExporter (cam CSV + det_emb_chunk_*.npz)
  -> src.mtmc.live_buffered (groups cameras by environment) -> IDF1/stability logs
```

Presets:

```text
DEFAULT (reid0):  configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml
quality (reidType:2): configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml
models: nvinfer_yolov11_mmp.yml (→ yolo11n_mmp_retailclean.onnx, deployed 2026-06-26;
  old yolo11n_mmp.onnx kept for rollback), nvinfer_reid_swin_sgie_all.yml
tracker: nvdcf_accuracy_mmp_recall_sgie[_reid0].yaml
live_buffered default: assign_thr=0.50 (tuned from 0.40, 2026-06-26)
```

Latest honest result — single-pass full-GT (every frame once, no loop/GT-trimming;
score with `scripts/eval/score_full_mmp_val.py` after `live_buffered --once`):

```text
Full val (24 scenes, buffered, reid0, retail-clean detector, assign_thr=0.50): mean IDF1 0.798
  lobby 0.906 | office 0.880 | industry 0.847 | café 0.839 | retail 0.661
  (4/5 envs ≥0.8; excluding retail the 16-scene mean is 0.866. Baseline DMCT-Ext-TD = 0.741.)
  per-camera: precision 0.94, MOTA 0.768, ~46 ID-switch/cam.
  (prior, old detector @ assign_thr=0.40 was 0.774 — superseded 2026-06-26.)
reid0 vs quality tie on IDF1 — global IDs come from SGIE embeddings, so NvDCF internal
ReID adds ~0. reid0 is default (leaner).
```

VRAM/throughput are driven by `maxTargetsPerStream` (NvDCF pre-allocates per-target
state for `maxTargetsPerStream × streams`), NOT the model. reid0@40 ≈ 3.4 GB / ~15 FPS/cam;
@150/175/200 ≈ 7.0/7.9/8.7 GB / 12.0/11.5/10.6 FPS; @220 ≈ 9.2 GB / 10.6 FPS — higher values
buy no quality on MMP (≤~12 ppl/cam). Default is `maxTargetsPerStream=40`.

Known weakness: **retail** (IDF1 0.661) is the quality limiter. Its phantom-box false-positive
root cause was FIXED (2026-06-26 retail-clean detector: precision 0.62→0.94, ID-switch −50%).
The remaining limit is **recall** — real people fully occluded behind shelves are missed; a
physical CCTV limit, not fixable in post-processing. Real production resolution is expected to
be 1920×1080, but the repo must keep passing 640×360 first.

---

## 1. Production Commands

```bash
# cheap non-GPU wiring check
scripts/setup/production_smoke.sh

# 20-cam eval (default reid0 preset)
PIPECFG=configs/pipelines/pipeline_mmp_nvdcf_online_sgie_reid0.yaml \
  bash scripts/eval/run_long_eval.sh 600 configs/sources/val_20cam_mixed.txt \
  "cafe:0-3,lobby:4-7,office:8-11,industry:12-15,retail:16-19"

# RTSP loop / multi-env cycling
scripts/eval/mediamtx_loop.sh start dataset/MMPTracking_10minute/val/64pm_office_0
scripts/eval/mediamtx_multienv.sh start dataset/MMPTracking_10minute/val

# persist a run to SQLite (the store the RAG layer will query — see §5)
python scripts/eval/persist_run.py --run-dir output/runs/<ts>_<preset> --db output/runs.sqlite
```

---

## 2. Done (high level)

- Cleaned root to the production path; archived training/conversion/prototype code (reversible).
- reid0 default; smoke script; run-dir + `run_manifest.json`; config validator (preflight).
- Live NPZ embedding export (uncompressed, final-chunk flush); env-grouped `live_buffered`.
- Honest single-pass full-GT eval (24 scenes); reid0/quality tie confirmed.
- **Retail-clean detector** (`yolo11n_mmp_retailclean.onnx`) retrained + deployed both presets
  (full-val 0.774→0.798, precision 0.62→0.94); `assign_thr` tuned 0.40→0.50; demos regenerated
  with real DeepStream OSD; fixed `GROUPS` bash-builtin footgun in demo scripts. (2026-06-26)
- RTSP smoke (5×office, reid0): clean source-ids, chunks flush, ~15 FPS/cam, 3.1 GB/5cams.
- Bounded 25-min soak: FPS/VRAM stable, GID plateau, no leak.
- Retail diagnosis: local identity (switches/frags), not cross-cam.
- Append-only SQLite sink (`scripts/eval/persist_run.py`) + run summarizer (`summarize_long_run.py`).
- MOTA/HOTA per-camera metrics saved to `metrics.json`.

---

## 3. Open Hardening Items

### 3.1 RTSP / stability
- [ ] Scale RTSP smoke to full 20 streams (multienv) and confirm ≥10 FPS/cam.
- [ ] 2h file-loop soak; overnight RTSP/file-loop soak (watch RSS creep, GID growth).

### 3.2 Exact-source end-to-end
- [ ] Add exact-source end-to-end IDF1 eval feeding official zip frames to DeepStream
  (generated videos/RTSP or image-sequence source). Keep the 10-min video benchmark passing.

### 3.3 Retail quality
- [x] Root cause = detector false positives (phantom boxes on shelves/mannequins), NOT identity.
  Fixed by retraining YOLO on cleaned retail labels (precision 0.62→0.94, retail IDF1 0.616→0.661,
  ID-switch −50%); deployed 2026-06-26. `assign_thr` tuned 0.40→0.50 (+0.003 full-val, AssA/HOTA up,
  DetA/MOTA unchanged — pure association gain).
- [ ] Remaining gap is **recall** under heavy shelf occlusion (physical). If pursued: higher-res
  input (1080p), camera placement, or amodal-aware detection — not post-processing.
- Do-not-retry: geometry on the scored `live_buffered` path (STCRA / geo-merge / geo-split) all
  REGRESS on MMP (overlapping FOV makes single-cam foot projection noisy). The static FP filter
  once used for retail is now redundant (detector is clean).

### 3.4 Docker
- [ ] `scripts/setup/docker_smoke_test.sh --build`; `docker compose run --rm tracker`.
- [ ] Confirm generated TensorRT engines stay local (not committed); document GPU/driver/TRT versions.

### 3.5 Pipeline audit findings (2026-06-24, static audit — no code changed)

🔴 Correctness
- [ ] Display sink missing `async: 0` (`runner.py` ~L401) — can deadlock with tee/dynamic RTSP.
- [ ] Mux defaults 1920×1080 for 640×360 sources (`run_config.py`) — 9× upscale wastes VRAM/bandwidth.

🟠 Perf / config
- [ ] SGIE `interval: 0` (`nvinfer_reid_swin_sgie_all.yml`) — try `interval:1/2`, measure IDF1 delta.
- [ ] `minDetectorConfidence: 0.12` dead zone vs nvinfer 0.25 — align to 0.25.
- [ ] `maxTargetsPerStream: 220` over-provisioned in quality preset — set 30–40.
- [ ] `earlyTerminationAge: 1` too aggressive vs `maxShadowTrackingAge: 240` — make consistent (≥10 / 60–90).
- [ ] `geo_weight` not in pipeline YAML → defaults 0.35; add explicit `geo_weight: 0.25`.

🟡 Hygiene
- [ ] Engine batch mismatch (built batch=4, runtime batch=n_cams) — pre-build at target cam count.
- [ ] Quality preset double-loads Swin (~0.4 GB) — prefer reid0 when VRAM-constrained.
- [ ] SGIE `maintain-aspect-ratio: 0` stretches crops — try `1` and re-eval.
- [ ] `minIouDiff4NewTarget: 0.50` may suppress new targets in dense scenes — consider 0.35–0.40.

---

## 4. Model Work (detector + ReID) — summary & verdict

Detector: **`yolo11n_mmp_retailclean.onnx` is the deployed detector (2026-06-26).** Retrained
30 epochs on hybrid labels = retail labels cleaned with a COCO-YOLO11x verifier + TTA (removes
~42% phantom/amodal retail boxes) while all other envs keep their labels. Internal mAP50 0.99;
full-val precision 0.62→0.94, MOTA 0.64→0.77, recall 0.874→0.825 (accepted cost). Cleaning ALL
envs was REJECTED (verifier over-removed partially-occluded real people → recall loss in
lobby/office); retail-only is the chosen tradeoff. Old `yolo11n_mmp.onnx` kept for rollback.

Root cause that was fixed: original MMP GT used **amodal projection** (each person's 3D box
projected into every camera even when fully occluded → teaches "background = person"), and the
train set had **0 negative images**. `clean_yolo_labels.py` (verifier) existed but had never been
applied to the deployed detector; `mmp_to_yolo.py` only drops out-of-frame boxes, not occluded ones.

Prior baseline (old detector, for reference): precision 0.965/recall 0.893/mAP50 0.957 on
exact-source frames — superseded by the retail-clean model above.

ReID retrieval (corrected per-scene gallery, balanced 40/scene-cam — the earlier
global-gallery numbers were understated by an eval bug pooling all 24 scenes):

```text
deployed swin_tiny_mmp_reid_all : top1 0.847 / mAP 0.773  <- KEEP
retrained full_env_envmerge_e20 : top1 0.729 / mAP 0.546
```

Verdict (stable across every retrain — exact-source, regrouped, YOLO11x-verified):
**keep `models/reid/swin_tiny_mmp_reid_all.onnx`; do not promote any ImageNet-Swin retrain.**
Retail has the largest crop-quality problem (confirmed by YOLO11x crop audit), but more
training from ImageNet is not the shortest path — recover the original trainable Swin
checkpoint or add a distillation/fine-tune-from-deployed-embeddings path first.

Tools (kept): `scripts/datasets/mmp_exact_to_{yolo,reid}.py`, `eval_{yolo,reid}_mmp_exact.py`,
`finetune_reid_mmp_exact.py`, `filter_reid_crops_yolo.py`. Manual label workflow + exact-source
relabel UI (`reid_label_app_exact.py` → `reid_labels_exact/`) documented in git history / scripts.
Key correction: ReID training must use regrouped identity labels (`env::manual_person`,
14 ids/env); raw zip `person_id` resets per scene.

---

## 5. Natural-Language Q&A / RAG Layer  (NEW — mentor requirement, 2026-06-26)

### 5.1 Goal

Let a user ask questions about the tracking/analytics data in two modes:

1. **Person-centric (image + question)** — upload a crop of a person, ask
   "when did this person appear?", "at 10am which areas did they pass through?".
   System returns: which cameras, appearance times, dwell, and the BEV trajectory
   history for that person (same artifacts the demo already renders).
2. **Aggregate analytics (text only)** — "which shelf got the most attention today?",
   "top-5 busiest areas this week?". System returns ranked zones/timeseries.

### 5.2 Recommended architecture — a tool-using LLM agent over our metadata

We deliberately do **not** copy NVIDIA VSS's VLM-captioning pipeline (VSS captions raw
video chunks because it starts from pixels). Our CV pipeline already emits *structured*
identity + trajectory metadata, so the cheaper, more accurate design is **structured-metadata
RAG**: an LLM router that calls deterministic retrieval tools and composes the answer.
This still follows the VSS principle (agentic tools over CV metadata) and the standard
text-to-SQL + ReID-vector-search patterns (see refs in 5.7).

```text
                         ┌─────────────────────────────────────────────┐
  user question  ─────►  │  LLM router (function-calling / tool-use)    │
  (+ optional image)     │  picks a tool, fills params, composes answer │
                         └───────┬─────────────────────────┬───────────┘
                                 │                          │
              ┌──────────────────▼─────┐      ┌─────────────▼───────────────┐
   ROUTE A    │ Aggregate analytics    │  R B │ Person image search          │
   (text)     │  - canned analytics    │(img) │  - embed crop (Swin ONNX)    │
              │    functions (safe)    │      │  - cosine search over saved  │
              │  - text-to-SQL fallback│      │    gallery -> global_id(s)   │
              │    (read-only, guarded)│      │  - then Route-A queries on gid│
              └──────────┬─────────────┘      └─────────────┬───────────────┘
                         └────────────┬─────────────────────┘
                                      ▼
                    SQLite (persist_run.py) + saved embedding gallery + named-zone registry
```

- **Route A (aggregate):** prefer a small set of **parameterized "analytics functions"**
  (e.g. `top_zones(time_range, metric)`, `zone_dwell(zone, time_range)`) — deterministic,
  testable, safe. Add **text-to-SQL** only as a fallback for open-ended questions, with
  read-only guardrails (SELECT-only, schema-grounded prompt, parameterized, row limits).
- **Route B (person):** embed the uploaded crop with the **deployed Swin ONNX**, cosine-rank
  against a **persisted gallery** (reuse `src/reid/matching.py` + `GalleryStore.rank`) to get
  candidate `global_id`s, then answer the "when/where/dwell/trajectory" parts via Route-A
  queries scoped to that gid.
- **LLM:** Claude via the Anthropic Messages API tool-use (Sonnet for routing/most queries,
  escalate to Opus for multi-step reasoning); a local open model is the on-prem fallback.
  The LLM never touches raw embeddings — it only orchestrates tools and writes prose.

### 5.3 Data gaps to close first (Phase A — foundation)

These are reconstructible from existing exports; all land in SQLite. Build offline/batch
from current run artifacts — low risk, high reuse.

- [ ] **Wall-clock timestamps.** Detections store `frame_no` per cam, not time. Add a run
  epoch + fps → `ts` column (or capture capture-time), so "today / 10am / this week" resolve.
- [ ] **Named-zone registry + foot→zone resolver.** Reuse the webUI **ROI editor** (it already
  emits named regions in `nvdsanalytics_*.txt` format) as the zone vocabulary. Add a resolver
  mapping a foot point (per-cam pixel ROI, or BEV world XY via `geometry.foot_to_world`) → zone
  name. This is what makes "which shelf/area" answerable.
- [ ] **Derived analytics tables** in SQLite (built from `detections` + `tracklet_bev` + zones):
  - `presence(gid, cam_id, zone, t_start, t_end)` — per-visit intervals.
  - `dwell(gid, zone, seconds, day/hour bucket)` — Occupancy/Footfall already exist as grids;
    add per-gid/per-zone dwell from foot points + timestamps.
  - `zone_occupancy(zone, time_bucket, count)` / `zone_footfall(zone, time_bucket, unique_gids)`
    timeseries — powers "busiest zone today / top-5 this week".
- [ ] **Persisted embedding gallery for query-time search.** Persist per-gid (and/or per-tracklet)
  mean embeddings (already in `tracklet_embeddings.npz`) into the DB / a vectors table so an
  uploaded image matches without re-running the pipeline. Start with brute-force cosine
  (gallery is small); add a vector index only if it grows (5.6).

### 5.4 Phase B — Query API (deterministic core, no LLM yet)

- [ ] Stand up a small **FastAPI** service (the missing backend; the webUI is currently static).
- [ ] Implement the analytics functions as JSON endpoints/tools, each independently testable:
  - `person_timeline(gid, time_range)` → cameras + appearance intervals.
  - `person_trajectory_bev(gid, time_range)` → BEV foot-point path (from `tracklet_bev`).
  - `person_dwell(gid, time_range)` → per-zone dwell.
  - `top_zones(time_range, metric)` / `zone_occupancy(zone, time_range)`.
  - `search_person_by_image(image)` → candidate gids + scores (embed + cosine).
- [ ] Unit tests on a persisted demo run (golden answers) — this is the correctness gate;
  the LLM layer must not be the thing under test.

### 5.5 Phase C — LLM agent + Phase D — WebUI

- [ ] **Agent layer:** wire the Phase-B tools as function-calling tools; the LLM selects a
  tool, fills params (resolve "10am today" → time range; resolve image → gid), and writes the
  answer + a structured payload. Add text-to-SQL fallback with the read-only guardrails (5.2).
- [ ] **WebUI "Ask" view** (7th nav entry; React/Vite, `webui/src/components/rag/`): chat box +
  image-upload; render results as a timeline, a BEV trajectory overlay (reuse heatmap/BEV view),
  camera jump-links, and a heatmap time-window. Add `webui/src/api/` fetch wrappers.
- [ ] Start replacing the webUI's mocked `src/data/*` with the same FastAPI endpoints (the
  README already documents these integration seams).

### 5.6 Phase E — optional scale-out

- [ ] Vector DB (pgvector / Milvus) once the gallery is large or multi-run.
- [ ] Knowledge-graph / GraphRAG for relationship queries ("who was with X", "co-occurrence")
  — VSS 2.4 uses GraphRAG-on-ArangoDB for exactly this class of long-form query.
- [ ] Live ingestion (message bus → online metadata) for real-time Q&A instead of batch persist.

### 5.7 Honest caveats / decisions needed

- **Timestamps & zones are prerequisites, not optional.** "Which shelf at 10am" cannot be
  answered until 5.3's timestamp + named-zone work lands. The MMPTracking eval set has no
  wall-clock and no named shelves — define zones via the ROI editor per environment first.
- **Retail accuracy caps person-search quality** there (IDF1 0.661, the lowest env); the detector
  phantom-box issue is fixed, but recall under shelf occlusion remains, so image search will be
  least reliable in retail.
- **Scope question for the mentor:** is this a demo over *recorded eval runs* (batch persist —
  fastest to ship) or must it run *live* (needs 5.6 ingestion)? Recommend batch first.

### 5.8 Reference systems (patterns borrowed)

- NVIDIA VSS / Metropolis blueprint — agentic tools over CV metadata; GraphRAG for relationships.
  https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization ,
  https://docs.nvidia.com/vss/latest/content/architecture.html
- Text-to-SQL LLM agents — schema-grounded prompts, agentic validate/refine, read-only guardrails:
  https://www.k2view.com/blog/llm-text-to-sql/
- Multi-camera ReID image search — embedding gallery + cosine/Euclidean nearest-neighbor:
  https://hailo.ai/blog/multi-camera-multi-person-re-identification/

---

## 6. Future Scale-Out (do not build until single-host is stable)

```text
DeepStream perception -> message bus (Redis Streams; Kafka only if multi-host)
  -> MTMC service -> TimescaleDB/Postgres (+ pgvector/Milvus if embedding search)
  -> dashboard/API
```

Storage today: SQLite/CSV/NPZ for eval; TimescaleDB/Postgres for production metadata;
pgvector only if ReID search becomes a product requirement (it now is — see §5).

---

## 7. Open Questions

- Tolerated Global ID latency: near-live, 10 s, 30 s, longer?
- Single-host 20-cam final, or multi-host?
- Real input all RTSP at 1920×1080?
- Is retail IDF1 a hard requirement, or is mean IDF1 the acceptance gate?
- **RAG:** batch (recorded runs) or live? Which zones/shelves must be named per environment?
  Acceptable answer latency / LLM hosting (Anthropic API vs on-prem)?
