# Production Readiness TODO

Roadmap to run the multi-stream tracker like production + long-duration (days)
stability eval. Organized in **data-flow order**: Overview → Input → Core
(MTMC) → Storage → Analytics → UI → Validation.

---

## 0. Overview & architecture decision

### The core tension
The **offline anchor-guided** result (retail ~0.78, lobby ~0.95) **cannot run on a
live stream** — it needs the whole clip + dense per-crop ReID. So pick a deployment
mode that trades latency for how close to offline you can get:

| Mode | Latency | Mechanism | Accuracy | Deployable? |
|---|---|---|---|---|
| **Online** (current `gallery.py`) | 0 (realtime) | greedy running gallery, irreversible | lowest (~0.45 retail regime) | yes |
| **Near-online / windowed** | ~8 s | `nearline_merge.py`, sliding frame-window | middle (nearline anchors: lobby_0 0.84, industry 0.84 @125 frames) | yes |
| **MDX-style micro-batch** ⭐ | batch interval (10–30 s) | cluster **tracklets** (not frames) + cross-batch state | near-offline | **yes — recommended** |
| **Offline anchor-guided** | whole clip | global banks + Hungarian + dense ReID | best (0.78/0.95) | no |

**Recommendation: the MDX-style micro-batch architecture (§3)** — it runs the
anchor-guided clustering as a decoupled service on tracklets, so it streams without
being online-greedy, and lands near offline quality at bounded latency.

### End-to-end target
```
MediaMTX (loop/live -> RTSP)               [§1 Input]
   -> DeepStream perception (detect+SCT+ReID), publish per-tracklet to bus
   -> micro-batch MTMC service (anchor clustering + cross-batch state)   [§2 Core]
   -> DB (tracklets + global IDs + embeddings)                          [§3 Storage]
   -> spatial analytics (routes / entry-exit / zone counts)             [§4]
   -> UI: zone editor + Grafana ops + Streamlit analyst views           [§5]
   (cross-cutting: §6 days-long stability eval)
```

---

## 1. Input — RTSP via MediaMTX

Pipeline **already supports `rtsp://`** — `src/pipeline/sources.py` detects rtsp,
sets `live-source=1` on mux + `sync=0` on sink. No code change to consume.

- [ ] Run MediaMTX: `docker run --rm -it -p 8554:8554 bluenviron/mediamtx`
- [ ] Loop each camera into an RTSP path (forever, native fps):
  ```bash
  ffmpeg -re -stream_loop -1 -i <scene>/cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
  # repeat cam2..camN
  ```
  - `-re` = real-time pacing (simulate live camera); `-stream_loop -1` = loop forever; `-c copy` = no re-encode.
- [ ] Point the pipeline at the RTSP list:
  ```bash
  python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml \
    --sources rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2 ... \
    --no-display --no-sync
  ```
- [x] **Launcher DONE**: `scripts/eval/mediamtx_loop.sh start <scene_dir> [port]` / `stop` — starts MediaMTX (docker, idempotent) + one looped real-time ffmpeg per camera, prints the `rtsp://` pipeline command; `stop` tears down. Validated (syntax + args + ffmpeg present); needs docker to actually run.

---

## 2. Core — MDX-style micro-batch MTMC

Deploy the offline anchor-guided method without forcing it online/nearline. NVIDIA
Metropolis MTMC
([overview](https://docs.nvidia.com/mms/text/MDX_Multi_Camera_Tracking_MS_Overview.html))
shows the pattern: per-camera perception publishes tracklets+embeddings to a message
bus; a **decoupled MTMC microservice** does **micro-batch** fusion via **hierarchical
clustering + Hungarian reassignment** with cross-batch state — the same algorithm
family as our anchor-guided. We run *our* clustering in *their* architecture, batching
**tracklets** (not frames).

### Component mapping (MDX → ours)
| MDX component | MDX tech | Ours | Build |
|---|---|---|---|
| Perception | DeepStream → `mdx-raw` Kafka (protobuf): SCT + ReID | `src.main` (YOLO FP32/nms0.7 + NvDCF + OSNet/Swin) | publish per-tracklet records to bus (not CSV) |
| Behavior State Mgmt | live behaviors across micro-batches | NEW | persistent anchor banks + live-tracklet buffer across windows |
| Behavior Processing | georeference + filter | `geometry.foot_to_world` + quality filters | tracklet → world foot + filter |
| Multi-Camera Tracking | hierarchical clustering + Hungarian, micro-batch | `src/eval/offline_anchor_faithful.py` | refactor to incremental per-batch clustering |
| Merging IDs | global IDs across sensors | anchor cluster ids / gallery GIDs | persist global-id ↔ tracklet map |
| Storage | **Elasticsearch** (metadata) + **Milvus** (ReID embeddings) + Kafka/Logstash | §3 DB | metadata + vectors + global-id store |

### Per-tracklet message schema (perception → bus)
```
{ sensor_id, tracklet_id, t_start, t_end,
  frames: [{frame, left, top, w, h, conf}],
  foot_world: [x_mm, y_mm],            # via geometry
  embedding: float[256|512],           # PER-TRACKLET mean (or k samples), NOT dense per-crop
  status: "live" | "done" }
```
> Per-tracklet (not per-detection) embedding is the throughput unlock for 20cam@10FPS
> — dense per-crop is what made the offline path un-deployable.

### Build checklist
- [x] **Incremental MTMC clustering core** — archived after validation. The old `IncrementalMTMC` + `run_incremental` simulator proved the tracklet-mean path was not enough for the target, so it was moved to `old_stuff/retired_20260620/src/mtmc/`. The active production consumer is `src.mtmc.live_buffered`.
- [ ] **Cross-batch anchor persistence** — partial: banks persist in-process via TTL/max-anchors; still need on-disk persist/reload for a long-running service.
- [ ] **Perception publisher**: in `CrossCameraGalleryProbe` / `SourceIdCollectorProbe`, accumulate per-tracklet (boxes + mean embedding + world foot) and publish to a bus (Kafka, or Redis Streams for a lighter start) on tracklet-complete / periodic flush. (Currently the sim harness reads `detection_embeddings.npz`; swap for a live bus consumer.)
- [~] **Close the 0.60→offline gap**: per-detection mode added (`--per-det`, with bank consolidation + union-find gid aliasing). office_0: per-det+consolidate_thr0.10 = **0.71** (vs tracklet-mean 0.60, offline 0.836) — recovers much of the gap but is **threshold-sensitive** (over-spawn ↔ collapse). TODO: auto/robust consolidate threshold, k prior.
- [x] **Per-tracklet k-crop BANK vs mean tested**: bank (k=8/16, max-pool cost) = 0.60/0.61 ≈ mean 0.60 — barely helps. Lever is **assignment GRANULARITY (per-detection, 0.71), not embedding aggregation**. `--bank-k` knob added (mean = bank-k=1).
- [x] **k/concurrency prior (hard cap) tested — FAILS**: forcing #anchors down to concurrency-floor k=7 by greedy closest-centroid merge fuses *different* people → office_0 **0.41** (ct0.10) / 0.17 (ct0.20), far worse than threshold-only **0.71**. Greedy streaming merge ≠ offline's global clustering. Made `--num-people` cap **opt-in** (default off; documented to hurt).
- **ONLINE CROSS-CAMERA CEILING (greedy): ~0.71** (per-det + threshold consolidation, no hard cap) on office_0.
- [x] **NEAR-OFFLINE buffered re-cluster — WINS** (`--buffered --window-frames W`): each window re-runs the OFFLINE clustering (`build_anchors`+`assign_per_frame`) from scratch + stitches windows by **k-to-k Hungarian on cluster centroids** (NOT tracklet-union, which collapsed to 1 id). office_0: **W=450 (30s) → 0.826, W=900 (60s) → 0.823**, W=9999 (whole-clip) → 0.836 = offline (confirms impl). vs greedy 0.71. **GENERALIZES** (W=450, vs whole-clip offline): office 0.826 (−0.01), retail 0.665 (−0.01), **industry 0.678 vs 0.621 (+0.06 — windowing *limits* OSNet's global over-merge of uniform look-alikes)**. **Recovers ~98–100%+ of offline at 30s latency, all envs.** Recommended deployable cross-camera path when latency budget allows ~30–60s. Knobs: `--window-frames`, `--window-step`, `--assign-thr`, k via `--num-people`/`--oracle-k`/concurrency-floor.
- [x] **buffered + SWIN (one-model) — FINAL deployable recipe**: re-ran with Swin per-det npz (`swin_reid_embed.py` → buffered). office_0 W=450 **0.859** (vs OSNet 0.826), industry_0 W=450 **0.875** (vs OSNet 0.678, **+0.197**); W=9999 = offline-Swin exactly (0.880 / 0.897 ✓). **buffered-Swin@30s recovers ~97–98% of offline-Swin, far above greedy 0.71.** Production cross-camera path = **buffered near-offline (W≈30s) + one Swin FP16 model.**
- FPS: MTMC stage is **not** a bottleneck — tracklet-mean ~25,000× realtime, per-det ~25–37× realtime; perception (GPU) is the only throughput limit.
- [ ] **Batch interval** config (10–30 s) = latency knob; tune accuracy vs latency.
- [ ] **Option B fallback**: feed our DeepStream `mdx-raw` into NVIDIA's shipped MTMC microservice and use *their* hierarchical clustering (less control, faster bring-up).

### Caveats
- Latency = batch interval (by design); accuracy lands **between online-greedy and offline 0.78/0.95** — recovers association/latency, but per-tracklet (vs dense) embedding costs some accuracy.
- Uniform-clothing ReID generalization (industry/retail) still caps those envs regardless of architecture (report 17/06).
- Resolves **online≈offline** + **throughput**; does NOT fix **ReID generalization**.

---

## 3. Storage — Database

### Decide what to persist (drives the choice)
- [ ] Per-frame / per-tracklet tracks: `(ts, cam_id, frame, local_id, global_id, left, top, w, h, conf)` — very high rate, append-only time-series.
- [ ] ReID gallery / anchor state: `global_id -> mean embedding` — low rate, mutable, vector.
- [ ] Events/analytics: entry/exit, dwell, per-zone occupancy — low rate, relational.

### Choice
- [ ] **Start (this eval): SQLite** — single file, zero ops, fine for single-host days-long runs (100M+ rows). Or hourly **Parquet**.
- [ ] **Production single-host: TimescaleDB (Postgres ext)** — hypertables + auto time-partitioning + retention policies, `pgvector` for embeddings, continuous aggregates for hourly rollups — **one Postgres for metadata + vectors + analytics** (simplest ops).
- [ ] **Production at scale / full-MDX (Option B): match MDX's split** — **Elasticsearch/OpenSearch** for metadata+search (queried by a Web API) + **Milvus** for the ReID embedding vectors (purpose-built for billion-scale ANN, beats pgvector at that scale) + Kafka/Logstash. Three services to operate.
- [ ] Reject: MongoDB/Cassandra (wrong access pattern), InfluxDB (second system for embeddings/events).
- Note: all of SQLite / Postgres+Timescale / OpenSearch / Milvus are **free to self-host**; Elasticsearch is free on the Basic tier (AGPLv3), paid only for advanced features / Elastic Cloud.

### Implementation
- [x] **Batched DB sink prototype** — archived under `old_stuff/retired_20260620/src/storage/`. It proved SQLite batch insert was viable, but it is not wired into the current production pipeline.
  - [ ] TODO: live path = call `add_track`/`add_zone_event` from the gallery probe / MTMC (currently offline ingest from CSV). Apply **min-dwell debounce to zone-enter events** (current ingest emits raw per-frame → 76k jittery enters; should reuse `zone_analytics` debounce). Schema maps to TimescaleDB hypertable on `ts`; add `gallery` (embedding vector) table + retention policy for multi-host.

---

## 4. Analytics — spatial analytics (mentor's asks)

Required: (a) **common routes between zones**, (b) **most-used entry/exit points**,
(c) **people count per zone over time**.

### What already exists (build on it)
`scripts/eval/trajectory_analytics.py` (world ground-plane):
- `journey_map.png` — per-identity polyline (routes, raw)
- `od_matrix.png/.csv` — zone-to-zone transition counts (flow between zones)
- `time_in_zone.csv` — seconds per global_id per zone
- `dwell_map.png` — occupancy heatmap

**Gaps**: zones are an **auto GxG grid** (not semantic named zones); no entry/exit ranking; no time-bucketed throughput.

### Zone space (design decision)
- [ ] **Semantic named zones** (entrance, checkout, aisle-1, exit-door…) as polygons in `configs/zones/<scene>.json`.
- [ ] **World (ground-plane) zones** to unify across cameras. Projection is noisy (~270 mm) **but coarse zones tolerate it** — the *good* use of geometry here (unlike fine STCRA). Fallback: per-camera image-space polygons.
- [ ] Point-in-polygon assignment per detection (foot point / `tracklet_bev.csv`).

### (a) Routes between zones
- [ ] Per `global_id`: ordered zone-visit sequence (collapse consecutive same-zone → `A→B→C`).
- [ ] Aggregate → most-frequent n-gram paths (not just pairwise OD).
- [ ] Output `routes_top.csv` (path, count, avg duration) + flow map (zone graph, edge width ∝ flow).

### (b) Entry/exit points
- [ ] Tag door zones in config, OR auto: track's first-seen zone = entry, last-seen = exit.
- [ ] Rank → `entry_exit_ranking.csv`. Guard fragmentation (require near a door zone / min-track-length so re-entries don't double-count).

### (c) People per zone over time
- [ ] Time-bucket (per minute) × zone → unique global_ids (occupancy) + entries (throughput).
- [ ] Output `zone_occupancy_timeseries.csv` (ts_bucket, zone, n_unique, n_enter). Natural TimescaleDB continuous aggregate over the §3 track stream.

### Implementation steps
- [x] `src/analytics/zones.py` prototype — archived under `old_stuff/retired_20260620/src/analytics/`. Restore only when implementing the real analytics product path.
- [x] `scripts/eval/zone_analytics.py` — DONE: consumes `tracklet_bev.csv` (world foot + global_id) → `routes_transitions.csv`, `routes_top.csv`, `entry_exit_ranking.csv`, `zone_occupancy_timeseries.csv` (n_unique + n_enter) + `flow_map.png`. **`--min-dwell` debounce** (default 15f) kills ground-plane boundary jitter (essential — raw gave 43 spurious entries/bucket). Model-free/CPU. Tested on `64pm_office_0`.
  - **Lesson:** auto-grid zones produce oscillating routes (people straddle arbitrary cell boundaries) → **needs SEMANTIC zones (§5 editor) + min-dwell**; coarse 2×2 + 3s dwell already gives sane throughput. Quality bounded by GID stability (held-out caveat) + ground-plane noise → keep zones coarse/semantic.
- [ ] Live path: zone-assign in the gallery probe / MTMC → write zone events to DB.

### Caveats
- [ ] Bounded by **GID stability + recall** — fragmented GIDs double-count routes/entries. (Confirmed: held-out `64pm_retail_0` had 3.3× more ID-switches than train → analytics over-count more on unseen data.) Fix tracking quality first; sanity-check against GT.
- [ ] Keep zones **coarse** — robust to ground-plane noise; fine-grained position analytics are not.

---

## 5. UI (mentor: "interactive UI will be good")

Three pieces, separate lifecycles:
- [ ] **Zone editor** — browser canvas: load a camera frame / top-down map, click polygons, name+tag (entry/exit/aisle/checkout), save `configs/zones/<scene>.json`. The old label-app prototype is archived under `old_stuff/retired_20260620/scripts/datasets/`; reuse only if needed.
- [ ] **Grafana + TimescaleDB** — ops/time-series panels: people-per-zone-over-time, throughput, occupancy + GID-count/FPS health for the days-long run. Live, no code.
- [ ] **Streamlit** — analyst views Grafana can't do: flow map, top-routes table, entry/exit bar charts, journey replay.
- [ ] (optional) Live zone overlay + per-zone counts on the OSD video (extend `src/reid/visualization.py`).

---

## 6. Validation — long-duration (days) stability eval

Mentor's ask: loop video for days. Validates **runtime stability/drift**, NOT accuracy
(looped GT repeats → meaningless IDF1; keep accuracy on held-out clips).

### What to watch
- [ ] **Global-ID growth**: does `total_gids_ever_assigned` plateau or climb forever (gallery leak)? (Today retail ~13.)
- [ ] **Memory/VRAM creep**: RSS + `nvidia-smi` flat over time.
- [ ] **FPS stability** over days.
- [ ] **ID drift across loop boundaries**: same person re-entering as new GID = gallery not persisting.
- [ ] **DB write lag** keeps up with track rate.

---

## Suggested order
1. [ ] **Held-out accuracy first** — full-val (`64pm_*`) offline numbers as the honest benchmark (in progress) before building plumbing on them.
2. [ ] **Input** (§1): MediaMTX loop launcher + run pipeline on RTSP — verify it consumes rtsp and stays up.
3. [ ] **Storage** (§4): batched SQLite sink in the gallery probe.
4. [ ] **Validation** (§7): days-long run; watch GID plateau / VRAM / FPS / DB lag.
5. [ ] **Analytics** (§5): named zones → routes / entry-exit / occupancy (extend `trajectory_analytics.py` offline; then live-in-DB).
6. [ ] **UI** (§6): zone editor → Streamlit + Grafana.
7. [ ] **Core** (§3): MDX-style micro-batch MTMC — the production form of the offline pipeline (resolves online-vs-offline + throughput). Biggest build; do once the rest is proven.
8. [ ] **Productionize**: Timescale + pgvector, retention, dashboards.

## Open questions
- [ ] Is online (greedy) accuracy acceptable, or is near-online / MDX micro-batch required for the use case? (See §0 modes table.)
- [ ] Single-host (one GPU, RTX 5060 Ti 16GB) vs multi-host scaling target for the camera count.
- [ ] Batch interval (§3) vs accuracy: what latency does the use case tolerate?
