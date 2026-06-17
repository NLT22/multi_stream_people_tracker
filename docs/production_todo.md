# Production Readiness TODO

Roadmap for running the multi-stream tracker like production + long-duration
(days) stability evaluation. Three tracks: **Database**, **Long-run eval**,
**RTSP via MediaMTX**.

> Key caveat that shapes everything below: the **offline anchor-guided** result
> (retail ~0.78, lobby ~0.95) **cannot run on a live RTSP stream** (it needs the
> whole clip + dense per-crop ReID). A days-long RTSP run exercises the **online**
> pipeline (greedy gallery, `pipeline_mmp_nvdcf_online.yaml`) — measure **runtime
> stability** there, keep **accuracy** measurement on held-out clips offline.

---

## 1. Database

### Decide what to persist (drives the choice)
- [ ] Per-frame tracks: `(ts, cam_id, frame, local_id, global_id, left, top, w, h, conf)` — very high rate, append-only time-series.
- [ ] ReID gallery state: `global_id -> running-mean embedding (256-d)` — low rate, mutable, vector.
- [ ] Events/analytics: entry/exit, dwell, per-zone occupancy — low rate, relational.

### Choice
- [ ] **Start (this eval): SQLite** — single file, zero ops, fine for single-host days-long runs (handles 100M+ rows). Or even hourly **Parquet** files (we already export CSV).
- [ ] **Production (multi-host): TimescaleDB (Postgres ext)** — hypertables + auto time-partitioning + **retention policies** (age out old rows for multi-day), `pgvector` for the gallery embeddings, continuous aggregates for hourly occupancy/dwell rollups — all in one Postgres.
- [ ] Reject for this workload: MongoDB/Cassandra (wrong access pattern), InfluxDB (second system for embeddings/events).

### Implementation
- [ ] Add a **batched DB sink** to `CrossCameraGalleryProbe` (`src/reid/gallery.py`): buffer rows, flush every N frames (NOT per-detection — at 20 cam × 10 fps × ~10 people ≈ **2000 rows/s ≈ 170M rows/day**, per-row insert will bottleneck before the GPU).
- [ ] Schema: tracks hypertable on `ts`; gallery table (`global_id`, `embedding vector(256)`, `last_seen`, `n_obs`); events table.
- [ ] Index: `(cam_id, ts)` and `(global_id, ts)` for the common queries.
- [ ] Retention policy: drop raw per-frame tracks older than X days; keep aggregates.

---

## 2. Long-duration (days) stability eval

Mentor's ask: loop video for days. Validates **runtime stability/drift**, NOT accuracy (looped GT repeats).

### What to watch (stability verdict)
- [ ] **Global-ID growth**: does `total_gids_ever_assigned` plateau or climb forever (gallery leak)? Today retail sits ~13 — confirm it stays bounded over hours.
- [ ] **Memory/VRAM creep**: RSS + `nvidia-smi` flat over time (tracker/gallery state leaks).
- [ ] **FPS stability**: no degradation over days.
- [ ] **ID drift across loop boundaries**: same person re-entering as new GID each loop = gallery not persisting/merging.
- [ ] **DB write lag**: insert rate keeps up with track rate.

### Caveat
- [ ] Do **NOT** read accuracy from a looped run (GT loops too → meaningless IDF1). Accuracy stays on held-out clips (e.g. `64pm_retail_0`, offline anchor-guided).

---

## 3. RTSP via MediaMTX (mentor's suggestion)

Pipeline **already supports `rtsp://`** — `src/pipeline/sources.py` detects rtsp, sets `live-source=1` on mux + `sync=0` on sink. No code change to consume.

- [ ] Run MediaMTX: `docker run --rm -it -p 8554:8554 bluenviron/mediamtx`
- [ ] Loop each camera into an RTSP path (forever, native fps):
  ```bash
  ffmpeg -re -stream_loop -1 -i <scene>/cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
  # repeat cam2..camN
  ```
  - [ ] **`-re`** = real-time pacing (simulate live camera; without it, feeds as fast as disk).
  - [ ] **`-stream_loop -1`** = loop forever; **`-c copy`** = no re-encode (low CPU, MediaMTX not the bottleneck).
- [ ] Point the **online** pipeline at the RTSP list:
  ```bash
  python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_online.yaml \
    --nvinfer-config configs/models/nvinfer_yolov11_10min_clean_fp32nms07.yml \
    rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2 ... --no-display
  ```
- [ ] Write a small **launcher script** to spin up N looped RTSP paths from a scene's cameras (1 ffmpeg per cam) + tear down.

---

## Target end-to-end (days-long run)

```
MediaMTX (loop files -> RTSP, -re)  ->  src.main ONLINE pipeline  ->  gallery probe
   -> batched DB sink (SQLite now / Timescale+pgvector later)  ->  dashboards / hourly aggregates
```

## 4. Spatial analytics (mentor's asks)

Required outputs: (a) **common movement routes between zones**, (b) **most-used
entry/exit points**, (c) **people count per zone over time**.

### What already exists (build on it, don't reinvent)
`scripts/eval/trajectory_analytics.py` already produces, in the world ground-plane:
- `journey_map.png` — per-identity world polyline (who went where) → routes (raw)
- `od_matrix.png/.csv` — zone-to-zone transition counts → **flow between zones**
- `time_in_zone.csv` — seconds each global_id spends per zone
- `dwell_map.png` — occupancy heatmap

**BUT** zones there are an **auto GxG grid**, not semantic named zones; and there's
no entry/exit ranking nor time-bucketed throughput. Those are the gaps.

### Design decision — zone space
- [ ] **Semantic named zones** (entrance, checkout, aisle-1, exit-door, …) as polygons, defined once per scene in a config (e.g. `configs/zones/<scene>.json`).
- [ ] **World (ground-plane) zones** to unify across cameras (a person in "checkout" is checkout from any camera). The ground-plane projection is noisy (~270 mm, see retail STCRA findings), **but coarse zones tolerate that noise** — this is the *good* use of geometry on this dataset (unlike fine STCRA). Fallback: per-camera **image-space** polygons if world is too noisy for a given scene.
- [ ] Point-in-polygon assignment per detection (use foot point / `tracklet_bev.csv` world coords).

### (a) Common movement routes between zones
- [ ] Per `global_id`, build the **ordered zone-visit sequence** (collapse consecutive same-zone frames → `A→B→C`).
- [ ] Aggregate sequences → **most frequent n-gram paths** (top routes), not just pairwise OD.
- [ ] Output: `routes_top.csv` (path, count, avg duration) + a **flow map** (zone graph, edge width ∝ flow) — extend `od_matrix` from grid to named zones + directed edges.

### (b) Most-used entry/exit points
- [ ] Tag zones as entry/exit (doors) in the zone config, OR auto-detect: a track's **first-seen** zone = entry, **last-seen** zone = exit.
- [ ] Rank entry zones by first-seen count, exit zones by last-seen count → `entry_exit_ranking.csv`.
- [ ] Guard against fragmentation: a track that drops + re-enters mid-scene shouldn't count as a new entry/exit (require the first/last to be near a designated door zone, or min-track-length).

### (c) People count per zone over time
- [ ] Time-bucket (e.g. per minute) × zone → **unique global_ids present** (occupancy) and **throughput** (entries into the zone that bucket).
- [ ] Output: `zone_occupancy_timeseries.csv` (ts_bucket, zone, n_unique, n_enter) + line charts per zone.
- [ ] This is a natural **TimescaleDB continuous aggregate** over the track stream (§1) — for live/long runs compute it in-DB; for offline, aggregate the CSV.

### Implementation steps
- [ ] Define zone schema + a small editor/loader (reuse `gt_editor.py`-style click-to-draw, or hand-author the JSON polygons).
- [ ] `src/analytics/zones.py` — load zones, assign detections to zones (point-in-polygon).
- [ ] Extend `trajectory_analytics.py` (offline) to consume named zones → routes / entry-exit / occupancy CSVs + plots.
- [ ] Live path: zone-assignment in the gallery probe → write zone events to DB → dashboard queries.

### Caveats
- [ ] Accuracy of these analytics is **bounded by GID stability + recall** — if the same person fragments into 2 GIDs, routes/entries double-count. So fix tracking quality (the work so far) before trusting the numbers; sanity-check on a scene with known ground truth. **(Confirmed risk: held-out `64pm_retail_0` had 3.3× more ID switches than the train scene → analytics will over-count more on unseen data.)**
- [ ] Coarse zones are robust to ground-plane noise; fine-grained position analytics are not — keep zones large.

### Interactive UI (mentor: "interactive UI will be good")
Two UI pieces — keep them separate (different lifecycles):

**(i) Zone editor** — draw/name zones once per scene:
- [ ] Browser canvas: load a camera frame (or the top-down ground-plane map), click to draw polygons, name + tag them (entry/exit/aisle/checkout), save → `configs/zones/<scene>.json`.
- [ ] Reuse the **zero-dep local web-server pattern** of `scripts/datasets/reid_label_app.py` (already serves images + saves JSON, no extra deps), or use `streamlit-drawable-canvas` if a richer canvas is wanted.

**(ii) Analytics dashboard** — view the results live/over time:
- [ ] **Grafana + TimescaleDB** (recommended for the time-series panels): people-per-zone-over-time, throughput, occupancy — live, no code, pairs with §1 DB. Best for the days-long run monitoring (also shows GID-count/FPS health).
- [ ] **Streamlit** for the custom spatial views Grafana doesn't do natively: flow map (zone graph, edge width ∝ flow), top-routes table, entry/exit ranking bar charts, journey replay. Reads the analytics CSVs or queries the DB.
- [ ] Live overlay (optional): draw zone polygons + per-zone live counts on the OSD video (extend `src/reid/visualization.py`) so the operator sees zones on the camera feed.

Suggested split: **Grafana** = ops/time-series monitoring; **Streamlit** = analyst-facing flow/route/entry-exit exploration; **zero-dep canvas** = zone authoring.

---

## Suggested order
1. [ ] **Held-out accuracy first** (`64pm_retail_0` offline) — confirm the real retail number before building production plumbing on it.
2. [ ] MediaMTX loop launcher + run online pipeline on RTSP (no DB yet) — verify it consumes rtsp and stays up for hours.
3. [ ] Add batched SQLite sink to the gallery probe.
4. [ ] Days-long run; watch GID plateau / VRAM / FPS / DB lag.
5. [ ] Spatial analytics (§4): named zones → routes / entry-exit / zone-occupancy-over-time (extend `trajectory_analytics.py` first, offline; then live-in-DB).
6. [ ] Interactive UI (§4): zero-dep zone editor → Streamlit analyst dashboard + Grafana ops/time-series.
7. [ ] Productionize: Timescale + pgvector, retention, dashboards.

## Open questions
- [ ] Online (greedy gallery) accuracy is the lower regime vs offline anchor-guided — is online-quality acceptable for the production use case, or is a **near-online windowed** path (`src/eval/nearline_merge.py`, ~8 s latency) needed? (CLAUDE.md nearline anchors: lobby_0 0.8365, industry 0.8360 @ window_frames=125.)
- [ ] Single-host (one GPU, RTX 5060 Ti 16GB) vs multi-host scaling target for the camera count.
