# SENTINEL — MTMC Operator Console

An interactive web console for the multi-camera people tracker (DeepStream MTMC
pipeline in this repo). It presents the 5 environments × 4 cameras as a live
surveillance/AI-monitoring system: live wall, zone views, an interactive
Region-of-Interest editor that emits real `nvdsanalytics` config, an analytics
function matrix, and density heatmaps.

> Prototype frontend. Camera tiles use real still frames from the pipeline with
> an animated synthetic tracking overlay; the per-camera **detail view plays the
> actual pipeline OSD video**. All data is mock/seeded but shaped to match the
> real pipeline so a backend can be wired in with minimal change.

---

## Run

```bash
cd webui
npm install
npm run assets     # copy/symlink real frames, heatmaps, OSD videos into public/
npm run dev        # http://localhost:5180
```

`npm run assets` pulls from this repo's existing outputs:

| public/ path        | source                                              |
|---------------------|-----------------------------------------------------|
| `frames/`           | `report/latex/Images/orig_<scene>_camN.jpg`         |
| `heatmaps/<scene>/` | `output/demo/<scene>/heatmap/*.png`                 |
| `feeds/<scene>_osd.mp4` | `output/demo/<scene>/<scene>_live_buffered_osd.mp4` (symlink) |

If those outputs don't exist yet, the app still runs — tiles fall back to the
synthetic overlay and heatmap/video panels show nothing for missing files.

Production build: `npm run build` → `dist/` (static, deployable anywhere).

### Live pipeline mode (real RTSP → DeepStream OSD → browser)

The **Live Wall → ◉ PIPELINE LIVE** tab plays the *actual* DeepStream output —
real RTSP cameras, the pipeline's own OSD with anchor-guided **Buffered IDs**
drawn on it — streamed into the browser over HLS. One command runs the whole
chain (needs the GPU + Docker for MediaMTX):

```bash
webui/scripts/start-live.sh                 # ALL 20 cameras (default)
# or one zone: webui/scripts/start-live.sh dataset/MMPTracking_10minute/val/64pm_retail_0
```

Chain: `cam*.mp4 → ffmpeg -re → MediaMTX (RTSP) → src.main (YOLO+NvDCF+SGIE,
OSD=buffered remap) → hlssink2 → webui/public/live/stream.m3u8`.
A `src.mtmc.live_buffered` consumer re-clusters embeddings every 2 s and rewrites
the `(cam,track)→GID` map the pipeline reads (`--buffered-remap`), so the OSD
shows stable Buffered IDs.

**The pipeline emits ONE tiled mosaic** (20 cams = 5 cols × 4 rows). The console
decodes it **once** (`LiveMosaicProvider`) and every camera tile draws its own
cell from that shared frame onto a canvas (`LiveCell`) — 1 decode, 20 live views.
So when the stream is up, the *whole console* goes live: grid tiles, zone view,
and camera detail all show genuine live DeepStream OSD (red **LIVE** badge);
when it's down they fall back to the recorded **REPLAY** clips. Camera
`streamIndex i` maps to mosaic cell `(i//5, i%5)`, matching `data/zones.ts`.

Start the console (`npm run dev`), open `#live`; tiles auto-switch to live once
segments appear (~15 s warmup). Stop the stream with Ctrl-C.

Verified end-to-end, all 20 cams: ~9 fps/cam, RAM stable (~16 GB free), live
mosaic + per-cell tiles playing in-browser. The pipeline gained a
`--stream-hls <dir>` flag; live needs RTP-over-TCP + `hlssink2 sync=0` +
pipeline `--no-sync` (see src changes) or it stalls at 0 fps.

Views are deep-linkable via hash: `#dashboard`, `#live`, `#zone`, `#roi`,
`#analytics`, `#heatmap`.

---

## Layout of the code

```
src/
  data/            domain model — swap these for API calls later
    types.ts         Camera, Zone, Roi, AnalyticsDef, AlertEvent
    zones.ts         5 zones × 4 cams, mirrors configs/sources/val_20cam_mixed.txt
    analytics.ts     the 7 analytics functions
    rois.ts          seed ROIs, normalized from configs/analytics/*.txt
    events.ts        seed alert feed
  lib/
    nvdsanalytics.ts ROI[] → DeepStream nvdsanalytics .txt (the real format)
    tracks.ts        deterministic synthetic track paths (replace w/ live bbox stream)
    useClock.ts      wall clock + relative-time helpers
  components/
    layout/   Sidebar (zone tree) · TopBar (KPIs, clock, alert ticker)
    common.tsx  Panel · StatusDot · Stat · Sparkline · Bar
    dashboard/  Dashboard — zone cards, throughput, alert feed
    live/       LiveView (1/4/9/zone/all-20 layouts) · CameraTile ·
                TrackOverlay (canvas) · CameraDetail (plays OSD video)
    zone/       ZoneView — zone cameras + summary + events
    roi/        RoiEditor + RoiCanvas — the interactive editor
    analytics/  AnalyticsConfig — per-camera function matrix
    heatmap/    HeatmapView — occupancy/footfall/dwell overlay
  App.tsx       view state + shared ROI store
```

Styling is plain CSS with a token system in `styles/tokens.css` (no UI library).
Design language: dark operations-control-room — deep slate, single teal signal
accent, amber/red telemetry, indigo analytics; Chakra Petch (HUD) + JetBrains
Mono (data) + Inter (body).

---

## Main components

- **Dashboard** — every zone as a card (live count, Global IDF1, per-camera
  status strip, IDF1 sparkline), a network throughput panel, and a clickable
  alert feed.
- **Live Wall** — switch between 1×1, 2×2, 3×3, per-zone, and all-20 grids.
  Each tile is a real frame + an animated canvas overlay drawing synthetic
  bounding boxes, Global IDs and trajectory trails. Click a tile → detail view,
  which can play the **real pipeline OSD video** or the synthetic overlay.
- **Zone View** — the 4 cameras of one environment with a zone-level summary and
  scoped event list.
- **ROI Editor** *(signature)* — pick a camera, choose a region type
  (detection / restricted / counting line / heatmap / ignore / overcrowd), and
  draw directly on the feed. Vertices are draggable; regions are movable;
  each region gets a name, assigned analytics functions, and (for overcrowd) an
  object threshold. A live panel renders the exact `nvdsanalytics` config block
  and offers Copy / Export / Reset.
- **Analytics Config** — a camera × function matrix to enable/disable the seven
  analytics functions per camera, seeded from the ROI assignments.
- **Heatmaps** — occupancy / footfall / dwell overlays (real PNGs) on BEV or any
  camera, with opacity and time-window controls and a GT-correlation readout.

---

## Wiring to a real backend later

The data layer is the only seam. Replace the static exports with fetches:

1. **Cameras / zones / status** — `data/zones.ts` → `GET /api/zones`,
   `GET /api/cameras`. Stream live `status`/`fps`/`people` over a WebSocket and
   keep the same `Camera` shape.
2. **Live detections** — `lib/tracks.ts` → subscribe to a bbox/track WebSocket
   (the pipeline already exports per-detection rows + embeddings). Feed real
   `{gid, x, y, w, h}` into `TrackOverlay` instead of `makeTracks`.
3. **Camera feeds** — point `Camera.feed` at WebRTC / HLS / MJPEG endpoints
   instead of the static OSD `.mp4`; `CameraDetail`'s `<video>` already handles a
   URL source.
4. **ROIs ↔ nvdsanalytics** — `lib/nvdsanalytics.ts` already produces the exact
   `configs/analytics/nvdsanalytics_<scene>.txt` format. `POST` the editor's
   `Roi[]` to a service that writes that file and hot-reloads the
   `nvdsanalytics` element. Read existing rules back with the inverse parser.
5. **Analytics matrix** — persist per-camera toggles to `GET/PUT /api/analytics`;
   region-bound functions map to nvdsanalytics rules, probe-based ones
   (heatmap/dwell/occupancy) map to tracker-metadata probes.
6. **Heatmaps** — `HeatmapView` reads PNGs by convention
   (`/heatmaps/<scene>/<view>_<metric>.png`); back it with a heatmap service or
   the existing `scripts/eval/heatmap_from_export.py` output.
7. **Alerts** — `data/events.ts` → an event WebSocket; the ticker and feeds
   already consume the `AlertEvent` shape.
