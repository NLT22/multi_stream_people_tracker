import type { Camera, Zone, CamStatus, ZoneKind } from './types'
import { BENCHMARK } from './benchmark'

// Two camera networks the console can switch between. MMP = the 20-cam mixed
// validation set (production path); MTMC = the AI-City warehouse (geometry-first
// cross-camera path). Each builds its own ZONES/CAMERAS; zones.ts points at the
// active one (see setActiveDataset).
export type DatasetKey = 'mmp' | 'mtmc'

interface ZoneSeed {
  key: ZoneKind
  name: string
  scene: string        // asset prefix (frames/feeds/heatmaps)
  framePrefix: string  // orig_<framePrefix>_camN.jpg
  idf1: number         // fallback if not in BENCHMARK
  accent: string
  blurb: string
  hasFeed?: boolean     // false => still-frame only (no OSD replay video)
  cams: [string, CamStatus, number, number, number][]  // name,status,fps,people,health
}

function build(seeds: ZoneSeed[]): { zones: Zone[]; cameras: Camera[] } {
  const zones: Zone[] = []
  const cameras: Camera[] = []
  let si = 0
  for (const s of seeds) {
    const zoneId = `zone-${s.key}`
    const camIds: string[] = []
    s.cams.forEach(([name, status, fps, people, health], i) => {
      const id = `cam-${si}`
      const camNo = i + 1
      cameras.push({
        id, streamIndex: si, name, zoneId, status, fps, targetFps: 10,
        resolution: '1920×1080', people, health,
        frame: `/frames/orig_${s.framePrefix}_cam${camNo}.jpg`,
        feed: s.hasFeed === false ? undefined : `/feeds/${s.scene}_cam${camNo}.mp4`,
      })
      camIds.push(id)
      si++
    })
    zones.push({
      id: zoneId, key: s.key, name: s.name, scene: s.scene, cameras: camIds,
      idf1: BENCHMARK.perScene[s.key] ?? s.idf1, accent: s.accent, blurb: s.blurb,
    })
  }
  return { zones, cameras }
}

const MMP_SEEDS: ZoneSeed[] = [
  { key: 'cafe', name: 'Café Shop', scene: 'cafe_shop', framePrefix: 'cafe_shop', idf1: 0.833, accent: '#2de2c8',
    blurb: 'Seated patrons around a central island. Moderate density, slow motion.',
    cams: [['CAFE · Entry', 'online', 10.7, 5, 0.98], ['CAFE · Counter', 'online', 10.5, 7, 0.96],
           ['CAFE · Island', 'online', 10.6, 6, 0.97], ['CAFE · Patio', 'warning', 9.2, 3, 0.71]] },
  { key: 'lobby', name: 'Lobby', scene: 'lobby', framePrefix: 'lobby', idf1: 0.895, accent: '#56b3ff',
    blurb: 'High-ceiling transit space. Best cross-camera continuity of the network.',
    cams: [['LOBBY · North', 'online', 10.8, 8, 0.99], ['LOBBY · Desk', 'online', 10.6, 4, 0.97],
           ['LOBBY · Elevators', 'online', 10.7, 6, 0.98], ['LOBBY · South', 'online', 10.5, 5, 0.95]] },
  { key: 'office', name: 'Office', scene: 'office', framePrefix: 'office', idf1: 0.861, accent: '#8b7cf6',
    blurb: 'Desks and meeting pods. Frequent partial occlusion behind monitors.',
    cams: [['OFFICE · Open Plan', 'online', 10.6, 9, 0.96], ['OFFICE · Pods', 'online', 10.4, 4, 0.94],
           ['OFFICE · Corridor', 'warning', 9.7, 2, 0.78], ['OFFICE · Kitchen', 'online', 10.5, 3, 0.95]] },
  { key: 'industry', name: 'Industry Safety', scene: 'industry_safety', framePrefix: 'industry_safety', idf1: 0.805, accent: '#f6a821',
    blurb: 'Uniformed workers, hi-vis PPE. ReID-hard: appearance is near-identical.',
    cams: [['IND · Floor A', 'online', 10.5, 7, 0.93], ['IND · Floor B', 'online', 10.3, 6, 0.92],
           ['IND · Loading', 'warning', 9.1, 4, 0.69], ['IND · Gantry', 'online', 10.4, 5, 0.9]] },
  { key: 'retail', name: 'Retail', scene: 'retail', framePrefix: 'retail', idf1: 0.660, accent: '#ff7a9c',
    blurb: 'Dense aisles, heavy occlusion. Weakest zone — local ID switches dominate.',
    cams: [['RETAIL · Aisle 1', 'online', 10.3, 11, 0.88], ['RETAIL · Aisle 2', 'warning', 9.0, 9, 0.66],
           ['RETAIL · Checkout', 'online', 10.4, 7, 0.9], ['RETAIL · Stockroom', 'offline', 0, 0, 0.0]] },
]

// MTMC_Tracking_2026 Warehouse_022: disjoint cameras, metric floor map, true
// cross-camera identity. Still-frames only in the console (no OSD replay video);
// the geometry global-linker gives the cross-camera IDF1 below.
const MTMC_SEEDS: ZoneSeed[] = [
  { key: 'warehouse', name: 'Warehouse 022', scene: 'warehouse022', framePrefix: 'warehouse022',
    idf1: 0.856, accent: '#5ad1ff', hasFeed: false,
    blurb: 'AI-City warehouse — disjoint FOV + metric floor map. Cross-camera IDs are geometry-first (world position), not appearance.',
    cams: [['WH · Cam 0', 'online', 15.0, 5, 0.9], ['WH · Cam 1', 'online', 15.0, 6, 0.9],
           ['WH · Cam 2', 'online', 15.0, 6, 0.9], ['WH · Cam 3', 'online', 15.0, 4, 0.88]] },
]

export const DATASETS = {
  mmp:  { key: 'mmp' as DatasetKey,  label: 'MMP',  sub: '5 ENV · 20 CAM',     ...build(MMP_SEEDS) },
  mtmc: { key: 'mtmc' as DatasetKey, label: 'MTMC', sub: 'WAREHOUSE · 4 CAM', ...build(MTMC_SEEDS) },
}
export const DATASET_LIST = [DATASETS.mmp, DATASETS.mtmc]
