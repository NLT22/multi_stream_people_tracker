import type { Camera, Zone, CamStatus } from './types'
import { BENCHMARK } from './benchmark'

// Mirrors configs/sources/val_20cam_mixed.txt — 5 scenes, 4 cams each,
// nvstreammux source_id 0..19 in the same order the pipeline ingests them.
interface ZoneSeed {
  key: Zone['key']
  name: string
  scene: string        // dataset folder + asset prefix
  framePrefix: string  // matches report/latex/Images/orig_<framePrefix>_camN.jpg
  idf1: number
  accent: string
  blurb: string
  // per-camera: [name, status, fps, people, health]
  cams: [string, CamStatus, number, number, number][]
}

const SEEDS: ZoneSeed[] = [
  {
    key: 'cafe', name: 'Café Shop', scene: 'cafe_shop', framePrefix: 'cafe_shop',
    idf1: 0.833, accent: '#2de2c8',
    blurb: 'Seated patrons around a central island. Moderate density, slow motion.',
    cams: [
      ['CAFE · Entry', 'online', 10.7, 5, 0.98],
      ['CAFE · Counter', 'online', 10.5, 7, 0.96],
      ['CAFE · Island', 'online', 10.6, 6, 0.97],
      ['CAFE · Patio', 'warning', 9.2, 3, 0.71],
    ],
  },
  {
    key: 'lobby', name: 'Lobby', scene: 'lobby', framePrefix: 'lobby',
    idf1: 0.895, accent: '#56b3ff',
    blurb: 'High-ceiling transit space. Best cross-camera continuity of the network.',
    cams: [
      ['LOBBY · North', 'online', 10.8, 8, 0.99],
      ['LOBBY · Desk', 'online', 10.6, 4, 0.97],
      ['LOBBY · Elevators', 'online', 10.7, 6, 0.98],
      ['LOBBY · South', 'online', 10.5, 5, 0.95],
    ],
  },
  {
    key: 'office', name: 'Office', scene: 'office', framePrefix: 'office',
    idf1: 0.861, accent: '#8b7cf6',
    blurb: 'Desks and meeting pods. Frequent partial occlusion behind monitors.',
    cams: [
      ['OFFICE · Open Plan', 'online', 10.6, 9, 0.96],
      ['OFFICE · Pods', 'online', 10.4, 4, 0.94],
      ['OFFICE · Corridor', 'warning', 9.7, 2, 0.78],
      ['OFFICE · Kitchen', 'online', 10.5, 3, 0.95],
    ],
  },
  {
    key: 'industry', name: 'Industry Safety', scene: 'industry_safety', framePrefix: 'industry_safety',
    idf1: 0.805, accent: '#f6a821',
    blurb: 'Uniformed workers, hi-vis PPE. ReID-hard: appearance is near-identical.',
    cams: [
      ['IND · Floor A', 'online', 10.5, 7, 0.93],
      ['IND · Floor B', 'online', 10.3, 6, 0.92],
      ['IND · Loading', 'warning', 9.1, 4, 0.69],
      ['IND · Gantry', 'online', 10.4, 5, 0.9],
    ],
  },
  {
    key: 'retail', name: 'Retail', scene: 'retail', framePrefix: 'retail',
    idf1: 0.660, accent: '#ff7a9c',
    blurb: 'Dense aisles, heavy occlusion. Weakest zone — local ID switches dominate.',
    cams: [
      ['RETAIL · Aisle 1', 'online', 10.3, 11, 0.88],
      ['RETAIL · Aisle 2', 'warning', 9.0, 9, 0.66],
      ['RETAIL · Checkout', 'online', 10.4, 7, 0.9],
      ['RETAIL · Stockroom', 'offline', 0, 0, 0.0],
    ],
  },
]

export const ZONES: Zone[] = []
export const CAMERAS: Camera[] = []

let streamIndex = 0
for (const s of SEEDS) {
  const zoneId = `zone-${s.key}`
  const camIds: string[] = []
  s.cams.forEach(([name, status, fps, people, health], i) => {
    const id = `cam-${streamIndex}`
    const camNo = i + 1
    CAMERAS.push({
      id,
      streamIndex,
      name,
      zoneId,
      status,
      fps,
      targetFps: 10,
      resolution: '1920×1080',
      people,
      health,
      frame: `/frames/orig_${s.framePrefix}_cam${camNo}.jpg`,
      // Each camera's real pipeline OSD = its quadrant cropped from the zone's
      // 2×2 tiled OSD video (see setup-assets.sh). Every camera has a real feed.
      feed: `/feeds/${s.scene}_cam${camNo}.mp4`,
    })
    camIds.push(id)
    streamIndex++
  })
  ZONES.push({
    id: zoneId, key: s.key, name: s.name, scene: s.scene,
    // IDF1 comes from the canonical benchmark (single source of truth), not the seed.
    cameras: camIds, idf1: BENCHMARK.perScene[s.key], accent: s.accent, blurb: s.blurb,
  })
}

export const zoneById = (id: string) => ZONES.find((z) => z.id === id)
export const camById = (id: string) => CAMERAS.find((c) => c.id === id)
export const camsOfZone = (zoneId: string) =>
  CAMERAS.filter((c) => c.zoneId === zoneId)
