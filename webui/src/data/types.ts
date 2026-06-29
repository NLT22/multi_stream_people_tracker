// Domain types — shaped to mirror the real DeepStream MTMC pipeline so this
// UI can later be backed by live data with minimal change.

export type CamStatus = 'online' | 'warning' | 'offline'

export interface Camera {
  id: string            // pipeline source id, e.g. "cam-0"
  streamIndex: number   // nvstreammux source_id (0..19)
  name: string          // operator-facing name
  zoneId: string
  status: CamStatus
  fps: number
  targetFps: number
  resolution: string
  people: number        // current tracked persons in view
  frame: string         // still-frame asset (public/)
  feed?: string         // OSD video asset (public/), if present
  health: number        // 0..1 stream health
}

export type ZoneKind = 'cafe' | 'lobby' | 'office' | 'industry' | 'retail' | 'warehouse'

export interface Zone {
  id: string
  key: ZoneKind
  name: string
  scene: string         // dataset scene folder name
  cameras: string[]     // camera ids
  idf1: number          // last measured Global IDF1 for the zone
  accent: string        // map/legend color
  blurb: string
}

// ---- analytics functions (per camera toggles) ----
export type AnalyticsKey =
  | 'heatmap'
  | 'counting'
  | 'lineCrossing'
  | 'intrusion'
  | 'direction'
  | 'dwell'
  | 'occupancy'

export interface AnalyticsDef {
  key: AnalyticsKey
  label: string
  blurb: string
  glyph: string         // single-char HUD glyph
  accent: string
}

// ---- ROI / nvdsanalytics rules ----
export type RoiKind =
  | 'detection'   // roi-filtering
  | 'restricted'  // roi-filtering (inverse / intrusion)
  | 'counting'    // line-crossing
  | 'heatmap'     // heatmap region
  | 'ignore'      // inverse roi
  | 'overcrowd'   // overcrowding region

export interface Pt { x: number; y: number }    // normalized 0..1 in config space

export interface Roi {
  id: string
  name: string
  kind: RoiKind
  cameraId: string
  points: Pt[]          // polygon vertices (≥3) OR 2 points for a line
  direction?: Pt[]      // line-crossing direction segment (2 pts), optional
  threshold?: number    // overcrowding object-threshold
  analytics: AnalyticsKey[]
  enabled: boolean
}

export interface AlertEvent {
  id: string
  ts: number
  zoneId: string
  cameraId: string
  kind: 'intrusion' | 'overcrowd' | 'linecross' | 'offline' | 'dwell'
  severity: 'info' | 'warn' | 'alarm'
  message: string
}
