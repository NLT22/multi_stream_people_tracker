import type { Roi } from './types'

// Seeded from the project's real configs/analytics/*.txt rules, normalized from
// 1920×1080 config space to 0..1. The ROI editor reads/writes these and the
// nvdsanalytics generator turns them back into the exact .txt format.
const N = (x: number, y: number) => ({ x: x / 1920, y: y / 1080 })

export const SEED_ROIS: Roi[] = [
  // --- cafe entry (cam-0): TABLE detection + central overcrowd + aisle tripwire
  {
    id: 'roi-cafe-table', name: 'TABLE', kind: 'detection', cameraId: 'cam-0',
    points: [N(330, 120), N(1590, 120), N(1590, 980), N(330, 980)],
    analytics: ['counting', 'occupancy'], enabled: true,
  },
  {
    id: 'roi-cafe-oc', name: 'OC', kind: 'overcrowd', cameraId: 'cam-0',
    points: [N(330, 120), N(1590, 120), N(1590, 980), N(330, 980)],
    threshold: 6, analytics: ['occupancy'], enabled: true,
  },
  {
    id: 'roi-cafe-aisle', name: 'Aisle', kind: 'counting', cameraId: 'cam-0',
    points: [N(300, 640), N(1620, 640)],
    direction: [N(960, 430), N(960, 820)],
    analytics: ['lineCrossing', 'direction'], enabled: true,
  },

  // --- industry floor A (cam-12): restricted loading bay + walkway tripwire
  {
    id: 'roi-ind-dock', name: 'Loading-Bay', kind: 'restricted', cameraId: 'cam-12',
    points: [N(1180, 180), N(1760, 180), N(1760, 940), N(1180, 940)],
    analytics: ['intrusion'], enabled: true,
  },
  {
    id: 'roi-ind-walk', name: 'Walkway', kind: 'counting', cameraId: 'cam-12',
    points: [N(320, 640), N(1600, 640)],
    direction: [N(960, 430), N(960, 820)],
    analytics: ['lineCrossing', 'direction'], enabled: true,
  },

  // --- retail aisle 1 (cam-16): full-floor detection + tighter overcrowd
  {
    id: 'roi-retail-floor', name: 'Aisle-1', kind: 'detection', cameraId: 'cam-16',
    points: [N(160, 120), N(1760, 120), N(1760, 1000), N(160, 1000)],
    analytics: ['counting', 'heatmap'], enabled: true,
  },
  {
    id: 'roi-retail-oc', name: 'OC', kind: 'overcrowd', cameraId: 'cam-16',
    points: [N(160, 120), N(1760, 120), N(1760, 1000), N(160, 1000)],
    threshold: 8, analytics: ['occupancy'], enabled: true,
  },
]
