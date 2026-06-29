import type { Camera, Zone } from './types'
import { DATASETS, type DatasetKey } from './datasets'

// ZONES / CAMERAS point at the ACTIVE dataset. They are reassigned by
// setActiveDataset (ES module live bindings → importers see the new arrays), and
// App remounts the view tree on switch so every component re-reads. This is why
// the ~11 consumers of these exports need no per-dataset changes.
export let ZONES: Zone[] = DATASETS.mmp.zones
export let CAMERAS: Camera[] = DATASETS.mmp.cameras

let activeKey: DatasetKey = 'mmp'
export const getActiveDatasetKey = (): DatasetKey => activeKey
export function setActiveDataset(key: DatasetKey) {
  activeKey = key
  ZONES = DATASETS[key].zones
  CAMERAS = DATASETS[key].cameras
}

export const zoneById = (id: string) => ZONES.find((z) => z.id === id)
export const camById = (id: string) => CAMERAS.find((c) => c.id === id)
export const camsOfZone = (zoneId: string) => CAMERAS.filter((c) => c.zoneId === zoneId)
