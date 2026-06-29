import { useEffect, useMemo, useState } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { Dashboard } from './components/dashboard/Dashboard'
import { LiveView } from './components/live/LiveView'
import { ZoneView } from './components/zone/ZoneView'
import { RoiEditor } from './components/roi/RoiEditor'
import { AnalyticsConfig } from './components/analytics/AnalyticsConfig'
import { HeatmapView } from './components/heatmap/HeatmapView'
import { AskView } from './components/rag/AskView'
import { LiveMosaicProvider } from './components/live/LiveMosaic'
import { camById, setActiveDataset, getActiveDatasetKey } from './data/zones'
import type { DatasetKey } from './data/datasets'
import { seedEvents } from './data/events'
import { SEED_ROIS } from './data/rois'
import type { Roi } from './data/types'
import './app.css'

export type View = 'dashboard' | 'live' | 'zone' | 'roi' | 'analytics' | 'heatmap' | 'ask'

export interface Nav {
  view: View
  zoneId: string | null
  cameraId: string | null
}

const VIEWS: View[] = ['dashboard', 'live', 'zone', 'roi', 'analytics', 'heatmap', 'ask']
// Hash forms: "#live" or "#live/cam-3" (deep-link a specific camera detail).
const navFromHash = (): Partial<Nav> => {
  const [v, cam] = window.location.hash.replace('#', '').split('/')
  const view = (VIEWS.includes(v as View) ? v : 'dashboard') as View
  const c = cam ? camById(cam) : undefined
  return { view, cameraId: c ? c.id : null, zoneId: c ? c.zoneId : null }
}

export default function App() {
  const init = navFromHash()
  const [nav, setNav] = useState<Nav>({ view: init.view ?? 'dashboard', zoneId: init.zoneId ?? null, cameraId: init.cameraId ?? null })
  // ROIs live at app level so the editor + analytics + config preview share them.
  const [rois, setRois] = useState<Roi[]>(SEED_ROIS)
  const [dataset, setDataset] = useState<DatasetKey>(getActiveDatasetKey())
  const events = useMemo(() => seedEvents(Date.now()), [])

  // Switch camera network (MMP ⇄ MTMC): point zones/cameras at the new dataset,
  // reset ROIs (MMP ships seeds; MTMC starts empty for custom drawing) + nav, and
  // remount the view tree (key={dataset}) so every view re-reads the new cameras.
  const switchDataset = (k: DatasetKey) => {
    if (k === dataset) return
    setActiveDataset(k)
    setRois(() => (k === 'mmp' ? SEED_ROIS : []))
    setNav({ view: 'dashboard', zoneId: null, cameraId: null })
    setDataset(k)
  }

  // keep the URL hash in sync so views/cameras are bookmarkable / deep-linkable
  useEffect(() => {
    window.location.hash = nav.view === 'live' && nav.cameraId ? `live/${nav.cameraId}` : nav.view
  }, [nav.view, nav.cameraId])

  const go = (next: Partial<Nav>) => setNav((n) => ({ ...n, ...next }))

  return (
    <LiveMosaicProvider>
    <div className="app">
      <Sidebar nav={nav} go={go} dataset={dataset} onSwitchDataset={switchDataset} />
      <div className="app-col">
        <TopBar nav={nav} go={go} events={events} />
        <main className="app-main app-content" key={dataset}>
          {nav.view === 'dashboard' && <Dashboard go={go} events={events} />}
          {nav.view === 'live' && <LiveView nav={nav} go={go} />}
          {nav.view === 'zone' && <ZoneView nav={nav} go={go} events={events} />}
          {nav.view === 'roi' && <RoiEditor nav={nav} go={go} rois={rois} setRois={setRois} />}
          {nav.view === 'analytics' && <AnalyticsConfig nav={nav} go={go} rois={rois} />}
          {nav.view === 'heatmap' && <HeatmapView nav={nav} go={go} />}
          {nav.view === 'ask' && <AskView nav={nav} go={go} />}
        </main>
      </div>
    </div>
    </LiveMosaicProvider>
  )
}
