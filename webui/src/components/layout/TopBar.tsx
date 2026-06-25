import type { Nav } from '../../App'
import type { AlertEvent } from '../../data/types'
import { CAMERAS, ZONES } from '../../data/zones'
import { useClock, fmtClock, fmtDate } from '../../lib/useClock'
import './TopBar.css'

const CRumb: Record<Nav['view'], string> = {
  dashboard: 'Overview', live: 'Live Wall', zone: 'Zone', roi: 'ROI Editor',
  analytics: 'Analytics Config', heatmap: 'Heatmaps',
}

export function TopBar({ nav, events }: {
  nav: Nav; go: (n: Partial<Nav>) => void; events: AlertEvent[]
}) {
  const now = useClock()
  const online = CAMERAS.filter((c) => c.status === 'online').length
  const warn = CAMERAS.filter((c) => c.status === 'warning').length
  const off = CAMERAS.filter((c) => c.status === 'offline').length
  const people = CAMERAS.reduce((s, c) => s + c.people, 0)
  const alarms = events.filter((e) => e.severity === 'alarm').length
  const meanIdf1 = (ZONES.reduce((s, z) => s + z.idf1, 0) / ZONES.length).toFixed(3)

  // marquee text doubled for seamless loop
  const ticker = events.map((e) => `${e.kind.toUpperCase()} · ${e.message}`).join('     ◆     ')

  return (
    <header className="topbar app-main">
      <div className="topbar__crumb">
        <span className="eyebrow">SECTOR</span>
        <span className="hud topbar__view">{CRumb[nav.view]}</span>
      </div>

      <div className="topbar__kpis">
        <Kpi label="Cameras" value={`${online}`} sub={`/${CAMERAS.length}`} tone="signal" />
        <Kpi label="Degraded" value={`${warn}`} tone={warn ? 'warn' : 'dim'} />
        <Kpi label="Offline" value={`${off}`} tone={off ? 'alert' : 'dim'} />
        <span className="topbar__div" />
        <Kpi label="Tracked" value={`${people}`} tone="signal" />
        <Kpi label="Mean IDF1" value={meanIdf1} tone="scan" />
        <Kpi label="Alarms" value={`${alarms}`} tone={alarms ? 'alert' : 'dim'} />
      </div>

      <div className="topbar__ticker" aria-label="Live alert feed">
        <span className="topbar__tlabel eyebrow">FEED</span>
        <div className="topbar__tviewport">
          <div className="topbar__tscroll mono">{ticker}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ticker}</div>
        </div>
      </div>

      <div className="topbar__clock mono">
        <span className="topbar__time">{fmtClock(now)}</span>
        <span className="topbar__date">{fmtDate(now)} · UTC+07</span>
      </div>
    </header>
  )
}

function Kpi({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone: 'signal' | 'warn' | 'alert' | 'scan' | 'dim'
}) {
  return (
    <div className="kpi">
      <div className={`kpi__v hud kpi--${tone}`}>{value}{sub && <span className="kpi__sub mono">{sub}</span>}</div>
      <div className="kpi__l eyebrow">{label}</div>
    </div>
  )
}
