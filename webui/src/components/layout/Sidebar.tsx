import { useState } from 'react'
import type { Nav, View } from '../../App'
import { ZONES, camsOfZone } from '../../data/zones'
import { StatusDot } from '../common'
import './Sidebar.css'

const NAV: { view: View; label: string; glyph: string }[] = [
  { view: 'dashboard', label: 'Overview', glyph: '⊞' },
  { view: 'live', label: 'Live Wall', glyph: '⊡' },
  { view: 'roi', label: 'ROI Editor', glyph: '✎' },
  { view: 'analytics', label: 'Analytics', glyph: '⚙' },
  { view: 'heatmap', label: 'Heatmaps', glyph: '▦' },
  { view: 'ask', label: 'Ask', glyph: '✦' },
]

export function Sidebar({ nav, go }: { nav: Nav; go: (n: Partial<Nav>) => void }) {
  // All zones collapsed by default — expand on demand.
  const [open, setOpen] = useState<Record<string, boolean>>({})

  return (
    <aside className="rail">
      <div className="rail__brand">
        <span className="rail__mark hud">◈</span>
        <div>
          <div className="rail__name hud">SENTINEL</div>
          <div className="rail__sub eyebrow">MTMC · 20 CAM</div>
        </div>
      </div>

      <nav className="rail__nav">
        {NAV.map((n) => (
          <button
            key={n.view}
            className={`rail__navitem ${nav.view === n.view ? 'is-active' : ''}`}
            onClick={() => go({ view: n.view })}
            aria-current={nav.view === n.view}
          >
            <span className="rail__glyph hud">{n.glyph}</span>
            {n.label}
          </button>
        ))}
      </nav>

      <div className="rail__zones">
        <div className="rail__zhd eyebrow">Zones / Environments</div>
        {ZONES.map((z) => {
          const cams = camsOfZone(z.id)
          const isOpen = open[z.id]
          const onlineN = cams.filter((c) => c.status === 'online').length
          return (
            <div key={z.id} className="ztree">
              <div className="ztree__row">
                <button
                  className="ztree__toggle"
                  onClick={() => setOpen((o) => ({ ...o, [z.id]: !o[z.id] }))}
                  aria-label={isOpen ? 'Collapse' : 'Expand'}
                >{isOpen ? '▾' : '▸'}</button>
                <button
                  className={`ztree__name ${nav.zoneId === z.id && nav.view === 'zone' ? 'is-active' : ''}`}
                  onClick={() => go({ view: 'zone', zoneId: z.id })}
                >
                  <i className="ztree__swatch" style={{ background: z.accent }} />
                  {z.name}
                </button>
                <span className="ztree__count mono">{onlineN}/{cams.length}</span>
              </div>
              {isOpen && (
                <ul className="ztree__cams">
                  {cams.map((c) => (
                    <li key={c.id}>
                      <button
                        className={`ztree__cam ${nav.cameraId === c.id ? 'is-active' : ''}`}
                        onClick={() => go({ view: 'live', zoneId: z.id, cameraId: c.id })}
                      >
                        <StatusDot status={c.status} />
                        <span className="ztree__camname">{c.name}</span>
                        <span className="mono ztree__fps">
                          {c.status === 'offline' ? '—' : c.fps.toFixed(1)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )
        })}
      </div>

      <div className="rail__foot">
        <StatusDot status="online" label />
        <span className="mono">reid0 · 10.6 fps/cam</span>
      </div>
    </aside>
  )
}
