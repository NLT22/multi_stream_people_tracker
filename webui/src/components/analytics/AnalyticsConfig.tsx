import { useMemo, useState } from 'react'
import type { Nav } from '../../App'
import type { AnalyticsKey, Roi } from '../../data/types'
import { ANALYTICS } from '../../data/analytics'
import { ZONES, camsOfZone, zoneById } from '../../data/zones'
import { Panel } from '../common'
import './analytics.css'

type Matrix = Record<string, Set<AnalyticsKey>>

function seedMatrix(rois: Roi[], camIds: string[]): Matrix {
  const m: Matrix = {}
  for (const id of camIds) {
    const set = new Set<AnalyticsKey>()
    rois.filter((r) => r.cameraId === id && r.enabled)
      .forEach((r) => r.analytics.forEach((a) => set.add(a)))
    m[id] = set
  }
  return m
}

export function AnalyticsConfig({ nav, go, rois }: {
  nav: Nav; go: (n: Partial<Nav>) => void; rois: Roi[]
}) {
  const zone = zoneById(nav.zoneId ?? '') ?? ZONES[0]
  const cams = camsOfZone(zone.id)
  const seeded = useMemo(() => seedMatrix(rois, cams.map((c) => c.id)), [rois, zone.id])
  const [matrix, setMatrix] = useState<Matrix>(seeded)
  // re-seed when zone changes
  const [activeZone, setActiveZone] = useState(zone.id)
  if (activeZone !== zone.id) { setActiveZone(zone.id); setMatrix(seeded) }

  const toggle = (camId: string, key: AnalyticsKey) =>
    setMatrix((m) => {
      const set = new Set(m[camId])
      set.has(key) ? set.delete(key) : set.add(key)
      return { ...m, [camId]: set }
    })

  const totalOn = Object.values(matrix).reduce((s, set) => s + set.size, 0)

  return (
    <div className="acfg">
      <div className="acfg__head">
        <div>
          <h2 style={{ fontSize: 18 }}>Analytics Configuration</h2>
          <span className="eyebrow">Enable functions per camera · {zone.name}</span>
        </div>
        <div className="acfg__zsel">
          {ZONES.map((z) => (
            <button key={z.id} className={`acfg__ztab ${z.id === zone.id ? 'is-active' : ''}`}
              onClick={() => go({ zoneId: z.id })}>
              <i style={{ background: z.accent }} />{z.name}
            </button>
          ))}
        </div>
      </div>

      <Panel title={`Function Matrix · ${totalOn} active`} ticks>
        <div className="acfg__scroll">
          <table className="acfg__table">
            <thead>
              <tr>
                <th className="acfg__corner eyebrow">Camera</th>
                {ANALYTICS.map((a) => (
                  <th key={a.key} className="acfg__col">
                    <span className="hud acfg__glyph" style={{ color: a.accent }}>{a.glyph}</span>
                    <span className="acfg__colname">{a.label}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cams.map((c) => (
                <tr key={c.id}>
                  <th className="acfg__rowhd">
                    <span className={`acfg__cdot acfg__cdot--${c.status}`} />
                    <span className="acfg__camname">{c.name}</span>
                    <span className="mono acfg__sid">src-{c.streamIndex}</span>
                  </th>
                  {ANALYTICS.map((a) => {
                    const on = matrix[c.id]?.has(a.key)
                    const off = c.status === 'offline'
                    return (
                      <td key={a.key} className="acfg__cell">
                        <button
                          className={`acfg__toggle ${on ? 'is-on' : ''}`}
                          style={{ '--ac': a.accent } as React.CSSProperties}
                          disabled={off}
                          aria-pressed={on}
                          aria-label={`${a.label} on ${c.name}`}
                          onClick={() => toggle(c.id, a.key)}>
                          {on ? '●' : ''}
                        </button>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="acfg__legend">
        {ANALYTICS.map((a) => (
          <div key={a.key} className="acfg__legrow">
            <span className="hud" style={{ color: a.accent }}>{a.glyph}</span>
            <div>
              <span className="acfg__legname">{a.label}</span>
              <span className="acfg__legblurb">{a.blurb}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="acfg__foot">
        <span className="mono">
          Region-bound functions (counting, line crossing, intrusion, overcrowd) generate
          nvdsanalytics rules from the ROI editor. Heatmap, dwell &amp; occupancy run as probes on the tracker metadata.
        </span>
        <button onClick={() => go({ view: 'roi', zoneId: zone.id, cameraId: cams[0].id })}>Define regions →</button>
      </div>
    </div>
  )
}
