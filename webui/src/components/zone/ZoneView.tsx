import type { Nav } from '../../App'
import type { AlertEvent } from '../../data/types'
import { ZONES, camsOfZone, zoneById } from '../../data/zones'
import { CameraTile } from '../live/CameraTile'
import { Panel, Stat, StatusDot } from '../common'
import { fmtAgo } from '../../lib/useClock'
import './zone.css'

export function ZoneView({ nav, go, events }: {
  nav: Nav; go: (n: Partial<Nav>) => void; events: AlertEvent[]
}) {
  const zone = zoneById(nav.zoneId ?? '') ?? ZONES[0]
  const cams = camsOfZone(zone.id)
  const people = cams.reduce((s, c) => s + c.people, 0)
  const online = cams.filter((c) => c.status === 'online').length
  const zoneEvents = events.filter((e) => e.zoneId === zone.id)
  const now = Date.now()

  return (
    <div className="zoneview">
      <div className="zoneview__head">
        <div className="zoneview__title">
          <span className="zoneview__swatch" style={{ background: zone.accent }} />
          <div>
            <h2 style={{ fontSize: 18 }}>{zone.name}</h2>
            <span className="eyebrow">{zone.scene} · {cams.length} cameras</span>
          </div>
        </div>
        <div className="zoneview__zswitch">
          {ZONES.map((z) => (
            <button key={z.id} className={`zoneview__ztab ${z.id === zone.id ? 'is-active' : ''}`}
              onClick={() => go({ zoneId: z.id })}>
              <i style={{ background: z.accent }} />{z.name}
            </button>
          ))}
        </div>
      </div>

      <div className="zoneview__body">
        <div className="zoneview__cams">
          {cams.map((c) => (
            <CameraTile key={c.id} cam={c}
              onOpen={(cam) => go({ view: 'live', cameraId: cam.id, zoneId: zone.id })} />
          ))}
        </div>

        <aside className="zoneview__side">
          <Panel title="Zone Summary">
            <div className="zoneview__stats">
              <Stat label="In view" value={people} accent="var(--signal)" />
              <Stat label="Cameras" value={`${online}/${cams.length}`} />
              <Stat label="Global IDF1" value={zone.idf1.toFixed(3)} accent={zone.idf1 >= 0.8 ? 'var(--scan)' : 'var(--warn)'} />
              <Stat label="Mean FPS" value={(cams.reduce((s,c)=>s+c.fps,0)/cams.length).toFixed(1)} />
            </div>
            <p className="zoneview__blurb">{zone.blurb}</p>
          </Panel>

          <Panel title="Quick Actions">
            <div className="zoneview__actions">
              <button onClick={() => go({ view: 'live', zoneId: zone.id, cameraId: null })}>Open live wall →</button>
              <button onClick={() => go({ view: 'roi', zoneId: zone.id, cameraId: cams[0].id })}>Edit ROIs →</button>
              <button onClick={() => go({ view: 'analytics', zoneId: zone.id })}>Analytics config →</button>
              <button onClick={() => go({ view: 'heatmap', zoneId: zone.id })}>Heatmaps →</button>
            </div>
          </Panel>

          <Panel title="Zone Events" right={<span className="eyebrow mono">{zoneEvents.length}</span>}>
            {zoneEvents.length === 0
              ? <p className="zoneview__empty">No events in this zone. All clear.</p>
              : <ul className="zoneview__events">
                  {zoneEvents.map((e) => (
                    <li key={e.id} className={`zev zev--${e.severity}`}>
                      <StatusDot status={e.severity === 'alarm' ? 'offline' : e.severity === 'warn' ? 'warning' : 'online'} />
                      <span className="zev__msg">{e.message}</span>
                      <span className="mono zev__ago">{fmtAgo(e.ts, now)}</span>
                    </li>
                  ))}
                </ul>}
          </Panel>
        </aside>
      </div>
    </div>
  )
}
