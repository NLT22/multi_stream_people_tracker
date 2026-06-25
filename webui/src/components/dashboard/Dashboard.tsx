import type { Nav } from '../../App'
import type { AlertEvent } from '../../data/types'
import { ZONES, CAMERAS, camsOfZone } from '../../data/zones'
import { BENCHMARK } from '../../data/benchmark'
import { Panel, Stat, Sparkline, Bar, StatusDot } from '../common'
import { fmtAgo } from '../../lib/useClock'
import './dashboard.css'

const spark = (seed: number, n = 16) =>
  Array.from({ length: n }, (_, i) => 50 + 30 * Math.sin(i * 0.6 + seed) + (i % 3) * 4)

export function Dashboard({ go, events }: { go: (n: Partial<Nav>) => void; events: AlertEvent[] }) {
  const now = Date.now()
  const totalPeople = CAMERAS.reduce((s, c) => s + c.people, 0)

  return (
    <div className="dash">
      <div className="bench panel panel--ticks">
        <div className="bench__head">
          <span className="bench__tag eyebrow">Validated Performance</span>
          <span className="bench__prov mono">{BENCHMARK.method} · {BENCHMARK.preset}</span>
        </div>
        <div className="bench__stats">
          <Stat label="Mean Global IDF1" value={BENCHMARK.meanIdf1.toFixed(4)} accent="var(--scan)" />
          <Stat label="Throughput" value={BENCHMARK.fpsPerCam.toFixed(1)} unit="fps/cam" accent="var(--signal)" />
          <Stat label="VRAM" value={BENCHMARK.vramGB.toFixed(1)} unit="GB" />
          <Stat label="Buffered window" value={BENCHMARK.windowSeconds} unit="s" />
          <Stat label="Cameras" value={CAMERAS.length} />
        </div>
        <div className="bench__foot mono">{BENCHMARK.dataset} · {BENCHMARK.hardware}</div>
      </div>

      <div className="dash__zones">
        {ZONES.map((z) => {
          const cams = camsOfZone(z.id)
          const people = cams.reduce((s, c) => s + c.people, 0)
          const online = cams.filter((c) => c.status === 'online').length
          const degraded = cams.some((c) => c.status !== 'online')
          return (
            <button key={z.id} className="zcard panel panel--ticks" onClick={() => go({ view: 'zone', zoneId: z.id })}>
              <div className="zcard__hd">
                <span className="zcard__swatch" style={{ background: z.accent }} />
                <span className="zcard__name hud">{z.name}</span>
                <span className={`zcard__pill ${degraded ? 'is-warn' : 'is-ok'} mono`}>
                  {online}/{cams.length}
                </span>
              </div>
              <div className="zcard__stats">
                <Stat label="In view" value={people} accent="var(--signal)" />
                <Stat label="Global IDF1" value={z.idf1.toFixed(3)} accent={z.idf1 >= 0.8 ? 'var(--scan)' : 'var(--warn)'} />
              </div>
              <Sparkline data={spark(z.idf1 * 10)} color={z.accent} w={210} h={30} />
              <div className="zcard__cams">
                {cams.map((c) => (
                  <span key={c.id} className={`zcard__cdot zcard__cdot--${c.status}`} title={`${c.name} · ${c.status}`} />
                ))}
              </div>
              <p className="zcard__blurb">{z.blurb}</p>
            </button>
          )
        })}
      </div>

      <div className="dash__lower">
        <Panel title="Network Throughput" className="dash__net">
          <div className="dash__netgrid">
            <Stat label="Cameras live" value={`${CAMERAS.filter(c=>c.status==='online').length}`} unit={`/${CAMERAS.length}`} accent="var(--signal)" />
            <Stat label="People tracked" value={totalPeople} accent="var(--ink-strong)" />
            <Stat label="Mean FPS/cam" value={BENCHMARK.fpsPerCam.toFixed(1)} accent="var(--ink-strong)" />
            <Stat label="VRAM" value={BENCHMARK.vramGB.toFixed(1)} unit="GB" accent="var(--ink-strong)" />
          </div>
          <div className="dash__rows">
            {ZONES.map((z) => (
              <div key={z.id} className="dash__zrow">
                <span className="dash__zlabel">{z.name}</span>
                <Bar value={z.idf1} color={z.accent} />
                <span className="mono dash__zval">{z.idf1.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Alert Feed" className="dash__alerts"
          right={<span className="eyebrow" style={{ color: 'var(--alert)' }}>
            {events.filter(e => e.severity === 'alarm').length} active
          </span>}>
          <ul className="alertlist">
            {events.map((e) => (
              <li key={e.id} className={`alertrow alertrow--${e.severity}`}
                onClick={() => go({ view: 'live', zoneId: e.zoneId, cameraId: e.cameraId })}>
                <span className="alertrow__sev" />
                <div className="alertrow__body">
                  <span className="alertrow__msg">{e.message}</span>
                  <span className="alertrow__meta mono">{e.kind.toUpperCase()} · {fmtAgo(e.ts, now)} ago</span>
                </div>
                <StatusDot status={e.severity === 'alarm' ? 'offline' : e.severity === 'warn' ? 'warning' : 'online'} />
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  )
}
