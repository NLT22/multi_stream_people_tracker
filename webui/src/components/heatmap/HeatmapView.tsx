import { useState } from 'react'
import type { Nav } from '../../App'
import { ZONES, zoneById } from '../../data/zones'
import { Panel } from '../common'
import './heatmap.css'

type Metric = 'occupancy' | 'footfall' | 'dwelltime'
type ViewKey = 'bev' | 'cam_0' | 'cam_1' | 'cam_2' | 'cam_3'

const METRICS: { k: Metric; label: string; blurb: string }[] = [
  { k: 'occupancy', label: 'Occupancy', blurb: 'Total presence time per floor cell.' },
  { k: 'footfall', label: 'Footfall', blurb: 'Distinct persons passing through each cell.' },
  { k: 'dwelltime', label: 'Dwell Time', blurb: 'Occupancy ÷ footfall (Little’s Law).' },
]
const VIEWS: { k: ViewKey; label: string }[] = [
  { k: 'bev', label: 'BEV' }, { k: 'cam_0', label: 'Cam 1' }, { k: 'cam_1', label: 'Cam 2' },
  { k: 'cam_2', label: 'Cam 3' }, { k: 'cam_3', label: 'Cam 4' },
]

export function HeatmapView({ nav, go }: { nav: Nav; go: (n: Partial<Nav>) => void }) {
  const zone = zoneById(nav.zoneId ?? '') ?? ZONES[0]
  const [metric, setMetric] = useState<Metric>('occupancy')
  const [view, setView] = useState<ViewKey>('bev')
  const [opacity, setOpacity] = useState(0.78)
  const [range, setRange] = useState(60)

  const heat = `/heatmaps/${zone.scene}/${view}_${metric}.png`
  const underlay = view === 'bev' ? null
    : `/frames/orig_${zone.scene}_cam${Number(view.split('_')[1]) + 1}.jpg`

  return (
    <div className="hm">
      <div className="hm__stagecol">
        <div className="hm__head">
          <div>
            <h2 style={{ fontSize: 18 }}>Density Heatmaps</h2>
            <span className="eyebrow">{zone.name} · {view === 'bev' ? 'bird’s-eye floor plane' : VIEWS.find(v=>v.k===view)!.label}</span>
          </div>
          <div className="hm__zsel">
            {ZONES.map((z) => (
              <button key={z.id} className={`hm__ztab ${z.id === zone.id ? 'is-active' : ''}`}
                onClick={() => go({ zoneId: z.id })}>
                <i style={{ background: z.accent }} />{z.name}
              </button>
            ))}
          </div>
        </div>

        <div className="hm__views">
          {VIEWS.map((v) => (
            <button key={v.k} className={`hm__viewbtn ${view === v.k ? 'is-active' : ''}`}
              onClick={() => setView(v.k)}>{v.label}</button>
          ))}
        </div>

        <div className="hm__stage">
          {underlay && <img className="hm__underlay" src={underlay} alt="" />}
          <img className="hm__heat" src={heat} alt={`${metric} heatmap`} style={{ opacity }}
            onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} />
          <div className="hm__ramp" aria-hidden>
            <span className="mono">low</span>
            <div className="hm__rampbar" />
            <span className="mono">high</span>
          </div>
        </div>
      </div>

      <aside className="hm__side">
        <Panel title="Metric">
          <div className="hm__metrics">
            {METRICS.map((m) => (
              <button key={m.k} className={`hm__metric ${metric === m.k ? 'is-active' : ''}`}
                onClick={() => setMetric(m.k)}>
                <span className="hm__metricname">{m.label}</span>
                <span className="hm__metricblurb">{m.blurb}</span>
              </button>
            ))}
          </div>
        </Panel>

        <Panel title="Overlay">
          <label className="hm__field">
            <div className="hm__fieldhd"><span className="eyebrow">Opacity</span><span className="mono">{Math.round(opacity * 100)}%</span></div>
            <input type="range" min={0} max={1} step={0.02} value={opacity}
              onChange={(e) => setOpacity(+e.target.value)} />
          </label>
          <label className="hm__field">
            <div className="hm__fieldhd"><span className="eyebrow">Time window</span><span className="mono">last {range}m</span></div>
            <input type="range" min={5} max={120} step={5} value={range}
              onChange={(e) => setRange(+e.target.value)} />
          </label>
          <div className="hm__btns">
            <button onClick={() => { setOpacity(0.78); setRange(60) }}>Reset</button>
            <a href={heat} download className="hm__export">Export PNG</a>
          </div>
        </Panel>

        <Panel title="Correlation vs GT">
          <div className="hm__cc">
            <span className="hud hm__ccval">{ccFor(zone.key)}</span>
            <span className="eyebrow">Pearson CC (occupancy)</span>
          </div>
          <p className="hm__note">Heatmaps stay accurate even where Global IDF1 is low — occupancy needs correct
            placement, not identity. Footfall &amp; dwell depend more on stable IDs.</p>
        </Panel>
      </aside>
    </div>
  )
}

const ccFor = (k: string) => ({ cafe: '0.99', lobby: '0.98', office: '0.97', industry: '0.98', retail: '0.97' } as Record<string, string>)[k] ?? '0.98'
