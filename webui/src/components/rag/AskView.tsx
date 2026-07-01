import { useEffect, useRef, useState } from 'react'
import type { Nav } from '../../App'
import { Panel } from '../common'
import { getActiveDatasetKey } from '../../data/zones'
import {
  ask, searchImage, topZones, personTimeline, personTrajectory, personDwell,
  type AskResult, type PersonCandidate, type ZoneRank,
  type TimelineInterval, type BevPoint,
} from '../../api/rag'
import './ask.css'

const EXAMPLES = [
  'Which zones got the most footfall today?',
  'How long did person 3 dwell in each zone?',
  'When and where did global id 5 appear?',
  'Top 5 busiest areas this week',
]

const hhmm = (t: number) => new Date(t * 1000).toLocaleTimeString([], {
  hour: '2-digit', minute: '2-digit', second: '2-digit',
})

interface Turn { role: 'user' | 'assistant'; text: string; tools?: AskResult['tool_calls']; pending?: boolean }

export function AskView({ nav: _nav, go: _go }: { nav: Nav; go: (n: Partial<Nav>) => void }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [q, setQ] = useState('')
  const [img, setImg] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const send = async (text: string) => {
    if (!text.trim() || busy) return
    setTurns((t) => [...t, { role: 'user', text }, { role: 'assistant', text: '', pending: true }])
    setQ(''); setBusy(true)
    try {
      const res = await ask(text, img)
      setTurns((t) => t.map((turn, i) =>
        i === t.length - 1 ? { role: 'assistant', text: res.answer, tools: res.tool_calls } : turn))
    } catch (e) {
      setTurns((t) => t.map((turn, i) =>
        i === t.length - 1
          ? { role: 'assistant', text: `Could not reach the RAG service — is it running? (${String(e)})` }
          : turn))
    } finally {
      setBusy(false); setImg(null)
      requestAnimationFrame(() => scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight))
    }
  }

  return (
    <div className="ask">
      <div className="ask__head">
        <div>
          <h2 style={{ fontSize: 18 }}>Ask the Data</h2>
          <span className="eyebrow">Natural-language Q&amp;A over tracking metadata · identity · trajectories · zones</span>
        </div>
      </div>

      <div className="ask__grid">
        <Panel title="Conversation" className="ask__chat">
          <div className="ask__log" ref={scrollRef}>
            {turns.length === 0 && (
              <div className="ask__empty">
                <p className="mono">Ask about people, zones, dwell time, or footfall.</p>
                <div className="ask__chips">
                  {EXAMPLES.map((ex) => (
                    <button key={ex} className="ask__chip" onClick={() => send(ex)}>{ex}</button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={`ask__turn ask__turn--${t.role}`}>
                <span className="ask__who eyebrow">{t.role === 'user' ? 'YOU' : 'SENTINEL'}</span>
                {t.pending ? <span className="ask__dots mono">analysing…</span>
                  : <div className="ask__msg">{t.text}</div>}
                {t.tools && t.tools.length > 0 && <ToolTrace calls={t.tools} />}
              </div>
            ))}
          </div>

          <div className="ask__composer">
            {img && <span className="ask__imgtag mono">📎 {img.name}<button onClick={() => setImg(null)}>×</button></span>}
            <label className="ask__attach" title="Attach a person crop">
              ＋<input type="file" accept="image/*" hidden
                onChange={(e) => setImg(e.target.files?.[0] ?? null)} />
            </label>
            <textarea
              className="ask__input mono" rows={1} value={q} placeholder="Ask a question…"
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(q) } }}
            />
            <button className="ask__send" disabled={busy || !q.trim()} onClick={() => send(q)}>Send</button>
          </div>
        </Panel>

        <div className="ask__side">
          <PersonSearch />
          <TopZones />
        </div>
      </div>
    </div>
  )
}

function ToolTrace({ calls }: { calls: AskResult['tool_calls'] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="ask__trace">
      <button className="ask__tracetoggle mono" onClick={() => setOpen((o) => !o)}>
        {open ? '▾' : '▸'} {calls.length} tool call{calls.length > 1 ? 's' : ''}
      </button>
      {open && calls.map((c, i) => (
        <pre key={i} className="ask__traceitem mono">{c.tool}({JSON.stringify(c.args)})</pre>
      ))}
    </div>
  )
}

function PersonSearch() {
  const [cands, setCands] = useState<PersonCandidate[] | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [timeline, setTimeline] = useState<TimelineInterval[]>([])
  const [bev, setBev] = useState<BevPoint[]>([])
  const [dwell, setDwell] = useState<{ zone: string; seconds: number; visits: number }[]>([])
  const [err, setErr] = useState<string | null>(null)

  const onFile = async (f: File | null) => {
    if (!f) return
    setErr(null); setCands(null); setSel(null)
    try { setCands(await searchImage(f, 5)) }
    catch (e) { setErr(String(e)) }
  }
  const pick = async (gid: number) => {
    setSel(gid)
    try {
      const [tl, tr, dw] = await Promise.all([
        personTimeline(gid), personTrajectory(gid, 4), personDwell(gid)])
      setTimeline(tl.intervals); setBev(tr.points); setDwell(dw.dwell)
    } catch (e) { setErr(String(e)) }
  }

  return (
    <Panel title="Person Search" right={<span className="eyebrow">image → identity</span>}>
      <label className="ask__drop">
        <span className="mono">Drop / choose a person crop</span>
        <input type="file" accept="image/*" hidden onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      </label>
      {err && <p className="ask__err mono">{err}</p>}
      {cands && (
        <div className="ask__cands">
          {cands.map((c) => (
            <button key={c.global_id} className={`ask__cand ${sel === c.global_id ? 'is-active' : ''}`}
              onClick={() => pick(c.global_id)}>
              <span className="hud">ID {c.global_id}</span>
              <span className="mono">{(c.score * 100).toFixed(0)}%</span>
            </button>
          ))}
        </div>
      )}
      {sel !== null && (
        <div className="ask__person">
          <div className="eyebrow">ID {sel} · dwell by zone</div>
          {dwell.length === 0 && <p className="mono ask__muted">no dwell records</p>}
          {dwell.map((d) => (
            <div key={d.zone} className="ask__dwellrow">
              <span className="mono">{d.zone}</span>
              <span className="ask__dwellbar"><i style={{
                width: `${Math.min(100, d.seconds / Math.max(...dwell.map(x => x.seconds), 1) * 100)}%` }} /></span>
              <span className="mono">{d.seconds.toFixed(0)}s</span>
            </div>
          ))}
          {bev.length > 1 && <BevPath points={bev} map={bevMapFor()} />}
          {timeline.length > 0 && (
            <div className="ask__timeline">
              <div className="eyebrow">appearances</div>
              {timeline.slice(0, 8).map((iv, i) => (
                <div key={i} className="ask__tlrow mono">
                  <span>cam {iv.cam} · {iv.zone}</span>
                  <span className="ask__muted">{hhmm(iv.t_start)}–{hhmm(iv.t_end)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

interface BevMap { url: string; sf: number; tx: number; ty: number; w: number; h: number }

// ponytail: W022 floor map + metric world->pixel transform from calibration.json
// (map_px = (world + translation) * scaleFactor). Add per-warehouse entries if more land.
const MTMC_BEV: BevMap = {
  url: '/maps/warehouse_022.png', sf: 21.586118551949482,
  tx: 102.84569562990934, ty: 100.24632708444943, w: 1920, h: 1080,
}
const bevMapFor = (): BevMap | undefined =>
  getActiveDatasetKey() === 'mtmc' ? MTMC_BEV : undefined

function BevPath({ points, map }: { points: BevPoint[]; map?: BevMap }) {
  if (map) {
    // draw the trajectory on the real warehouse floor map using the metric transform
    const px = (p: BevPoint) => (p.x + map.tx) * map.sf
    const py = (p: BevPoint) => (p.y + map.ty) * map.sf
    // drop back-projection outliers that fall outside the floor map
    const inb = points.filter((p) => px(p) >= 0 && px(p) <= map.w && py(p) >= 0 && py(p) <= map.h)
    const pts = inb.length > 1 ? inb : points
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${px(p).toFixed(1)},${py(p).toFixed(1)}`).join(' ')
    return (
      <div className="ask__bev">
        <div className="eyebrow">world trajectory (BEV · floor map)</div>
        <svg viewBox={`0 0 ${map.w} ${map.h}`} className="ask__bevsvg" preserveAspectRatio="xMidYMid meet">
          <image href={map.url} x={0} y={0} width={map.w} height={map.h} />
          <path d={d} fill="none" stroke="var(--signal)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
          <circle cx={px(pts[0])} cy={py(pts[0])} r={22} fill="var(--signal)" stroke="#fff" strokeWidth={4} />
          <circle cx={px(pts[pts.length - 1])} cy={py(pts[pts.length - 1])} r={22} fill="var(--alert)" stroke="#fff" strokeWidth={4} />
        </svg>
      </div>
    )
  }
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
  const W = 220, H = 130, pad = 8
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad)
  const sy = (y: number) => pad + ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad)
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')
  return (
    <div className="ask__bev">
      <div className="eyebrow">world trajectory (BEV)</div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="ask__bevsvg">
        <path d={d} fill="none" stroke="var(--scan)" strokeWidth="1.5" />
        <circle cx={sx(points[0].x)} cy={sy(points[0].y)} r="3" fill="var(--signal)" />
        <circle cx={sx(points[points.length - 1].x)} cy={sy(points[points.length - 1].y)} r="3" fill="var(--alert)" />
      </svg>
    </div>
  )
}

function TopZones() {
  const [metric, setMetric] = useState<'footfall' | 'occupancy'>('footfall')
  const [rows, setRows] = useState<ZoneRank[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const load = async (m: 'footfall' | 'occupancy') => {
    setMetric(m); setErr(null)
    try { setRows((await topZones(m, 6)).top) } catch (e) { setErr(String(e)) }
  }
  // initial load in an effect, not during render (render-time setState infinite-loops)
  useEffect(() => { void load('footfall') }, [])
  const max = Math.max(...(rows ?? []).map((r) => r.value), 1)
  return (
    <Panel title="Top Zones" right={
      <div className="ask__metricsel">
        {(['footfall', 'occupancy'] as const).map((m) => (
          <button key={m} className={`ask__mtab ${metric === m ? 'is-active' : ''}`}
            onClick={() => load(m)}>{m}</button>
        ))}
      </div>}>
      {err && <p className="ask__err mono">{err}</p>}
      {(rows ?? []).map((r) => (
        <div key={r.zone} className="ask__zrow">
          <span className="mono ask__zname">{r.zone}</span>
          <span className="ask__zbar"><i style={{ width: `${(r.value / max) * 100}%` }} /></span>
          <span className="mono">{metric === 'occupancy' ? `${r.value.toFixed(0)}s` : r.value}</span>
        </div>
      ))}
      {rows && rows.length === 0 && <p className="mono ask__muted">no zone data in this run</p>}
    </Panel>
  )
}
