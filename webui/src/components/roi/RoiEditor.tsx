import { useEffect, useMemo, useState } from 'react'
import type { Nav } from '../../App'
import type { AnalyticsKey, Pt, Roi, RoiKind } from '../../data/types'
import { CAMERAS, camById } from '../../data/zones'
import { ANALYTICS } from '../../data/analytics'
import { SEED_ROIS } from '../../data/rois'
import { generateConfig, roiSummary } from '../../lib/nvdsanalytics'
import { Panel } from '../common'
import { RoiCanvas, KIND_COLOR } from './RoiCanvas'
import './roi.css'

const KINDS: { k: RoiKind; label: string }[] = [
  { k: 'detection', label: 'Detection' },
  { k: 'restricted', label: 'Restricted' },
  { k: 'counting', label: 'Counting Line' },
  { k: 'heatmap', label: 'Heatmap' },
  { k: 'ignore', label: 'Ignore' },
  { k: 'overcrowd', label: 'Overcrowd' },
]
const DEFAULT_ANALYTICS: Record<RoiKind, AnalyticsKey[]> = {
  detection: ['counting'], restricted: ['intrusion'], counting: ['lineCrossing'],
  heatmap: ['heatmap'], ignore: [], overcrowd: ['occupancy'],
}

let _seq = 100
const uid = () => `roi-${Date.now().toString(36)}-${_seq++}`

export function RoiEditor({ nav, go, rois, setRois }: {
  nav: Nav; go: (n: Partial<Nav>) => void; rois: Roi[]; setRois: (u: (r: Roi[]) => Roi[]) => void
}) {
  const camsWithFrame = CAMERAS.filter((c) => c.status !== 'offline')
  const [camId, setCamId] = useState(nav.cameraId && camById(nav.cameraId)?.status !== 'offline'
    ? nav.cameraId : camsWithFrame[0].id)
  const cam = camById(camId)!

  const [mode, setMode] = useState<'select' | 'draw'>('select')
  const [drawKind, setDrawKind] = useState<RoiKind>('detection')
  const [draft, setDraft] = useState<Pt[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const camRois = useMemo(() => rois.filter((r) => r.cameraId === camId), [rois, camId])
  const selected = camRois.find((r) => r.id === selectedId) ?? null
  const configText = useMemo(() => generateConfig(rois, camId), [rois, camId])

  // The config is EDITABLE: it follows the drawn regions until you hand-edit the text
  // (then it "detaches"); the ↻ regions button re-syncs. editedCfg=null = follow generated.
  const [editedCfg, setEditedCfg] = useState<string | null>(null)
  const effCfg = editedCfg ?? configText
  useEffect(() => { setEditedCfg(null) }, [camId])   // switch camera -> fresh generated config

  // AUTOSAVE: debounced write of the (possibly hand-edited) config straight to disk
  // (configs/analytics/) via the Vite dev middleware — no manual Export needed.
  const [saveStatus, setSaveStatus] = useState('idle')
  useEffect(() => {
    const name = `nvdsanalytics_${cam.streamIndex}.txt`
    setSaveStatus('saving…')
    const t = setTimeout(() => {
      fetch('/__save-analytics', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name, text: effCfg }),
      })
        .then((r) => r.json())
        .then((j) => setSaveStatus(j.ok ? `autosaved → ${j.path}` : `save failed`))
        .catch(() => setSaveStatus('autosave needs npm run dev'))
    }, 600)
    return () => clearTimeout(t)
  }, [effCfg, cam.streamIndex])

  // Esc cancels a draft; Enter closes a polygon.
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setDraft([]); setMode('select') }
      if (e.key === 'Enter' && mode === 'draw' && drawKind !== 'counting') finishPolygon()
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  })

  const beginDraw = (k: RoiKind) => { setDrawKind(k); setMode('draw'); setDraft([]); setSelectedId(null) }

  const commitRoi = (kind: RoiKind, points: Pt[], direction?: Pt[]) => {
    const name = defaultName(kind, camRois)
    const roi: Roi = {
      id: uid(), name, kind, cameraId: camId, points, direction,
      analytics: [...DEFAULT_ANALYTICS[kind]],
      threshold: kind === 'overcrowd' ? 6 : undefined,
      enabled: true,
    }
    setRois((all) => [...all, roi])
    setDraft([]); setMode('select'); setSelectedId(roi.id)
  }

  const addPoint = (p: Pt) => {
    if (drawKind === 'counting') {
      const next = [...draft, p]
      if (next.length === 2) {
        // auto perpendicular direction through midpoint
        const [a, b] = next
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
        const dx = b.x - a.x, dy = b.y - a.y
        const len = Math.hypot(dx, dy) || 1
        const nx = -dy / len, ny = dx / len
        const dir: Pt[] = [
          { x: clamp(mx - nx * 0.12), y: clamp(my - ny * 0.12) },
          { x: clamp(mx + nx * 0.12), y: clamp(my + ny * 0.12) },
        ]
        commitRoi('counting', next, dir)
      } else setDraft(next)
    } else setDraft((d) => [...d, p])
  }
  const finishPolygon = () => { if (draft.length >= 3) commitRoi(drawKind, draft) }

  const patch = (id: string, fn: (r: Roi) => Roi) =>
    setRois((all) => all.map((r) => (r.id === id ? fn(r) : r)))
  const moveVertex = (id: string, i: number, p: Pt) =>
    patch(id, (r) => ({ ...r, points: r.points.map((q, k) => (k === i ? p : q)) }))
  const moveDir = (id: string, i: number, p: Pt) =>
    patch(id, (r) => ({ ...r, direction: (r.direction ?? []).map((q, k) => (k === i ? p : q)) }))
  const moveBody = (id: string, dx: number, dy: number) =>
    patch(id, (r) => ({
      ...r,
      points: r.points.map((q) => ({ x: clamp(q.x + dx), y: clamp(q.y + dy) })),
      direction: r.direction?.map((q) => ({ x: clamp(q.x + dx), y: clamp(q.y + dy) })),
    }))
  const removeRoi = (id: string) => { setRois((all) => all.filter((r) => r.id !== id)); setSelectedId(null) }
  const resetCam = () => setRois((all) => [
    ...all.filter((r) => r.cameraId !== camId),
    ...SEED_ROIS.filter((r) => r.cameraId === camId),
  ])

  return (
    <div className="roi">
      <div className="roi__stagecol">
        <div className="roi__toolbar">
          <select className="roi__camsel mono" value={camId}
            onChange={(e) => { setCamId(e.target.value); setSelectedId(null); setDraft([]); setMode('select') }}>
            {camsWithFrame.map((c) => <option key={c.id} value={c.id} style={{ background: 'var(--card)', color: 'var(--ink-strong)' }}>
              CAM{String(c.streamIndex).padStart(2, '0')} · {c.name}</option>)}
          </select>
          <span className="roi__div" />
          <div className="roi__modes">
            <button className={`roi__mode ${mode === 'select' ? 'is-active' : ''}`}
              onClick={() => { setMode('select'); setDraft([]) }}>▣ Select</button>
            <button className={`roi__mode ${mode === 'draw' ? 'is-active' : ''}`}
              onClick={() => beginDraw(drawKind)}>✎ Draw</button>
          </div>
          {mode === 'draw' && drawKind !== 'counting' && (
            <button className="roi__finish" onClick={finishPolygon} disabled={draft.length < 3}>
              Finish polygon ⏎
            </button>
          )}
          {mode === 'draw' && (
            <span className="roi__hint mono">
              {drawKind === 'counting' ? 'click 2 points for the tripwire' : 'click to add vertices · Enter to close · Esc cancels'}
            </span>
          )}
        </div>

        <div className="roi__kinds">
          {KINDS.map((k) => (
            <button key={k.k}
              className={`roi__kind ${drawKind === k.k ? 'is-active' : ''}`}
              style={{ '--kc': KIND_COLOR[k.k] } as React.CSSProperties}
              onClick={() => beginDraw(k.k)}>
              <i /> {k.label}
            </button>
          ))}
        </div>

        <div className="roi__stage">
          <RoiCanvas frame={cam.frame} rois={camRois} draft={draft} drawKind={drawKind}
            mode={mode} selectedId={selectedId}
            onAddPoint={addPoint} onSelect={setSelectedId}
            onMoveVertex={moveVertex} onMoveDir={moveDir} onMoveBody={moveBody} />
        </div>
      </div>

      <aside className="roi__side">
        <Panel title="Regions" right={<span className="eyebrow mono">{camRois.length}</span>}>
          <ul className="roi__list">
            {camRois.length === 0 && <li className="roi__empty">No regions yet. Pick a type and draw on the feed.</li>}
            {camRois.map((r) => (
              <li key={r.id} className={`roi__item ${r.id === selectedId ? 'is-sel' : ''}`}
                onClick={() => { setSelectedId(r.id); setMode('select') }}>
                <i style={{ background: KIND_COLOR[r.kind] }} />
                <div className="roi__itembody">
                  <span className="roi__itemname">{r.name}</span>
                  <span className="roi__itemsub mono">{r.kind} · {roiSummary(r)}</span>
                </div>
                <button className={`roi__eye ${r.enabled ? '' : 'is-off'}`}
                  onClick={(e) => { e.stopPropagation(); patch(r.id, (x) => ({ ...x, enabled: !x.enabled })) }}
                  aria-label="Toggle">{r.enabled ? '◉' : '○'}</button>
              </li>
            ))}
          </ul>
        </Panel>

        {selected && (
          <Panel title="Inspector">
            <label className="roi__field">
              <span className="eyebrow">Name</span>
              <input className="mono" value={selected.name}
                onChange={(e) => patch(selected.id, (r) => ({ ...r, name: e.target.value.replace(/[^A-Za-z0-9_-]/g, '') }))} />
            </label>
            <label className="roi__field">
              <span className="eyebrow">Type</span>
              <select className="mono" value={selected.kind}
                onChange={(e) => patch(selected.id, (r) => ({ ...r, kind: e.target.value as RoiKind }))}>
                {KINDS.map((k) => <option key={k.k} value={k.k} style={{ background: 'var(--card)', color: 'var(--ink-strong)' }}>{k.label}</option>)}
              </select>
            </label>
            {selected.kind === 'overcrowd' && (
              <label className="roi__field">
                <span className="eyebrow">Object threshold</span>
                <input type="number" min={1} className="mono" value={selected.threshold ?? 6}
                  onChange={(e) => patch(selected.id, (r) => ({ ...r, threshold: Math.max(1, +e.target.value) }))} />
              </label>
            )}
            {selected.kind === 'counting' && (
              <div className="roi__field">
                <span className="eyebrow">Crossing direction (arrow = counted way)</span>
                <button className="roi__flip" onClick={() => patch(selected.id, (r) => ({
                  ...r, direction: (r.direction ?? r.points).slice().reverse(),
                }))}>⇄ Flip direction</button>
              </div>
            )}
            <div className="roi__field">
              <span className="eyebrow">Assigned analytics</span>
              <div className="roi__chips">
                {ANALYTICS.map((a) => {
                  const on = selected.analytics.includes(a.key)
                  return (
                    <button key={a.key} className={`roi__chip ${on ? 'is-on' : ''}`}
                      style={{ '--ac': a.accent } as React.CSSProperties}
                      onClick={() => patch(selected.id, (r) => ({
                        ...r, analytics: on ? r.analytics.filter((x) => x !== a.key) : [...r.analytics, a.key],
                      }))}>
                      <span className="hud">{a.glyph}</span>{a.label}
                    </button>
                  )
                })}
              </div>
            </div>
            <div className="roi__actions">
              <button className="roi__del" onClick={() => removeRoi(selected.id)}>Delete region</button>
            </div>
          </Panel>
        )}

        <Panel title="nvdsanalytics config — editable · autosaved" right={
          <div className="roi__cfgbtns">
            <button onClick={() => navigator.clipboard?.writeText(effCfg)}>Copy</button>
            <button onClick={() => download(`nvdsanalytics_${cam.streamIndex}.txt`, effCfg)}>Download</button>
            {editedCfg !== null && <button onClick={() => setEditedCfg(null)}>↻ regions</button>}
            <button onClick={resetCam}>Reset</button>
          </div>
        }>
          <div className={`roi__savestatus ${saveStatus.startsWith('autosaved') ? 'is-ok' : ''}`}>
            ● {saveStatus}{editedCfg !== null ? '  ·  hand-edited' : ''}
          </div>
          <textarea className="roi__config mono" value={effCfg} spellCheck={false}
            onChange={(e) => setEditedCfg(e.target.value)} />
        </Panel>

        <button className="roi__gozone" onClick={() => go({ view: 'analytics', zoneId: cam.zoneId })}>
          Open analytics matrix →
        </button>
      </aside>
    </div>
  )
}

const clamp = (v: number) => Math.min(1, Math.max(0, v))
function defaultName(kind: RoiKind, existing: Roi[]): string {
  const base = { detection: 'Zone', restricted: 'Restricted', counting: 'Line',
    heatmap: 'Heat', ignore: 'Ignore', overcrowd: 'OC' }[kind]
  const n = existing.filter((r) => r.kind === kind).length + 1
  return `${base}-${n}`
}
function download(name: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob); a.download = name; a.click()
  URL.revokeObjectURL(a.href)
}
