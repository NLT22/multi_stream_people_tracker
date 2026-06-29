import { useState } from 'react'
import type { Nav } from '../../App'
import type { Camera } from '../../data/types'
import { CAMERAS, ZONES, camsOfZone, camById, zoneById } from '../../data/zones'
import { CameraTile } from './CameraTile'
import { CameraDetail } from './CameraDetail'
import { LivePlayer } from './LivePlayer'

type Layout = 'pipeline' | '1' | '4' | '9' | 'zone' | '20'
const LAYOUTS: { k: Layout; label: string }[] = [
  { k: 'pipeline', label: '◉ PIPELINE LIVE' },
  { k: '1', label: '1×1' }, { k: '4', label: '2×2' }, { k: '9', label: '3×3' },
  { k: 'zone', label: 'ZONE' }, { k: '20', label: 'ALL 20' },
]

export function LiveView({ nav, go }: { nav: Nav; go: (n: Partial<Nav>) => void }) {
  const detailCam = nav.cameraId ? camById(nav.cameraId) : null
  const [layout, setLayout] = useState<Layout>(detailCam ? '1' : 'pipeline')
  const zone = nav.zoneId ? zoneById(nav.zoneId) : ZONES[0]

  if (detailCam && layout === '1') {
    return <CameraDetail cam={detailCam} go={go} onClose={() => go({ cameraId: null })} />
  }

  if (layout === 'pipeline') {
    return (
      <div className="wall">
        <div className="wall__tools">
          <div className="wall__layouts" role="tablist" aria-label="Grid layout">
            {LAYOUTS.map((l) => (
              <button key={l.k} role="tab" aria-selected={layout === l.k}
                className={`wall__lbtn ${layout === l.k ? 'is-active' : ''}`}
                onClick={() => setLayout(l.k)}>{l.label}</button>
            ))}
          </div>
          <span className="wall__spacer" />
          <span className="eyebrow">real DeepStream output · RTSP → HLS</span>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}><LivePlayer /></div>
      </div>
    )
  }

  let cams: Camera[]
  if (layout === '20') cams = CAMERAS
  else if (layout === 'zone') cams = camsOfZone(zone!.id)
  else if (layout === '9') cams = CAMERAS.slice(0, 9)
  else cams = camsOfZone(zone!.id).slice(0, 4)

  const open = (c: Camera) => { setLayout('1'); go({ view: 'live', zoneId: c.zoneId, cameraId: c.id }) }

  return (
    <div className="wall">
      <div className="wall__tools">
        <div className="wall__layouts" role="tablist" aria-label="Grid layout">
          {LAYOUTS.map((l) => (
            <button key={l.k} role="tab" aria-selected={layout === l.k}
              className={`wall__lbtn ${layout === l.k ? 'is-active' : ''}`}
              onClick={() => setLayout(l.k)}>{l.label}</button>
          ))}
        </div>

        {(layout === 'zone' || layout === '4') && (
          <div className="wall__zonechip">
            <i style={{ width: 8, height: 8, borderRadius: 2, background: zone!.accent }} />
            <select
              className="mono"
              value={zone!.id}
              onChange={(e) => go({ zoneId: e.target.value })}
              style={{ background: 'transparent', color: 'var(--ink)', border: 'none', fontSize: 11.5 }}
            >
              {ZONES.map((z) => <option key={z.id} value={z.id} style={{ background: 'var(--card)', color: 'var(--ink-strong)' }}>{z.name}</option>)}
            </select>
          </div>
        )}
        <span className="wall__spacer" />
        <span className="eyebrow">{cams.length} feeds · {cams.filter(c=>c.status==='online').length} online</span>
      </div>

      <div className={`wall__grid wall__grid--${layout}`} style={{ overflowY: 'auto' }}>
        {cams.map((c) => (
          <CameraTile key={c.id} cam={c} onOpen={open} dense={layout === '20' || layout === '9'} />
        ))}
      </div>
    </div>
  )
}
