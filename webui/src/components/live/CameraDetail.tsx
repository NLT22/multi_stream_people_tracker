import { useState } from 'react'
import type { Nav } from '../../App'
import type { Camera } from '../../data/types'
import { zoneById } from '../../data/zones'
import { Panel, StatusDot, Bar } from '../common'
import { TrackOverlay } from './TrackOverlay'
import { useLiveMosaic, LiveCell } from './LiveMosaic'

// Detailed single-camera view. Plays the real pipeline OSD video when present,
// otherwise the still frame with the synthetic overlay.
export function CameraDetail({ cam, go, onClose }: {
  cam: Camera; go: (n: Partial<Nav>) => void; onClose: () => void
}) {
  const zone = zoneById(cam.zoneId)!
  const { live } = useLiveMosaic()
  const [mode, setMode] = useState<'osd' | 'still'>(cam.feed ? 'osd' : 'still')

  return (
    <div className="detail">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>
        <button className="detail__back" onClick={onClose}>← back to wall</button>
        <div className="detail__stage">
          {live ? (
            <LiveCell index={cam.streamIndex} className="detail__cell" />
          ) : mode === 'osd' && cam.feed ? (
            <video src={cam.feed} autoPlay loop muted playsInline />
          ) : (
            <>
              {cam.status === 'offline'
                ? <div className="tile__lost" style={{ position: 'static', width: '100%', height: '100%' }}>
                    <span className="hud">SIGNAL LOST</span></div>
                : <img src={cam.frame} alt={cam.name} />}
              {cam.status !== 'offline' && (
                <div className="detail__overlaywrap">
                  <TrackOverlay seed={cam.streamIndex + 1} count={Math.max(2, cam.people)} />
                </div>
              )}
            </>
          )}
          {live
            ? <span className="tile__live mono"><i />LIVE · DEEPSTREAM OSD</span>
            : cam.status !== 'offline' && <span className="tile__replay mono"><i />REPLAY · BUFFERED ID</span>}
        </div>
        {cam.feed && !live && (
          <div className="wall__layouts" style={{ alignSelf: 'flex-start' }}>
            <button className={`wall__lbtn ${mode === 'osd' ? 'is-active' : ''}`} onClick={() => setMode('osd')}>PIPELINE OSD</button>
            <button className={`wall__lbtn ${mode === 'still' ? 'is-active' : ''}`} onClick={() => setMode('still')}>SYNTH OVERLAY</button>
          </div>
        )}
      </div>

      <div className="detail__side">
        <Panel title="Camera">
          <h3 style={{ fontSize: 15, marginBottom: 8 }}>{cam.name}</h3>
          <div className="detail__rows">
            <Row k="Status" v={<StatusDot status={live ? "online" : cam.status} label />} />
            <Row k="Zone" v={zone.name} />
            <Row k="Stream ID" v={<span className="mono">source-{cam.streamIndex}</span>} />
            <Row k="Resolution" v={<span className="mono">{cam.resolution}</span>} />
            <Row k="Throughput" v={<span className="mono">{cam.fps.toFixed(1)} / {cam.targetFps} fps</span>} />
            <Row k="Tracked now" v={<span className="mono" style={{ color: 'var(--signal)' }}>{cam.people}</span>} />
          </div>
        </Panel>

        <Panel title="Stream Health">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
            <span className="eyebrow">Link quality</span>
            <span className="mono" style={{ color: cam.health > 0.8 ? 'var(--signal)' : cam.health > 0.5 ? 'var(--warn)' : 'var(--alert)' }}>
              {Math.round(cam.health * 100)}%
            </span>
          </div>
          <Bar value={cam.health} color={cam.health > 0.8 ? 'var(--signal)' : cam.health > 0.5 ? 'var(--warn)' : 'var(--alert)'} />
        </Panel>

        <Panel title="Actions">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <ActionBtn label="Configure ROIs on this camera" onClick={() => go({ view: 'roi', cameraId: cam.id, zoneId: cam.zoneId })} />
            <ActionBtn label="Open zone analytics" onClick={() => go({ view: 'zone', zoneId: cam.zoneId })} />
            <ActionBtn label="View heatmaps" onClick={() => go({ view: 'heatmap', zoneId: cam.zoneId })} />
          </div>
        </Panel>
      </div>
    </div>
  )
}

const Row = ({ k, v }: { k: string; v: React.ReactNode }) => (
  <div className="drow"><span>{k}</span><span>{v}</span></div>
)
const ActionBtn = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <button onClick={onClick} style={{
    textAlign: 'left', padding: '9px 11px', background: 'var(--card)',
    border: '1px solid var(--line)', borderRadius: 'var(--r)', fontSize: 12, color: 'var(--ink)',
  }} className="action-btn">{label} →</button>
)
