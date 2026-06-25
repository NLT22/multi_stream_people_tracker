import { useEffect, useRef } from 'react'
import type { Camera } from '../../data/types'
import { StatusDot } from '../common'
import { TrackOverlay } from './TrackOverlay'
import { useLiveMosaic, LiveCell } from './LiveMosaic'
import './live.css'

export function CameraTile({ cam, onOpen, dense = false, showOverlay = true }: {
  cam: Camera; onOpen?: (c: Camera) => void; dense?: boolean; showOverlay?: boolean
}) {
  const offline = cam.status === 'offline'
  const { live } = useLiveMosaic()
  const videoRef = useRef<HTMLVideoElement | null>(null)

  // Stagger playback start a touch so many tiles don't all decode the same frame
  // at once, and guarantee muted autoplay even if the browser is picky.
  useEffect(() => {
    const v = videoRef.current
    if (!v || !cam.feed) return
    v.muted = true
    const t = setTimeout(() => v.play().catch(() => {}), (cam.streamIndex % 4) * 120)
    return () => clearTimeout(t)
  }, [cam.feed, cam.streamIndex])

  return (
    <button
      className={`tile ${dense ? 'tile--dense' : ''} ${offline ? 'tile--offline' : ''}`}
      onClick={() => onOpen?.(cam)}
      aria-label={`Open ${cam.name}`}
    >
      <div className="tile__feed">
        {/* When the real stream is up, every camera is genuinely live — the
            mock 'offline' seed only applies to the REPLAY demo. */}
        {live ? (
          <>
            {/* genuine live cell, cropped from the one 20-cam pipeline HLS stream */}
            <LiveCell index={cam.streamIndex} className="tile__cell" />
            <span className="tile__live mono"><i />LIVE</span>
            <span className="tile__cid mono">CAM{String(cam.streamIndex).padStart(2, '0')}</span>
          </>
        ) : offline ? (
          <div className="tile__lost">
            <span className="hud">SIGNAL LOST</span>
            <span className="mono">no packets · 41s</span>
          </div>
        ) : cam.feed ? (
          <>
            {/* real per-camera pipeline OSD (cropped from the zone's tiled output) */}
            <video ref={videoRef} src={cam.feed} poster={cam.frame}
              autoPlay loop muted playsInline preload="metadata" draggable={false} />
            <span className="tile__replay mono"><i />REPLAY</span>
            <span className="tile__cid mono">CAM{String(cam.streamIndex).padStart(2, '0')}</span>
          </>
        ) : (
          <>
            {/* fallback: still frame + synthetic overlay (no real feed available) */}
            <img src={cam.frame} alt="" loading="lazy" draggable={false} />
            {showOverlay && (
              <TrackOverlay seed={cam.streamIndex + 1}
                count={Math.max(2, Math.min(cam.people, 10))}
                showTrails={!dense} showIds={!dense} />
            )}
            <span className="tile__replay mono"><i />REPLAY</span>
            <span className="tile__cid mono">CAM{String(cam.streamIndex).padStart(2, '0')}</span>
          </>
        )}
      </div>

      <div className="tile__bar">
        <StatusDot status={live ? 'online' : cam.status} />
        <span className="tile__name">{cam.name}</span>
        {!dense && (
          <span className="tile__meta mono">
            {offline ? '—' : `${cam.fps.toFixed(1)} fps`}
            <span className="tile__people">◍ {cam.people}</span>
          </span>
        )}
      </div>
    </button>
  )
}
