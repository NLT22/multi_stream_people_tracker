import { useEffect, useRef, useState } from 'react'
import Hls from 'hls.js'

type State = 'connecting' | 'live' | 'waiting'

// Plays the pipeline's live HLS output (the tiled OSD canvas with Buffered IDs).
// Uses hls.js where MSE is available; falls back to native HLS (Safari).
// Polls when the manifest isn't up yet so it auto-connects once start-live.sh runs.
export function LivePlayer({ src = '/live/stream.m3u8' }: { src?: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [state, setState] = useState<State>('connecting')

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let hls: Hls | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    const attach = () => {
      if (cancelled) return
      if (Hls.isSupported()) {
        hls = new Hls({ liveSyncDurationCount: 2, lowLatencyMode: true, manifestLoadingMaxRetry: 0 })
        hls.loadSource(`${src}?t=${Date.now()}`)
        hls.attachMedia(video)
        hls.on(Hls.Events.MANIFEST_PARSED, () => { setState('live'); video.play().catch(() => {}) })
        hls.on(Hls.Events.ERROR, (_e, data) => {
          if (data.fatal) { hls?.destroy(); hls = null; setState('waiting'); retry = setTimeout(attach, 2500) }
        })
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src
        video.addEventListener('loadedmetadata', () => { setState('live'); video.play().catch(() => {}) })
        video.addEventListener('error', () => { setState('waiting'); retry = setTimeout(attach, 2500) })
      }
    }
    attach()
    return () => { cancelled = true; if (retry) clearTimeout(retry); hls?.destroy() }
  }, [src])

  return (
    <div className="liveplayer">
      <video ref={videoRef} muted playsInline className="liveplayer__video" />
      <div className="liveplayer__hud">
        <span className={`liveplayer__badge liveplayer__badge--${state === 'live' ? 'on' : 'off'}`}>
          <i />{state === 'live' ? 'LIVE · DEEPSTREAM OSD' : state === 'connecting' ? 'CONNECTING' : 'WAITING FOR STREAM'}
        </span>
        <span className="liveplayer__tag mono">buffered ID · anchor-guided</span>
      </div>
      {state !== 'live' && (
        <div className="liveplayer__placeholder">
          <span className="hud">NO LIVE STREAM</span>
          <p>Start the pipeline to feed this player:</p>
          <code className="mono">webui/scripts/start-live.sh</code>
          <span className="liveplayer__note mono">RTSP → DeepStream (buffered-ID OSD) → HLS · auto-connects when up</span>
        </div>
      )}
    </div>
  )
}
