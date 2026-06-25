import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import Hls from 'hls.js'

// The pipeline emits ONE tiled mosaic over HLS (the 20-cam live wall). We decode
// it exactly once into a single hidden <video>, then every camera tile draws its
// own cell from that shared frame onto a <canvas> — 1 decode, 20 live views.
//
// Grid is fixed to the 20-cam mixed layout: nvmultistreamtiler packs source_id
// row-major into 5 columns × 4 rows, so camera streamIndex i → cell (i//5, i%5),
// which lines up 1:1 with data/zones.ts stream indices (cafe 0-3 … retail 16-19).
export const MOSAIC_COLS = 5
export const MOSAIC_ROWS = 4

interface Ctx { video: HTMLVideoElement | null; live: boolean }
const LiveMosaicCtx = createContext<Ctx>({ video: null, live: false })
export const useLiveMosaic = () => useContext(LiveMosaicCtx)

export function LiveMosaicProvider({ src = '/live/stream.m3u8', children }: {
  src?: string; children: ReactNode
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [live, setLive] = useState(false)

  useEffect(() => {
    const video = document.createElement('video')
    video.muted = true; video.playsInline = true; video.crossOrigin = 'anonymous'
    videoRef.current = video
    let hls: Hls | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    // Fall back to REPLAY whenever the stream isn't genuinely live-and-advancing:
    // a finished/stale playlist (leftover segments after start-live.sh stops) or a
    // stall would otherwise play briefly then freeze/blank. Retry so it reconnects
    // automatically once a real stream comes up.
    const drop = () => {
      if (cancelled) return
      setLive(false)
      hls?.destroy(); hls = null
      if (retry) clearTimeout(retry)
      retry = setTimeout(attach, 3000)
    }
    const onPlaying = () => !cancelled && setLive(true)
    const onEnded = () => drop()
    video.addEventListener('playing', onPlaying)
    video.addEventListener('ended', onEnded)

    // Watchdog: if currentTime stops advancing while we think we're live, drop.
    let lastT = -1, stalls = 0
    const watch = setInterval(() => {
      if (cancelled) return
      if (!video.paused && video.currentTime === lastT) {
        if (++stalls >= 2) drop()
      } else stalls = 0
      lastT = video.currentTime
    }, 2000)

    function attach() {
      if (cancelled) return
      if (Hls.isSupported()) {
        hls = new Hls({ liveSyncDurationCount: 2, lowLatencyMode: true, manifestLoadingMaxRetry: 0 })
        hls.loadSource(`${src}?t=${Date.now()}`)
        hls.attachMedia(video)
        hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(() => {}))
        // A non-live (ENDLIST / VOD) playlist = a finished, stale stream → not live.
        hls.on(Hls.Events.LEVEL_LOADED, (_e, d) => { if (d.details && d.details.live === false) drop() })
        hls.on(Hls.Events.ERROR, (_e, d) => { if (d.fatal) drop() })
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src
        video.addEventListener('loadedmetadata', () => video.play().catch(() => {}))
        video.addEventListener('error', drop)
      }
    }
    attach()
    return () => {
      cancelled = true; if (retry) clearTimeout(retry); clearInterval(watch)
      video.removeEventListener('playing', onPlaying); video.removeEventListener('ended', onEnded)
      hls?.destroy()
    }
  }, [src])

  return (
    <LiveMosaicCtx.Provider value={{ video: videoRef.current, live }}>
      {children}
    </LiveMosaicCtx.Provider>
  )
}

// Draws one camera's cell from the shared mosaic video onto a canvas, each frame.
export function LiveCell({ index, className }: { index: number; className?: string }) {
  const { video, live } = useLiveMosaic()
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !video || !live) return
    const ctx = canvas.getContext('2d')!
    let raf = 0
    const col = index % MOSAIC_COLS, row = Math.floor(index / MOSAIC_COLS)

    const draw = () => {
      const vw = video.videoWidth, vh = video.videoHeight
      if (vw && vh) {
        const cw = vw / MOSAIC_COLS, ch = vh / MOSAIC_ROWS
        const r = canvas.getBoundingClientRect()
        const dpr = Math.min(window.devicePixelRatio || 1, 2)
        if (canvas.width !== Math.round(r.width * dpr)) {
          canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr)
        }
        ctx.drawImage(video, col * cw, row * ch, cw, ch, 0, 0, canvas.width, canvas.height)
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [video, live, index])

  return <canvas ref={ref} className={className} aria-hidden />
}
