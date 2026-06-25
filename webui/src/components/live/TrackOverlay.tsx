import { useEffect, useRef } from 'react'
import { makeTracks } from '../../lib/tracks'

// Self-contained canvas animation of synthetic tracks for one camera tile.
// Runs its own rAF loop — never triggers a React re-render. Replace makeTracks
// with a live bbox stream to show real detections.
export function TrackOverlay({ seed, count, showTrails = true, showIds = true }: {
  seed: number; count: number; showTrails?: boolean; showIds?: boolean
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const tracks = makeTracks(seed, count)
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const trails: Record<number, { x: number; y: number }[]> = {}
    let raf = 0
    const t0 = performance.now()

    const resize = () => {
      const r = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = r.width * dpr
      canvas.height = r.height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const draw = (now: number) => {
      const t = reduced ? 8 : (now - t0) / 1000
      const r = canvas.getBoundingClientRect()
      const W = r.width, H = r.height
      ctx.clearRect(0, 0, W, H)

      for (const tr of tracks) {
        const p = tr.at(t)
        const bw = p.w * W, bh = p.h * H
        const bx = p.x * W - bw / 2, by = p.y * H - bh / 2
        const col = `hsl(${tr.hue} 75% 60%)`

        // trail of foot points
        if (showTrails) {
          const foot = { x: p.x * W, y: by + bh }
          const buf = (trails[tr.gid] ||= [])
          buf.push(foot)
          if (buf.length > 26) buf.shift()
          ctx.beginPath()
          buf.forEach((pt, i) => i ? ctx.lineTo(pt.x, pt.y) : ctx.moveTo(pt.x, pt.y))
          ctx.strokeStyle = `hsl(${tr.hue} 70% 55% / 0.35)`
          ctx.lineWidth = 1.5
          ctx.stroke()
        }

        // bbox
        ctx.strokeStyle = col
        ctx.lineWidth = 1.5
        ctx.strokeRect(bx, by, bw, bh)
        // corner ticks (HUD)
        const k = 5
        ctx.beginPath()
        ctx.moveTo(bx, by + k); ctx.lineTo(bx, by); ctx.lineTo(bx + k, by)
        ctx.moveTo(bx + bw - k, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + k)
        ctx.lineWidth = 2; ctx.stroke()

        // id chip
        if (showIds) {
          const label = `G${tr.gid}`
          ctx.font = '600 9px "JetBrains Mono", monospace'
          const tw = ctx.measureText(label).width + 8
          ctx.fillStyle = col
          ctx.fillRect(bx, by - 12, tw, 11)
          ctx.fillStyle = '#06090d'
          ctx.fillText(label, bx + 4, by - 3.5)
        }
      }
      if (!reduced) raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [seed, count, showTrails, showIds])

  return <canvas ref={ref} className="track-overlay" aria-hidden />
}
