// Deterministic synthetic tracks: each "person" follows a smooth Lissajous path
// so a tile looks alive without any backend. Positions are normalized 0..1.
// Swap makeTracks() for a websocket of real bbox metadata later.

export interface SynthTrack {
  gid: number
  hue: number
  /** normalized centre + size at time t (seconds) */
  at: (t: number) => { x: number; y: number; w: number; h: number }
}

// tiny seeded RNG so a given (seed) always yields the same crowd
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0xffffffff
  }
}

export function makeTracks(seed: number, count: number): SynthTrack[] {
  const r = rng(seed * 2654435761)
  const out: SynthTrack[] = []
  for (let i = 0; i < count; i++) {
    const ax = 0.18 + r() * 0.22
    const ay = 0.12 + r() * 0.18
    const cx = 0.25 + r() * 0.5
    const cy = 0.3 + r() * 0.45
    const fx = 0.05 + r() * 0.12
    const fy = 0.05 + r() * 0.12
    const phx = r() * Math.PI * 2
    const phy = r() * Math.PI * 2
    const w = 0.06 + r() * 0.05
    const h = w * (2.1 + r() * 0.5)
    const gid = 100 + Math.floor(r() * 900)
    // hue palette: teal/indigo/amber spread, consistent per gid
    const hue = [168, 248, 38, 210, 320][gid % 5]
    out.push({
      gid,
      hue,
      at: (t: number) => ({
        x: cx + ax * Math.sin(t * fx * 2 * Math.PI + phx),
        y: cy + ay * Math.sin(t * fy * 2 * Math.PI + phy),
        w,
        h,
      }),
    })
  }
  return out
}
