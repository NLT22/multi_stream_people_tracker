import { useEffect, useState } from 'react'

// Ticking wall clock for the top bar. Returns a Date that updates each second.
export function useClock(): Date {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

export const fmtClock = (d: Date) =>
  d.toLocaleTimeString('en-GB', { hour12: false })

export const fmtDate = (d: Date) =>
  d.toLocaleDateString('en-CA') // yyyy-mm-dd

export function fmtAgo(ts: number, now: number): string {
  const s = Math.max(0, Math.floor((now - ts) / 1000))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  return `${Math.floor(s / 3600)}h`
}
