import type { ReactNode } from 'react'
import type { CamStatus } from '../data/types'
import './common.css'

export function Panel({ title, right, children, ticks = true, className = '' }: {
  title?: string; right?: ReactNode; children: ReactNode; ticks?: boolean; className?: string
}) {
  return (
    <section className={`panel ${ticks ? 'panel--ticks' : ''} c-panel ${className}`}>
      {title && (
        <header className="c-panel__hd">
          <span className="eyebrow">{title}</span>
          {right}
        </header>
      )}
      <div className="c-panel__body">{children}</div>
    </section>
  )
}

const STATUS_LABEL: Record<CamStatus, string> = {
  online: 'ONLINE', warning: 'DEGRADED', offline: 'OFFLINE',
}
export function StatusDot({ status, label }: { status: CamStatus; label?: boolean }) {
  return (
    <span className={`c-dot c-dot--${status}`}>
      <i />{label && <em className="mono">{STATUS_LABEL[status]}</em>}
    </span>
  )
}

export function Stat({ label, value, unit, accent }: {
  label: string; value: ReactNode; unit?: string; accent?: string
}) {
  return (
    <div className="c-stat">
      <div className="c-stat__v hud" style={accent ? { color: accent } : undefined}>
        {value}{unit && <span className="c-stat__u mono">{unit}</span>}
      </div>
      <div className="c-stat__l eyebrow">{label}</div>
    </div>
  )
}

// Lightweight inline sparkline (no deps).
export function Sparkline({ data, color = 'var(--signal)', w = 80, h = 22 }: {
  data: number[]; color?: string; w?: number; h?: number
}) {
  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const span = max - min || 1
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((d - min) / span) * (h - 3) - 1.5
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg className="c-spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

export function Bar({ value, color = 'var(--signal)' }: { value: number; color?: string }) {
  return (
    <div className="c-bar">
      <span style={{ width: `${Math.round(value * 100)}%`, background: color }} />
    </div>
  )
}
