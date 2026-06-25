import type { AnalyticsDef, AnalyticsKey } from './types'

// The seven analytics functions exposed per camera. Glyphs are single chars
// rendered in the HUD face so the panel needs no icon assets.
export const ANALYTICS: AnalyticsDef[] = [
  { key: 'heatmap',     label: 'Heatmap',        glyph: '▦', accent: 'var(--scan)',
    blurb: 'Accumulate occupancy / footfall / dwell density over the floor plane.' },
  { key: 'counting',    label: 'Object Counting', glyph: '#', accent: 'var(--signal)',
    blurb: 'Count persons currently inside a region of interest.' },
  { key: 'lineCrossing', label: 'Line Crossing',  glyph: '⇄', accent: 'var(--signal)',
    blurb: 'Tally directional crossings over a virtual tripwire.' },
  { key: 'intrusion',   label: 'Region Intrusion', glyph: '⚠', accent: 'var(--alert)',
    blurb: 'Raise an alarm when a person enters a restricted polygon.' },
  { key: 'direction',   label: 'Direction',       glyph: '↗', accent: 'var(--warn)',
    blurb: 'Flag motion against an expected flow direction.' },
  { key: 'dwell',       label: 'Dwell Time',      glyph: '◷', accent: 'var(--warn)',
    blurb: 'Measure how long each person lingers within a zone.' },
  { key: 'occupancy',   label: 'Occupancy',       glyph: '◉', accent: 'var(--scan)',
    blurb: 'Estimate live occupancy and trip an overcrowding threshold.' },
]

export const analyticsByKey = (k: AnalyticsKey) =>
  ANALYTICS.find((a) => a.key === k)!
