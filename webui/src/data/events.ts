import type { AlertEvent } from './types'

// Seed alert feed. Timestamps are relative offsets (seconds ago) resolved to
// wall-clock at load so the ticker reads as a live stream.
interface Seed { ago: number; zoneId: string; cameraId: string;
  kind: AlertEvent['kind']; severity: AlertEvent['severity']; message: string }

const SEEDS: Seed[] = [
  { ago: 12,  zoneId: 'zone-retail',   cameraId: 'cam-19', kind: 'offline',   severity: 'alarm', message: 'Stream lost — RETAIL · Stockroom (no packets 41s)' },
  { ago: 38,  zoneId: 'zone-industry', cameraId: 'cam-12', kind: 'intrusion', severity: 'alarm', message: 'Intrusion in Loading-Bay — track G#1180' },
  { ago: 64,  zoneId: 'zone-retail',   cameraId: 'cam-16', kind: 'overcrowd', severity: 'warn',  message: 'Overcrowd: Aisle-1 at 9 / threshold 8' },
  { ago: 95,  zoneId: 'zone-cafe',     cameraId: 'cam-0',  kind: 'linecross', severity: 'info',  message: 'Aisle tripwire +1 (in) — hourly 47' },
  { ago: 140, zoneId: 'zone-industry', cameraId: 'cam-14', kind: 'offline',   severity: 'warn',  message: 'Degraded FPS 9.1 — IND · Loading' },
  { ago: 190, zoneId: 'zone-office',   cameraId: 'cam-10', kind: 'dwell',     severity: 'info',  message: 'Dwell > 8m in Corridor — track G#842' },
  { ago: 240, zoneId: 'zone-retail',   cameraId: 'cam-17', kind: 'overcrowd', severity: 'warn',  message: 'Overcrowd cleared: Aisle-2 back to 6' },
  { ago: 305, zoneId: 'zone-lobby',    cameraId: 'cam-4',  kind: 'linecross', severity: 'info',  message: 'North tripwire +3 (in) — hourly 212' },
]

export function seedEvents(now: number): AlertEvent[] {
  return SEEDS.map((s, i) => ({
    id: `evt-${i}`,
    ts: now - s.ago * 1000,
    zoneId: s.zoneId,
    cameraId: s.cameraId,
    kind: s.kind,
    severity: s.severity,
    message: s.message,
  }))
}
