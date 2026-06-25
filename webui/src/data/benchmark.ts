import type { ZoneKind } from './types'

// Validated performance = the project's canonical regression anchors (CLAUDE.md):
// honest SINGLE-PASS full-GT (every frame once, no loop, no GT trimming), scored
// with scripts/eval/score_longrun_idf1.py after `live_buffered --once`.
// These are real measured eval results, not display placeholders — the dashboard
// and zone IDF1 read from here so the console reports the actual benchmark.

export interface Benchmark {
  meanIdf1: number
  fpsPerCam: number
  vramGB: number
  windowFrames: number
  windowSeconds: number
  perScene: Record<ZoneKind, number>
  preset: string
  dataset: string
  method: string
  hardware: string
}

export const BENCHMARK: Benchmark = {
  meanIdf1: 0.8109,
  fpsPerCam: 10.6,
  // VRAM is driven by maxTargetsPerStream, not the preset/model (measured 2026-06-25):
  //   reid0   reidType:0 maxTargets:40  = 3.5 GB   ← current default / the live console
  //   reid0   reidType:0 maxTargets:220 = ~9.4 GB  ← the older figure: NOT wrong, just 220
  //   quality reidType:2 maxTargets:40  = 4.2 GB   ← the ReID model itself adds only ~0.7 GB
  //   quality reidType:2 maxTargets:220 = 12.8 GB
  // NvDCF pre-allocates per-target state for maxTargetsPerStream × streams. The 4.4 audit
  // cut reid0's maxTargets 220→40, so today's default sits at 3.5 GB.
  vramGB: 3.5,
  windowFrames: 200,
  windowSeconds: 20,
  perScene: {
    cafe: 0.833,
    lobby: 0.895,
    office: 0.861,
    industry: 0.805,
    retail: 0.660,
  },
  preset: 'reid0 (NvDCF reidType:0 + SGIE Swin)',
  dataset: 'MMPTracking · 20-cam mixed val · 5 environments',
  method: 'honest single-pass full-GT · Global IDF1',
  hardware: 'RTX 5060 Ti 16 GB',
}

// Quality preset, for reference / comparison panels.
export const BENCHMARK_QUALITY = {
  meanIdf1: 0.8132, fpsPerCam: 9.5, vramGB: 12.7,
  preset: 'quality (reidType:2 + SGIE)',
}
