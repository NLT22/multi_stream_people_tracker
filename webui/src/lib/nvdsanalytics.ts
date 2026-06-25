import type { Roi, Pt } from '../data/types'
import { camById } from '../data/zones'

// Generate a DeepStream nvdsanalytics config block that matches the project's
// real configs/analytics/*.txt format. ROIs are stored normalized (0..1) and
// de-normalized back to the 1920×1080 config space here.
const W = 1920
const H = 1080

const px = (p: Pt) => `${Math.round(p.x * W)};${Math.round(p.y * H)}`
const poly = (pts: Pt[]) => pts.map(px).join(';')

function header(): string {
  return [
    '[property]',
    'enable=1',
    `config-width=${W}`,
    `config-height=${H}`,
    'osd-mode=2',
    'display-font-size=14',
    '',
  ].join('\n')
}

function blockFor(roi: Roi, stream: number): string {
  const lines: string[] = []
  switch (roi.kind) {
    case 'detection':
    case 'heatmap':
      lines.push(`[roi-filtering-stream-${stream}]`, 'enable=1',
        `roi-${roi.name}=${poly(roi.points)}`, 'inverse-roi=0', 'class-id=0')
      break
    case 'restricted':
    case 'ignore':
      lines.push(`[roi-filtering-stream-${stream}]`, 'enable=1',
        `roi-${roi.name}=${poly(roi.points)}`, 'inverse-roi=1', 'class-id=0')
      break
    case 'overcrowd':
      lines.push(`[overcrowding-stream-${stream}]`, 'enable=1',
        `roi-${roi.name}=${poly(roi.points)}`,
        `object-threshold=${roi.threshold ?? 6}`, 'class-id=0')
      break
    case 'counting': {
      const dir = roi.direction ?? roi.points
      lines.push(`[line-crossing-stream-${stream}]`, 'enable=1',
        `line-crossing-${roi.name}=${poly(dir)};${poly(roi.points)}`,
        'class-id=0', 'extended=0', 'mode=loose')
      break
    }
  }
  return lines.join('\n')
}

// Config for a single camera (one stream index).
export function generateConfig(rois: Roi[], cameraId: string): string {
  const cam = camById(cameraId)
  const stream = cam?.streamIndex ?? 0
  const active = rois.filter((r) => r.cameraId === cameraId && r.enabled)
  const blocks = active.map((r) => blockFor(r, stream))
  return header() + '\n' + (blocks.length
    ? blocks.join('\n\n')
    : `# no enabled rules on stream-${stream}`) + '\n'
}

export const roiSummary = (r: Roi): string =>
  r.kind === 'counting'
    ? `tripwire · ${r.points.length} pts`
    : r.kind === 'overcrowd'
      ? `polygon · thr ${r.threshold ?? 6}`
      : `polygon · ${r.points.length} pts`
