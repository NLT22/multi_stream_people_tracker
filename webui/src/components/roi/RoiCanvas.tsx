import { useRef } from 'react'
import type { Pt, Roi, RoiKind } from '../../data/types'

// Config space = 1920×1080, matching nvdsanalytics. Points stored normalized;
// rendered into a 1920×1080 viewBox so coords map 1:1 to the emitted config.
const VW = 1920, VH = 1080

export const KIND_COLOR: Record<RoiKind, string> = {
  detection: '#2de2c8', restricted: '#ff4d5e', counting: '#f6a821',
  heatmap: '#7c83ff', ignore: '#687889', overcrowd: '#ff7a9c',
}

interface Drag {
  roiId: string; kind: 'vertex' | 'dir' | 'body'; index: number
  // for body drag: last pointer pos
  last?: Pt
}

export function RoiCanvas({
  frame, rois, draft, drawKind, mode, selectedId,
  onAddPoint, onSelect, onMoveVertex, onMoveDir, onMoveBody,
}: {
  frame: string
  rois: Roi[]
  draft: Pt[]
  drawKind: RoiKind
  mode: 'select' | 'draw'
  selectedId: string | null
  onAddPoint: (p: Pt) => void
  onSelect: (id: string | null) => void
  onMoveVertex: (roiId: string, index: number, p: Pt) => void
  onMoveDir: (roiId: string, index: number, p: Pt) => void
  onMoveBody: (roiId: string, dx: number, dy: number) => void
}) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const drag = useRef<Drag | null>(null)

  const toNorm = (e: { clientX: number; clientY: number }): Pt => {
    const r = svgRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    }
  }

  const onCanvasClick = (e: React.MouseEvent) => {
    if (mode === 'draw') onAddPoint(toNorm(e))
    else if (e.target === svgRef.current) onSelect(null)
  }

  const startDrag = (d: Drag) => (e: React.PointerEvent) => {
    if (mode !== 'select') return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    drag.current = { ...d, last: toNorm(e) }
    onSelect(d.roiId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const p = toNorm(e)
    if (d.kind === 'vertex') onMoveVertex(d.roiId, d.index, p)
    else if (d.kind === 'dir') onMoveDir(d.roiId, d.index, p)
    else if (d.kind === 'body' && d.last) {
      onMoveBody(d.roiId, p.x - d.last.x, p.y - d.last.y)
      d.last = p
    }
  }
  const endDrag = () => { drag.current = null }

  const X = (p: Pt) => p.x * VW
  const Y = (p: Pt) => p.y * VH
  const ptsAttr = (pts: Pt[]) => pts.map((p) => `${X(p)},${Y(p)}`).join(' ')

  return (
    <svg
      ref={svgRef}
      className={`roicanvas roicanvas--${mode}`}
      viewBox={`0 0 ${VW} ${VH}`}
      preserveAspectRatio="xMidYMid meet"
      onClick={onCanvasClick}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
    >
      <image href={frame} x={0} y={0} width={VW} height={VH} preserveAspectRatio="xMidYMid slice" />
      <rect x={0} y={0} width={VW} height={VH} fill="rgba(5,8,12,0.12)" />

      {rois.map((r) => {
        const col = KIND_COLOR[r.kind]
        const sel = r.id === selectedId
        const op = r.enabled ? 1 : 0.4
        if (r.kind === 'counting') {
          const [a, b] = r.points
          const dir = r.direction ?? r.points
          // explicit filled arrowhead at the end of the direction vector so the
          // crossing direction is unmistakable (the dashed line alone was ambiguous)
          const d0 = { x: X(dir[0]), y: Y(dir[0]) }, d1 = { x: X(dir[1]), y: Y(dir[1]) }
          const ang = Math.atan2(d1.y - d0.y, d1.x - d0.x)
          const L = VW * 0.02, AW = VW * 0.011
          const head = [
            `${d1.x},${d1.y}`,
            `${d1.x - L * Math.cos(ang) + AW * Math.sin(ang)},${d1.y - L * Math.sin(ang) - AW * Math.cos(ang)}`,
            `${d1.x - L * Math.cos(ang) - AW * Math.sin(ang)},${d1.y - L * Math.sin(ang) + AW * Math.cos(ang)}`,
          ].join(' ')
          return (
            <g key={r.id} opacity={op}>
              <line x1={X(a)} y1={Y(a)} x2={X(b)} y2={Y(b)} stroke={col} strokeWidth={sel ? 5 : 3}
                vectorEffect="non-scaling-stroke" onPointerDown={startDrag({ roiId: r.id, kind: 'body', index: 0 })}
                style={{ cursor: 'move' }} />
              {/* direction arrow (line + filled arrowhead) */}
              <line x1={d0.x} y1={d0.y} x2={d1.x} y2={d1.y} stroke={col}
                strokeWidth={2} strokeDasharray="6 5" vectorEffect="non-scaling-stroke" opacity={0.85} />
              <polygon points={head} fill={col} opacity={0.95} />
              {sel && r.points.map((p, i) => (
                <Handle key={i} x={X(p)} y={Y(p)} color={col}
                  onDown={startDrag({ roiId: r.id, kind: 'vertex', index: i })} />
              ))}
              {sel && dir.map((p, i) => (
                <Handle key={`d${i}`} x={X(p)} y={Y(p)} color={col} small
                  onDown={startDrag({ roiId: r.id, kind: 'dir', index: i })} />
              ))}
              <text x={X(a)} y={Y(a) - 12} className="roicanvas__lbl" fill={col}>{r.name}</text>
            </g>
          )
        }
        return (
          <g key={r.id} opacity={op}>
            <polygon points={ptsAttr(r.points)} fill={col} fillOpacity={sel ? 0.22 : 0.12}
              stroke={col} strokeWidth={sel ? 4 : 2.5} strokeDasharray={r.kind === 'ignore' ? '10 6' : undefined}
              vectorEffect="non-scaling-stroke" style={{ cursor: 'move' }}
              onPointerDown={startDrag({ roiId: r.id, kind: 'body', index: 0 })} />
            {sel && r.points.map((p, i) => (
              <Handle key={i} x={X(p)} y={Y(p)} color={col}
                onDown={startDrag({ roiId: r.id, kind: 'vertex', index: i })} />
            ))}
            <text x={X(r.points[0])} y={Y(r.points[0]) - 12} className="roicanvas__lbl" fill={col}>
              {r.name}{r.kind === 'overcrowd' ? ` ≤${r.threshold ?? 6}` : ''}
            </text>
          </g>
        )
      })}

      {/* draft being drawn */}
      {draft.length > 0 && (
        <g className="roicanvas__draft">
          {drawKind === 'counting'
            ? <polyline points={ptsAttr(draft)} fill="none" stroke={KIND_COLOR[drawKind]} strokeWidth={3} vectorEffect="non-scaling-stroke" />
            : <polygon points={ptsAttr(draft)} fill={KIND_COLOR[drawKind]} fillOpacity={0.15}
                stroke={KIND_COLOR[drawKind]} strokeWidth={2.5} strokeDasharray="8 5" vectorEffect="non-scaling-stroke" />}
          {draft.map((p, i) => <Handle key={i} x={X(p)} y={Y(p)} color={KIND_COLOR[drawKind]} />)}
        </g>
      )}
    </svg>
  )
}

function Handle({ x, y, color, small, onDown }: {
  x: number; y: number; color: string; small?: boolean; onDown?: (e: React.PointerEvent) => void
}) {
  return (
    <circle cx={x} cy={y} r={small ? 8 : 11} fill="#0a0e14" stroke={color} strokeWidth={3}
      vectorEffect="non-scaling-stroke" className="roicanvas__handle"
      onPointerDown={onDown} style={{ cursor: onDown ? 'grab' : 'default' }} />
  )
}
