/**
 * Charts — per-frame P(fake) bar chart with laplacian quality overlay.
 *
 * Uses Recharts ComposedChart:
 * - Bar: prob_fake per frame, fill color varies by band
 * - Line: laplacian_score (quality proxy) overlaid in gold
 */

import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import type { FrameResult } from '../types'

interface ChartsProps {
  frames: FrameResult[]
  verdictHi?: number
  verdictLo?: number
}

function bandColor(value: number): string {
  if (value < 0.2) return '#27AE60'
  if (value < 0.4) return '#A9D86E'
  if (value < 0.6) return '#F1C40F'
  if (value < 0.8) return '#E67E22'
  return '#C0392B'
}

const TOOLTIP_STYLE: React.CSSProperties = {
  backgroundColor: '#0D1219',
  border: '1px solid rgba(212,175,55,0.25)',
  borderRadius: '6px',
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: '11px',
  color: '#C8C0A8',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div style={TOOLTIP_STYLE} className="px-3 py-2">
      <p style={{ color: 'var(--muted)', fontSize: '10px', marginBottom: '4px' }}>
        Frame {label}
      </p>
      {payload.map((entry: { name: string; value: number; color: string }) => (
        <p key={entry.name} style={{ color: entry.color, margin: '2px 0' }}>
          {entry.name}: {entry.value?.toFixed(4) ?? '—'}
        </p>
      ))}
    </div>
  )
}

export function Charts({ frames, verdictHi = 0.6, verdictLo = 0.4 }: ChartsProps) {
  const data = frames.map(f => ({
    frame: f.frame_index,
    prob_fake: f.face_detected ? f.prob_fake : null,
    laplacian: f.face_detected ? f.laplacian_score : null,
  }))

  return (
    <div>
      <p className="font-dmmono text-xs mb-6" style={{ color: 'var(--muted)' }}>
        Bar = P(fake) per frame · Gold line = Laplacian quality (higher = sharper frame)
      </p>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.05)"
            horizontal={true}
            vertical={false}
          />

          {/* Decision boundary lines */}
          <ReferenceLine y={verdictHi} stroke="rgba(255,65,54,0.4)"  strokeDasharray="4 4" />
          <ReferenceLine y={verdictLo} stroke="rgba(46,204,113,0.4)" strokeDasharray="4 4" />

          <XAxis
            dataKey="frame"
            tick={{ fontFamily: '"JetBrains Mono"', fontSize: 10, fill: '#4A5568' }}
            axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            tickLine={false}
            label={{ value: 'Frame', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#4A5568', fontFamily: '"DM Mono"' }}
          />

          <YAxis
            domain={[0, 1]}
            tick={{ fontFamily: '"JetBrains Mono"', fontSize: 10, fill: '#4A5568' }}
            axisLine={false}
            tickLine={false}
            tickCount={6}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(212,175,55,0.05)' }} />

          <Legend
            wrapperStyle={{
              fontFamily: '"DM Mono", monospace',
              fontSize: '11px',
              color: '#4A5568',
              paddingTop: '8px',
            }}
          />

          <Bar dataKey="prob_fake" name="P(fake)" maxBarSize={32} radius={[2, 2, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.prob_fake !== null ? bandColor(entry.prob_fake) : 'transparent'}
                fillOpacity={0.85}
              />
            ))}
          </Bar>

          <Line
            dataKey="laplacian"
            name="Laplacian quality"
            stroke="#D4AF37"
            strokeWidth={1.5}
            dot={{ fill: '#D4AF37', r: 2, strokeWidth: 0 }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend annotation */}
      <div className="flex gap-6 mt-4 flex-wrap" style={{ fontSize: '10px', fontFamily: '"DM Mono", monospace', color: 'var(--muted)' }}>
        <span style={{ color: 'rgba(255,65,54,0.6)' }}>— — FAKE threshold ({verdictHi.toFixed(2)})</span>
        <span style={{ color: 'rgba(46,204,113,0.6)' }}>— — REAL threshold ({verdictLo.toFixed(2)})</span>
      </div>
    </div>
  )
}
