/**
 * PerFrameScores — data table with heatmapped P(fake) column.
 *
 * Columns: Frame · Time (s) · P(fake) [heatmapped] · Artifact · FFT · Laplacian · Face
 */

import type { FrameResult } from '../types'

interface PerFrameScoresProps {
  frames: FrameResult[]
  verdictHi?: number
  verdictLo?: number
}

/** Map 0→1 to a background-color string for the P(fake) heatmap */
function heatmapBg(value: number, verdictHi: number, verdictLo: number): string {
  const mid = (verdictHi + verdictLo) / 2
  const lowMid = verdictLo * 0.5
  if (value < lowMid) return 'rgba(39, 174, 96, 0.20)'
  if (value < verdictLo) return 'rgba(169, 216, 110, 0.20)'
  if (value < mid) return 'rgba(241, 196, 15, 0.20)'
  if (value < verdictHi) return 'rgba(230, 126, 34, 0.20)'
  return 'rgba(192, 57, 43, 0.25)'
}

function heatmapText(value: number, verdictHi: number, verdictLo: number): string {
  const mid = (verdictHi + verdictLo) / 2
  const lowMid = verdictLo * 0.5
  if (value < lowMid) return '#2ECC71'
  if (value < verdictLo) return '#A9D86E'
  if (value < mid) return '#F1C40F'
  if (value < verdictHi) return '#E67E22'
  return '#FF4136'
}

const COL_STYLE: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'right',
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: '12px',
  color: 'var(--text)',
  borderBottom: '1px solid var(--border-dim)',
}

const HEAD_STYLE: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'right',
  fontFamily: '"DM Mono", monospace',
  fontSize: '10px',
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
  color: 'var(--muted)',
  borderBottom: '1px solid var(--border-amber)',
  whiteSpace: 'nowrap',
}

export function PerFrameScores({ frames, verdictHi = 0.6, verdictLo = 0.4 }: PerFrameScoresProps) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ backgroundColor: 'var(--surface)' }}>
            <th style={{ ...HEAD_STYLE, textAlign: 'left' }}>Frame</th>
            <th style={HEAD_STYLE}>Time (s)</th>
            <th style={HEAD_STYLE}>P(fake)</th>
            <th style={HEAD_STYLE}>Artifact</th>
            <th style={HEAD_STYLE}>FFT</th>
            <th style={HEAD_STYLE}>Laplacian</th>
            <th style={HEAD_STYLE}>Face</th>
          </tr>
        </thead>
        <tbody>
          {frames.map((f) => (
            <tr
              key={f.frame_index}
              style={{
                backgroundColor: 'transparent',
                transition: 'background-color 0.1s ease',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLTableRowElement).style.backgroundColor = 'rgba(212,175,55,0.04)'
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLTableRowElement).style.backgroundColor = 'transparent'
              }}
            >
              {/* Frame index */}
              <td style={{ ...COL_STYLE, textAlign: 'left', color: 'var(--muted)', fontSize: '11px' }}>
                f{f.frame_index}
              </td>

              {/* Timestamp */}
              <td style={COL_STYLE}>{f.timestamp_sec.toFixed(2)}</td>

              {/* P(fake) — heatmapped */}
              <td
                className="heatmap-cell"
                style={{
                  ...COL_STYLE,
                  backgroundColor: f.face_detected ? heatmapBg(f.prob_fake, verdictHi, verdictLo) : 'transparent',
                  color: f.face_detected ? heatmapText(f.prob_fake, verdictHi, verdictLo) : 'var(--muted)',
                  fontWeight: 500,
                }}
              >
                {f.face_detected ? f.prob_fake.toFixed(4) : '—'}
              </td>

              {/* Artifact */}
              <td style={{ ...COL_STYLE, color: f.face_detected ? 'var(--text)' : 'var(--muted)' }}>
                {f.face_detected ? f.artifact_score.toFixed(4) : '—'}
              </td>

              {/* FFT */}
              <td style={{ ...COL_STYLE, color: f.face_detected ? 'var(--text)' : 'var(--muted)' }}>
                {f.face_detected ? f.fft_score.toFixed(4) : '—'}
              </td>

              {/* Laplacian */}
              <td style={{ ...COL_STYLE, color: f.face_detected ? 'var(--text)' : 'var(--muted)' }}>
                {f.face_detected ? f.laplacian_score.toFixed(4) : '—'}
              </td>

              {/* Face detected */}
              <td style={COL_STYLE}>
                <span
                  style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: f.face_detected ? 'var(--real)' : 'var(--muted)',
                    boxShadow: f.face_detected ? '0 0 4px var(--real)' : 'none',
                  }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
