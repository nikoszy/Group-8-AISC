/**
 * ExportButtons — download JSON report and CSV of per-frame scores.
 */

import type { AnalysisResponse } from '../types'

interface ExportButtonsProps {
  data: AnalysisResponse
}

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function makeCSV(data: AnalysisResponse): string {
  const headers = [
    'frame_index', 'timestamp_sec', 'face_detected',
    'prob_fake', 'artifact_score', 'fft_score', 'laplacian_score', 'ear_score',
  ]
  const rows = data.frames.map(f => [
    f.frame_index,
    f.timestamp_sec,
    f.face_detected ? 1 : 0,
    f.prob_fake,
    f.artifact_score,
    f.fft_score,
    f.laplacian_score,
    f.ear_score,
  ])
  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
}

function makeJSONReport(data: AnalysisResponse): string {
  // Omit large base64 crops from the JSON report to keep it readable
  const { frames, ...meta } = data
  const trimmedFrames = frames.map(({ face_crop_b64: _b64, ...f }) => f)
  return JSON.stringify({ ...meta, frames: trimmedFrames }, null, 2)
}

interface BtnProps {
  onClick: () => void
  label: string
  sublabel: string
}

function ExportBtn({ onClick, label, sublabel }: BtnProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 px-5 py-3 rounded-lg transition-all duration-150 group"
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-amber)',
      }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLButtonElement
        el.style.borderColor = 'rgba(212,175,55,0.5)'
        el.style.backgroundColor = 'rgba(212,175,55,0.06)'
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLButtonElement
        el.style.borderColor = 'var(--border-amber)'
        el.style.backgroundColor = 'var(--surface)'
      }}
    >
      {/* Arrow down icon */}
      <svg
        width="16" height="16" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" strokeWidth="2"
        style={{ color: 'var(--gold)', flexShrink: 0 }}
      >
        <path d="M12 5v14M5 12l7 7 7-7" />
      </svg>
      <div className="text-left">
        <p className="font-rajdhani font-semibold text-sm uppercase tracking-wide"
          style={{ color: 'var(--text)', lineHeight: 1.2 }}>
          {label}
        </p>
        <p className="font-dmmono" style={{ fontSize: '10px', color: 'var(--muted)' }}>
          {sublabel}
        </p>
      </div>
    </button>
  )
}

export function ExportButtons({ data }: ExportButtonsProps) {
  const baseName = data.video_name.replace(/\.[^.]+$/, '')

  const handleJSON = () => {
    downloadBlob(makeJSONReport(data), `${baseName}_deepfake_report.json`, 'application/json')
  }

  const handleCSV = () => {
    downloadBlob(makeCSV(data), `${baseName}_per_frame_scores.csv`, 'text/csv')
  }

  return (
    <div className="flex gap-3 flex-wrap">
      <ExportBtn
        onClick={handleJSON}
        label="Download JSON Report"
        sublabel="Aggregate results · no face crops"
      />
      <ExportBtn
        onClick={handleCSV}
        label="Download CSV Scores"
        sublabel="Per-frame feature scores"
      />
    </div>
  )
}
