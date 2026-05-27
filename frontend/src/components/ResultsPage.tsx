/**
 * ResultsPage — layout shell for all result panels.
 *
 * Composition:
 *   header (video name + reset button)
 *   Verdict
 *   StatTiles
 *   Tabs
 *   ExportButtons
 *   Warnings (if any)
 */

import type { AnalysisResponse } from '../types'
import { Verdict } from './Verdict'
import { StatTiles } from './StatTiles'
import { Tabs } from './Tabs'
import { ExportButtons } from './ExportButtons'

interface ResultsPageProps {
  data: AnalysisResponse
  onReset: () => void
}

export function ResultsPage({ data, onReset }: ResultsPageProps) {
  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: 'var(--bg)' }}
    >
      <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">

        {/* ── Page header ─────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <p
              className="font-dmmono text-xs uppercase tracking-widest"
              style={{ color: 'var(--muted)' }}
            >
              Analysis complete
            </p>
            <h1
              className="font-rajdhani font-bold text-2xl uppercase mt-0.5"
              style={{ color: 'var(--text)', letterSpacing: '0.08em' }}
            >
              {data.video_name}
            </h1>
          </div>
          <button
            onClick={onReset}
            className="font-dmmono text-xs uppercase tracking-widest px-4 py-2 rounded transition-all duration-150"
            style={{
              border: '1px solid var(--border-amber)',
              color: 'var(--muted)',
              backgroundColor: 'transparent',
            }}
            onMouseEnter={e => {
              const el = e.currentTarget as HTMLButtonElement
              el.style.color = 'var(--gold)'
              el.style.borderColor = 'rgba(212,175,55,0.5)'
            }}
            onMouseLeave={e => {
              const el = e.currentTarget as HTMLButtonElement
              el.style.color = 'var(--muted)'
              el.style.borderColor = 'var(--border-amber)'
            }}
          >
            ← Analyze another
          </button>
        </div>

        {/* ── Verdict hero ────────────────────────────────────────────── */}
        <Verdict
          verdict={data.verdict}
          probFakeMean={data.prob_fake_mean}
          confidence={data.confidence}
          modelUsed={data.model_used}
          modelId={data.model_id}
          modelType={data.model_type}
          modelF1={data.model_f1}
        />

        {/* ── Stat tiles ──────────────────────────────────────────────── */}
        <StatTiles
          framesAnalyzed={data.frames_analyzed}
          framesSampled={data.frames_sampled}
          fps={data.fps}
          durationSec={data.duration_sec}
          cnnActive={data.cnn_active}
        />

        {/* ── Tabs ────────────────────────────────────────────────────── */}
        <Tabs data={data} />

        {/* ── Export buttons ──────────────────────────────────────────── */}
        <div>
          <p
            className="font-dmmono text-xs uppercase tracking-widest mb-3"
            style={{ color: 'var(--muted)' }}
          >
            Export
          </p>
          <ExportButtons data={data} />
        </div>

        {/* ── Warnings ────────────────────────────────────────────────── */}
        {data.warnings.length > 0 && (
          <div
            className="rounded-lg p-4"
            style={{
              backgroundColor: 'rgba(243,156,18,0.06)',
              border: '1px solid rgba(243,156,18,0.2)',
            }}
          >
            <p
              className="font-rajdhani font-semibold text-sm uppercase tracking-wide mb-2"
              style={{ color: 'var(--uncertain)' }}
            >
              Processing Warnings
            </p>
            {data.warnings.map((w, i) => (
              <p key={i} className="font-dmmono text-xs mt-1" style={{ color: 'var(--muted)' }}>
                · {w}
              </p>
            ))}
          </div>
        )}

        {/* ── Footer ──────────────────────────────────────────────────── */}
        <div
          className="text-center pt-4 pb-2"
          style={{ borderTop: '1px solid var(--border-dim)' }}
        >
          <p className="font-dmmono" style={{ fontSize: '10px', color: 'rgba(74,85,104,0.5)' }}>
            Group 8 · AISC · FaceForensics++ C23 · handcrafted features
          </p>
        </div>
      </div>
    </div>
  )
}
