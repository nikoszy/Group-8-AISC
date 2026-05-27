/**
 * Verdict — hero section.
 *
 * Enormous P(fake) number that counts up from 0.
 * Verdict label (FAKE / REAL / UNCERTAIN) in Rajdhani.
 * ConfidenceBand below.
 * Model metadata footer.
 */

import { useEffect, useState, useRef } from 'react'
import { ConfidenceBand } from './ConfidenceBand'

interface VerdictProps {
  verdict: 'FAKE' | 'REAL' | 'UNCERTAIN'
  probFakeMean: number
  confidence: number
  modelUsed: string
  // Registry provenance (v2) — optional for backward compat with Streamlit
  modelId?:   string
  modelType?: string
  modelF1?:   number | null
}

/** easeOutCubic count-up hook */
function useCountUp(target: number, durationMs: number = 1200): number {
  const [value, setValue] = useState(0)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(eased * target)
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick)
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [target, durationMs])

  return value
}

const VERDICT_COLORS: Record<string, string> = {
  FAKE:      'var(--fake)',
  REAL:      'var(--real)',
  UNCERTAIN: 'var(--uncertain)',
}

const VERDICT_GLOWS: Record<string, string> = {
  FAKE:      '0 0 60px rgba(255,65,54,0.15)',
  REAL:      '0 0 60px rgba(46,204,113,0.15)',
  UNCERTAIN: '0 0 60px rgba(243,156,18,0.15)',
}

export function Verdict({ verdict, probFakeMean, confidence, modelUsed, modelId, modelType, modelF1 }: VerdictProps) {
  // Resolve display label: prefer registry model_type, fall back to model_used
  const scoringLabel = modelType && modelType !== 'equal_weights'
    ? modelType
    : (modelUsed === 'ensemble_learned' ? 'LogReg (trained)' : 'Equal weights')
  const displayValue = useCountUp(probFakeMean, 1200)
  const verdictColor = VERDICT_COLORS[verdict] ?? 'var(--text)'
  const verdictGlow = VERDICT_GLOWS[verdict] ?? 'none'

  return (
    <div
      className="rounded-xl p-8 md:p-12"
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-amber)',
        boxShadow: verdictGlow,
      }}
    >
      {/* Analysis verdict label */}
      <p
        className="font-dmmono text-xs uppercase tracking-widest mb-6"
        style={{ color: 'var(--muted)' }}
      >
        Analysis Verdict
      </p>

      {/* Verdict word */}
      <div className="mb-2">
        <span
          className="font-rajdhani font-bold uppercase"
          style={{
            fontSize: 'clamp(2rem, 5vw, 3.5rem)',
            letterSpacing: '0.3em',
            color: verdictColor,
          }}
        >
          {verdict}
        </span>
      </div>

      {/* P(fake) number — the hero element */}
      <div className="scanline-shimmer my-4">
        <span
          className="font-jbmono font-light leading-none block"
          style={{
            fontSize: 'clamp(5rem, 15vw, 10rem)',
            color: verdictColor,
            textShadow: `0 0 30px ${verdictColor}40`,
          }}
        >
          {displayValue.toFixed(3)}
        </span>
      </div>

      {/* Sub-label */}
      <p
        className="font-dmmono text-xs uppercase tracking-widest mb-8"
        style={{ color: 'var(--muted)' }}
      >
        P(fake) — ensemble probability
      </p>

      {/* Confidence band */}
      <ConfidenceBand probFake={probFakeMean} />

      {/* Metadata footer */}
      <div className="flex flex-wrap gap-6 mt-8 pt-6" style={{ borderTop: '1px solid var(--border-dim)' }}>
        <div>
          <p className="font-dmmono text-xs uppercase" style={{ color: 'var(--muted)' }}>
            Confidence
          </p>
          <p className="font-jbmono font-medium text-sm mt-0.5" style={{ color: 'var(--text)' }}>
            {(confidence * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="font-dmmono text-xs uppercase" style={{ color: 'var(--muted)' }}>
            Scoring Model
          </p>
          <p className="font-jbmono font-medium text-sm mt-0.5" style={{ color: 'var(--text)' }}>
            {scoringLabel}
            {modelF1 != null && (
              <span style={{ color: 'var(--muted)', marginLeft: '0.4em' }}>
                (F1 = {modelF1.toFixed(2)})
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Collapsible model details */}
      {(modelId || modelType) && (
        <details
          className="mt-4"
          style={{ borderTop: '1px solid var(--border-dim)', paddingTop: '0.75rem' }}
        >
          <summary
            className="font-dmmono text-xs uppercase tracking-widest cursor-pointer select-none"
            style={{ color: 'var(--muted)', listStyle: 'none' }}
          >
            ▸ Model details
          </summary>
          <div className="mt-3 space-y-1 pl-2">
            {modelId && (
              <p className="font-dmmono text-xs" style={{ color: 'var(--muted)' }}>
                <span style={{ opacity: 0.6 }}>ID:</span>{' '}
                <span style={{ color: 'var(--text)' }}>{modelId}</span>
              </p>
            )}
            {modelType && (
              <p className="font-dmmono text-xs" style={{ color: 'var(--muted)' }}>
                <span style={{ opacity: 0.6 }}>Type:</span>{' '}
                <span style={{ color: 'var(--text)' }}>{modelType}</span>
              </p>
            )}
            {modelF1 != null && (
              <p className="font-dmmono text-xs" style={{ color: 'var(--muted)' }}>
                <span style={{ opacity: 0.6 }}>Val F1:</span>{' '}
                <span style={{ color: 'var(--text)' }}>{modelF1.toFixed(4)}</span>
                <span style={{ opacity: 0.5 }}> (held-out 20% split, seed=42)</span>
              </p>
            )}
          </div>
        </details>
      )}
    </div>
  )
}
