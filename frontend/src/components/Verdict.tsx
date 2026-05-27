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

export function Verdict({ verdict, probFakeMean, confidence, modelUsed }: VerdictProps) {
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
            {modelUsed === 'ensemble_learned' ? 'LogReg (trained)' : 'Equal weights'}
          </p>
        </div>
      </div>
    </div>
  )
}
