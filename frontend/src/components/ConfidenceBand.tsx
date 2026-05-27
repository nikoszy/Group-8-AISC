/**
 * ConfidenceBand — horizontal gradient bar with an elastic sliding needle.
 *
 * Gradient (left → right): REAL → UNCERTAIN → FAKE
 * Needle slides from left=0 to final position on mount with elastic overshoot.
 */

import { useEffect, useState } from 'react'

interface ConfidenceBandProps {
  probFake: number  // 0.0–1.0
}

const GRADIENT =
  'linear-gradient(to right, #27AE60, #A9D86E 25%, #F1C40F 50%, #E67E22 75%, #C0392B)'

/** Return the interpolated hex color at a position in the 5-stop gradient. */
function getNeedleColor(p: number): string {
  if (p < 0.2) return '#27AE60'
  if (p < 0.4) return '#A9D86E'
  if (p < 0.6) return '#F1C40F'
  if (p < 0.8) return '#E67E22'
  return '#C0392B'
}

const TICK_LABELS = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

export function ConfidenceBand({ probFake }: ConfidenceBandProps) {
  const [needleLeft, setNeedleLeft] = useState(0)

  // Trigger the CSS elastic transition after mount
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      setTimeout(() => setNeedleLeft(probFake * 100), 50)
    })
    return () => cancelAnimationFrame(id)
  }, [probFake])

  const needleColor = getNeedleColor(probFake)

  return (
    <div className="w-full">
      {/* Gradient bar + needle */}
      <div className="relative h-5 rounded-full overflow-visible" style={{ background: GRADIENT }}>
        {/* Needle */}
        <div
          className="confidence-needle absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
          style={{ left: `${needleLeft}%` }}
        >
          <div
            style={{
              width: '2px',
              height: '28px',
              backgroundColor: '#fff',
              borderRadius: '1px',
              boxShadow: `0 0 8px 2px ${needleColor}`,
            }}
          />
        </div>
      </div>

      {/* Tick marks */}
      <div className="relative mt-1">
        <div className="flex justify-between">
          {TICK_LABELS.map((label) => (
            <span
              key={label}
              className="font-jbmono"
              style={{ fontSize: '10px', color: 'var(--muted)' }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Zone labels */}
      <div className="flex justify-between mt-1" style={{ paddingLeft: '2px', paddingRight: '2px' }}>
        <span className="font-dmmono uppercase" style={{ fontSize: '9px', color: '#27AE60', letterSpacing: '0.1em' }}>
          Real
        </span>
        <span className="font-dmmono uppercase" style={{ fontSize: '9px', color: '#F1C40F', letterSpacing: '0.1em' }}>
          Uncertain
        </span>
        <span className="font-dmmono uppercase" style={{ fontSize: '9px', color: '#C0392B', letterSpacing: '0.1em' }}>
          Fake
        </span>
      </div>
    </div>
  )
}
