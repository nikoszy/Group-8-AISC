/**
 * SignalBreakdown — three labeled signal rows with score bars.
 *
 * Signals:
 *   1. Quality-weighted P(fake)  — mean weighted by laplacian (sharpness)
 *   2. Temporal score            — std-dev of per-frame probabilities
 *   3. rPPG fake score           — always "Not available" (stub)
 */

interface SignalBreakdownProps {
  qualityWeightedProb: number
  temporalScore: number
  rppgFakeScore: number
  rppgAvailable: boolean
}

interface SignalRowProps {
  label: string
  description: string
  value: number
  available: boolean
  delay?: number
}

function ScoreBar({ value, available }: { value: number; available: boolean }) {
  const barColor = available
    ? value < 0.4 ? '#2ECC71' : value < 0.6 ? '#F1C40F' : '#FF4136'
    : 'rgba(74,85,104,0.4)'

  return (
    <div
      className="flex-1 rounded-full overflow-hidden"
      style={{
        height: '4px',
        backgroundColor: 'rgba(255,255,255,0.06)',
      }}
    >
      <div
        className="score-bar-fill h-full rounded-full"
        style={{
          width: available ? `${value * 100}%` : '0%',
          backgroundColor: barColor,
        }}
      />
    </div>
  )
}

function SignalRow({ label, description, value, available, delay = 0 }: SignalRowProps) {
  return (
    <div
      className="py-4"
      style={{
        borderBottom: '1px solid var(--border-dim)',
        animation: `fadeUp 0.5s ease-out ${delay}s both`,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="font-rajdhani font-semibold text-sm uppercase tracking-wide"
            style={{ color: 'var(--text)' }}>
            {label}
          </p>
          <p className="font-dmmono text-xs mt-0.5" style={{ color: 'var(--muted)' }}>
            {description}
          </p>
        </div>
        <div className="text-right ml-6 flex-shrink-0">
          {available ? (
            <span className="font-jbmono font-medium text-lg" style={{ color: 'var(--text)' }}>
              {value.toFixed(4)}
            </span>
          ) : (
            <span
              className="font-dmmono text-xs px-2 py-1 rounded"
              style={{
                backgroundColor: 'rgba(74,85,104,0.15)',
                color: 'var(--muted)',
                border: '1px solid rgba(74,85,104,0.3)',
              }}
            >
              Not available
            </span>
          )}
        </div>
      </div>
      <ScoreBar value={value} available={available} />
    </div>
  )
}

export function SignalBreakdown({
  qualityWeightedProb,
  temporalScore,
  rppgFakeScore,
  rppgAvailable,
}: SignalBreakdownProps) {
  return (
    <div>
      <SignalRow
        label="Quality-Weighted P(fake)"
        description="P(fake) averaged by frame sharpness — blurry frames contribute less"
        value={qualityWeightedProb}
        available={true}
        delay={0.05}
      />
      <SignalRow
        label="Temporal Inconsistency"
        description="Std-dev of per-frame P(fake) — high values indicate temporal fluctuation"
        value={temporalScore}
        available={true}
        delay={0.10}
      />
      <SignalRow
        label="rPPG Fake Score"
        description="Remote photoplethysmography blood-flow analysis (Module 4 — not integrated)"
        value={rppgFakeScore}
        available={rppgAvailable}
        delay={0.15}
      />
    </div>
  )
}
