/**
 * Tabs — four-tab panel container.
 *
 * Tabs: Signal Breakdown · Sampled Frames · Per-Frame Scores · Charts
 */

import { useState } from 'react'
import type { AnalysisResponse } from '../types'
import { SignalBreakdown } from './SignalBreakdown'
import { SampledFrames } from './SampledFrames'
import { PerFrameScores } from './PerFrameScores'
import { Charts } from './Charts'

interface TabsProps {
  data: AnalysisResponse
}

const TAB_DEFS = [
  { id: 'signals',  label: 'Signal Breakdown' },
  { id: 'frames',   label: 'Sampled Frames' },
  { id: 'scores',   label: 'Per-Frame Scores' },
  { id: 'charts',   label: 'Charts' },
] as const

type TabId = typeof TAB_DEFS[number]['id']

export function Tabs({ data }: TabsProps) {
  const [active, setActive] = useState<TabId>('signals')

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-amber)',
      }}
    >
      {/* Tab bar */}
      <div
        className="flex overflow-x-auto"
        style={{ borderBottom: '1px solid var(--border-amber)' }}
      >
        {TAB_DEFS.map(tab => {
          const isActive = tab.id === active
          return (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className="flex-shrink-0 px-5 py-4 relative transition-colors duration-150"
              style={{
                fontFamily: '"Rajdhani", sans-serif',
                fontWeight: 600,
                fontSize: '13px',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                color: isActive ? 'var(--gold)' : 'var(--muted)',
                backgroundColor: 'transparent',
                border: 'none',
                cursor: 'pointer',
                outline: 'none',
              }}
            >
              {tab.label}
              {/* Active underline */}
              {isActive && (
                <div
                  className="absolute bottom-0 left-0 right-0"
                  style={{
                    height: '2px',
                    backgroundColor: 'var(--gold)',
                  }}
                />
              )}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="p-6">
        {active === 'signals' && (
          <SignalBreakdown
            qualityWeightedProb={data.quality_weighted_prob_fake}
            temporalScore={data.temporal_score}
            rppgFakeScore={data.rppg_fake_score}
            rppgAvailable={data.rppg_available}
          />
        )}

        {active === 'frames' && (
          <SampledFrames frames={data.frames} />
        )}

        {active === 'scores' && (
          <PerFrameScores frames={data.frames} />
        )}

        {active === 'charts' && (
          <Charts frames={data.frames} />
        )}
      </div>
    </div>
  )
}
