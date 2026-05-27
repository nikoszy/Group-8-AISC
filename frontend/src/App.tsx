import { useState } from 'react'
import { postAnalyze } from './api'
import type { AnalysisResponse } from './types'
import { Upload } from './components/Upload'
import { ResultsPage } from './components/ResultsPage'
import './index.css'

type AppState = 'upload' | 'loading' | 'results' | 'error'

export default function App() {
  const [state, setState] = useState<AppState>('upload')
  const [results, setResults] = useState<AnalysisResponse | null>(null)
  const [error, setError] = useState<string>('')

  const handleAnalyze = async (file: File, nFrames: number) => {
    setState('loading')
    setError('')
    try {
      const data = await postAnalyze(file, nFrames)
      setResults(data)
      setState('results')
    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : 'Analysis failed. Is the backend running?'
      setError(msg)
      setState('error')
    }
  }

  const handleReset = () => {
    setState('upload')
    setResults(null)
    setError('')
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg)' }}>
      {/* ── Upload ──────────────────────────────────────────────────── */}
      {state === 'upload' && (
        <Upload onAnalyze={handleAnalyze} loading={false} />
      )}

      {/* ── Loading ─────────────────────────────────────────────────── */}
      {state === 'loading' && (
        <div className="min-h-screen flex flex-col items-center justify-center gap-8">
          {/* Amber pulsing ring */}
          <div className="relative w-20 h-20">
            <div
              className="absolute inset-0 rounded-full animate-ping"
              style={{ background: 'rgba(212,175,55,0.15)', animationDuration: '1.2s' }}
            />
            <div
              className="absolute inset-2 rounded-full border-2"
              style={{ borderColor: 'var(--gold)', borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite' }}
            />
          </div>
          <div className="text-center">
            <p
              className="font-rajdhani text-2xl font-semibold tracking-widest uppercase"
              style={{ color: 'var(--gold)' }}
            >
              Analyzing
            </p>
            <p className="font-dmmono text-xs mt-1" style={{ color: 'var(--muted)' }}>
              Extracting frames · Detecting faces · Scoring signals
            </p>
          </div>
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────────── */}
      {state === 'results' && results && (
        <ResultsPage data={results} onReset={handleReset} />
      )}

      {/* ── Error ───────────────────────────────────────────────────── */}
      {state === 'error' && (
        <div className="min-h-screen flex items-center justify-center p-8">
          <div
            className="max-w-md w-full rounded-lg p-8 text-center"
            style={{
              backgroundColor: 'var(--surface)',
              border: '1px solid rgba(255,65,54,0.3)',
            }}
          >
            <div className="text-4xl mb-4">⚠</div>
            <h2
              className="font-rajdhani text-xl font-semibold mb-2"
              style={{ color: 'var(--fake)' }}
            >
              Analysis Failed
            </h2>
            <p className="font-dmmono text-xs mb-6" style={{ color: 'var(--muted)' }}>
              {error}
            </p>
            <button
              onClick={handleReset}
              className="font-rajdhani font-semibold text-sm uppercase tracking-widest px-6 py-2 rounded transition-opacity hover:opacity-80"
              style={{ backgroundColor: 'var(--gold)', color: '#07090E' }}
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Spin keyframe (inline so Tailwind doesn't tree-shake it) */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
