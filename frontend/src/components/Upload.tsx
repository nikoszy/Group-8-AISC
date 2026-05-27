/**
 * Upload — drag-and-drop video upload panel.
 *
 * Design: "forensic intake form" aesthetic.
 * Large amber-dashed drop zone, file info strip, n_frames slider, analyze button.
 */

import { useRef, useState, useCallback } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface UploadProps {
  onAnalyze: (file: File, nFrames: number) => void
  loading: boolean
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function Upload({ onAnalyze, loading }: UploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [nFrames, setNFrames] = useState(12)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const acceptFile = useCallback((f: File) => {
    setFile(f)
  }, [])

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) acceptFile(dropped)
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(true)
  }
  const onDragLeave = () => setDragging(false)

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) acceptFile(selected)
  }

  const handleSubmit = () => {
    if (file && !loading) onAnalyze(file, nFrames)
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-8"
      style={{ backgroundColor: 'var(--bg)' }}
    >
      {/* Header */}
      <div className="mb-12 text-center" style={{ animation: 'fadeUp 0.6s ease-out both' }}>
        <p
          className="font-dmmono text-xs uppercase tracking-widest mb-3"
          style={{ color: 'var(--muted)' }}
        >
          Group 8 · AISC · Deepfake Detection System
        </p>
        <h1
          className="font-rajdhani font-bold uppercase"
          style={{
            fontSize: 'clamp(2.5rem, 6vw, 5rem)',
            letterSpacing: '0.12em',
            color: 'var(--text)',
            lineHeight: 1,
          }}
        >
          Deepfake Detector
        </h1>
        <p className="font-dmmono text-xs mt-3" style={{ color: 'var(--muted)' }}>
          Upload a video · detect deepfake signals · receive a verdict
        </p>
      </div>

      {/* Drop zone */}
      <div
        className="w-full max-w-xl mb-8 cursor-pointer select-none"
        style={{ animation: 'fadeUp 0.6s ease-out 0.1s both' }}
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <div
          className="rounded-lg flex flex-col items-center justify-center p-16 transition-all duration-200"
          style={{
            backgroundColor: dragging
              ? 'rgba(212, 175, 55, 0.06)'
              : 'var(--surface)',
            border: dragging
              ? '2px dashed rgba(212, 175, 55, 0.7)'
              : '2px dashed rgba(212, 175, 55, 0.25)',
            boxShadow: dragging
              ? '0 0 24px rgba(212, 175, 55, 0.08) inset'
              : 'none',
          }}
        >
          {/* Icon */}
          <div className="mb-4" style={{ opacity: 0.4 }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 16l4-4 4 4 4-8 4 8" />
              <rect x="2" y="3" width="20" height="18" rx="2" />
            </svg>
          </div>

          {file ? (
            /* File selected state */
            <div className="text-center">
              <p
                className="font-jbmono font-medium text-sm mb-1"
                style={{ color: 'var(--gold)' }}
              >
                {file.name}
              </p>
              <p className="font-dmmono text-xs" style={{ color: 'var(--muted)' }}>
                {formatBytes(file.size)}
              </p>
              <p
                className="font-dmmono text-xs mt-3"
                style={{ color: 'rgba(212,175,55,0.5)' }}
              >
                Click to change file
              </p>
            </div>
          ) : (
            /* Empty state */
            <div className="text-center">
              <p className="font-rajdhani font-semibold text-lg uppercase tracking-widest mb-1"
                style={{ color: 'var(--text)' }}>
                Drop video here
              </p>
              <p className="font-dmmono text-xs" style={{ color: 'var(--muted)' }}>
                or click to browse · mp4, avi, mov, webm
              </p>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={onFileChange}
        />
      </div>

      {/* Frame count slider */}
      <div
        className="w-full max-w-xl mb-8"
        style={{ animation: 'fadeUp 0.6s ease-out 0.2s both' }}
      >
        <div
          className="rounded-lg p-5"
          style={{
            backgroundColor: 'var(--surface)',
            border: '1px solid var(--border-amber)',
          }}
        >
          <div className="flex justify-between items-baseline mb-3">
            <label className="font-dmmono text-xs uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
              Frames to sample
            </label>
            <span className="font-jbmono font-medium text-sm" style={{ color: 'var(--gold)' }}>
              {nFrames}
            </span>
          </div>
          <input
            type="range"
            min={4}
            max={24}
            step={1}
            value={nFrames}
            onChange={e => setNFrames(Number(e.target.value))}
            className="w-full h-1 rounded appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, var(--gold) ${((nFrames - 4) / 20) * 100}%, rgba(212,175,55,0.15) ${((nFrames - 4) / 20) * 100}%)`,
              outline: 'none',
            }}
          />
          <div className="flex justify-between font-dmmono text-xs mt-2" style={{ color: 'var(--muted)' }}>
            <span>4 frames — fast</span>
            <span>24 frames — thorough</span>
          </div>
        </div>
      </div>

      {/* Analyze button */}
      <div style={{ animation: 'fadeUp 0.6s ease-out 0.3s both' }}>
        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="font-rajdhani font-bold uppercase tracking-widest text-base px-12 py-3 rounded transition-all duration-200"
          style={{
            backgroundColor: file ? 'var(--gold)' : 'rgba(212,175,55,0.15)',
            color: file ? '#07090E' : 'var(--muted)',
            cursor: file ? 'pointer' : 'not-allowed',
            letterSpacing: '0.2em',
          }}
        >
          {loading ? 'Analyzing…' : 'Analyze Video'}
        </button>
      </div>

      {/* Footer note */}
      <p
        className="font-dmmono text-xs mt-8 text-center"
        style={{ color: 'rgba(74,85,104,0.6)', maxWidth: '400px' }}
      >
        Video is processed locally and not stored. Results are returned as JSON.
      </p>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        input[type=range]::-webkit-slider-thumb {
          appearance: none;
          width: 14px; height: 14px;
          border-radius: 50%;
          background: var(--gold);
          cursor: pointer;
        }
      `}</style>
    </div>
  )
}
