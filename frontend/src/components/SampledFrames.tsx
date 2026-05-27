/**
 * SampledFrames — grid of face crop cards.
 *
 * Each card shows the base64 face crop image (or a "no face" placeholder)
 * with the frame index, timestamp, and P(fake) score badge.
 */

import type { FrameResult } from '../types'

interface SampledFramesProps {
  frames: FrameResult[]
}

function probColor(p: number): string {
  if (p < 0.4) return 'var(--real)'
  if (p < 0.6) return 'var(--uncertain)'
  return 'var(--fake)'
}

function FrameCard({ frame, delay }: { frame: FrameResult; delay: number }) {
  return (
    <div
      className="rounded-lg overflow-hidden flex flex-col"
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-amber)',
        animation: `fadeUp 0.4s ease-out ${delay}s both`,
      }}
    >
      {/* Image or placeholder */}
      <div
        className="relative flex-shrink-0"
        style={{ aspectRatio: '1', backgroundColor: '#050810' }}
      >
        {frame.face_detected && frame.face_crop_b64 ? (
          <img
            src={`data:image/jpeg;base64,${frame.face_crop_b64}`}
            alt={`Face crop frame ${frame.frame_index}`}
            className="w-full h-full object-cover"
            style={{ imageRendering: 'pixelated' }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div style={{ fontSize: '2rem', opacity: 0.2 }}>○</div>
              <p className="font-dmmono text-xs mt-1" style={{ color: 'var(--muted)', fontSize: '9px' }}>
                No face
              </p>
            </div>
          </div>
        )}

        {/* P(fake) badge */}
        {frame.face_detected && (
          <div
            className="absolute top-1.5 right-1.5 font-jbmono font-medium rounded px-1.5 py-0.5"
            style={{
              fontSize: '10px',
              backgroundColor: 'rgba(7,9,14,0.85)',
              color: probColor(frame.prob_fake),
              border: `1px solid ${probColor(frame.prob_fake)}40`,
            }}
          >
            {frame.prob_fake.toFixed(3)}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2">
        <div className="flex justify-between items-center">
          <span className="font-jbmono" style={{ fontSize: '10px', color: 'var(--muted)' }}>
            f{frame.frame_index}
          </span>
          <span className="font-jbmono" style={{ fontSize: '10px', color: 'var(--muted)' }}>
            {frame.timestamp_sec.toFixed(2)}s
          </span>
        </div>
      </div>
    </div>
  )
}

export function SampledFrames({ frames }: SampledFramesProps) {
  const detectedFrames = frames.filter(f => f.face_detected)
  const allFrames = frames

  return (
    <div>
      <p className="font-dmmono text-xs mb-4" style={{ color: 'var(--muted)' }}>
        {detectedFrames.length} faces detected across {allFrames.length} sampled frames
      </p>

      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
        }}
      >
        {allFrames.map((frame, i) => (
          <FrameCard
            key={frame.frame_index}
            frame={frame}
            delay={i * 0.04}
          />
        ))}
      </div>
    </div>
  )
}
