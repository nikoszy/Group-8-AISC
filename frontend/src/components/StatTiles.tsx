/**
 * StatTiles — row of 5 compact amber-bordered metric tiles.
 *
 * Tiles: Total Frames · FPS · Duration · Faces Found · CNN Active
 */

interface StatTilesProps {
  framesAnalyzed: number
  framesSampled: number
  fps: number
  durationSec: number
  cnnActive: boolean
}

interface TileProps {
  label: string
  value: string
  accent?: boolean
  delay?: number
}

function Tile({ label, value, accent = false, delay = 0 }: TileProps) {
  return (
    <div
      className="flex-1 min-w-0 rounded-lg p-4"
      style={{
        backgroundColor: 'var(--surface)',
        border: `2px solid ${accent ? 'rgba(212,175,55,0.4)' : 'var(--border-amber)'}`,
        animation: `fadeUp 0.5s ease-out ${delay}s both`,
      }}
    >
      <p
        className="font-dmmono uppercase truncate"
        style={{ fontSize: '9px', letterSpacing: '0.1em', color: 'var(--muted)' }}
      >
        {label}
      </p>
      <p
        className="font-jbmono font-medium mt-1 truncate"
        style={{ fontSize: '1.05rem', color: accent ? 'var(--gold)' : 'var(--text)' }}
      >
        {value}
      </p>
    </div>
  )
}

export function StatTiles({
  framesAnalyzed,
  framesSampled,
  fps,
  durationSec,
  cnnActive,
}: StatTilesProps) {
  const durationStr =
    durationSec >= 60
      ? `${Math.floor(durationSec / 60)}m ${(durationSec % 60).toFixed(0)}s`
      : `${durationSec.toFixed(1)}s`

  return (
    <div className="flex gap-3 flex-wrap md:flex-nowrap">
      <Tile label="Faces Found"    value={`${framesAnalyzed}/${framesSampled}`} accent delay={0.05} />
      <Tile label="FPS"            value={fps.toFixed(1)}                        delay={0.10} />
      <Tile label="Duration"       value={durationStr}                           delay={0.15} />
      <Tile label="Frames Sampled" value={String(framesSampled)}                 delay={0.20} />
      <Tile
        label="CNN Active"
        value={cnnActive ? 'YES' : 'NO'}
        accent={false}
        delay={0.25}
      />
    </div>
  )
}
