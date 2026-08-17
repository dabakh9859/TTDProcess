import Skeleton, { card } from '../../components/Skeleton'

const ROWS = 8
/** Column widths that make the placeholder read like real tabular data. */
const COL_WIDTHS = [44, 120, 90, 110, 80, 100, 95]

/** Stand-in for the preview table while `preview_file` is running. */
export default function PreviewSkeleton() {
  return (
    <div style={{ ...card, overflow: 'hidden' }}>
      <div
        style={{
          padding: '14px 20px',
          borderBottom: '1px solid var(--border-2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Skeleton width={18} height={18} round />
          <Skeleton width={140} height={14} />
        </div>
        <Skeleton width={200} height={12} />
      </div>

      <div
        style={{
          display: 'flex',
          padding: '10px 0',
          background: 'var(--bg-3)',
          borderBottom: '1px solid var(--border-1)',
        }}
      >
        <div style={{ width: 56, padding: '0 12px' }}>
          <Skeleton width={20} height={10} />
        </div>
        {COL_WIDTHS.map((w, i) => (
          <div key={i} style={{ flex: 1, padding: '0 12px' }}>
            <Skeleton width={w * 0.6} height={10} />
          </div>
        ))}
      </div>

      {Array.from({ length: ROWS }).map((_, row) => (
        <div
          key={row}
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '9px 0',
            borderBottom: '1px solid var(--row-divider)',
          }}
        >
          <div style={{ width: 56, padding: '0 12px' }}>
            <Skeleton width={16} height={12} delay={row * 60} />
          </div>
          {COL_WIDTHS.map((w, col) => (
            <div key={col} style={{ flex: 1, padding: '0 12px' }}>
              {/* Jitter the widths a little so rows don't look stamped out. */}
              <Skeleton
                width={w + ((row % 3) * 10 - 10)}
                height={12}
                delay={row * 60 + col * 30}
              />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
