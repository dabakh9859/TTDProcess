import { Clock, Hash, TrendingDown, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'

import { useT } from '../../i18n'
import type { ColumnStats } from '../../lib/types'

interface Issue {
  icon: ReactNode
  text: string
  color: string
}

/** One column's quality summary: score, bar, dtype and any detected issues. */
export default function ColumnCard({ col }: { col: ColumnStats }) {
  const { t } = useT()

  const score = col.quality_score
  const scoreColor =
    score >= 90 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--error)'
  const scoreBg =
    score >= 90
      ? 'var(--success-tint-10)'
      : score >= 60
        ? 'var(--warning-tint-10)'
        : 'var(--error-tint-10)'

  const issues: Issue[] = []

  if (col.null_count > 0) {
    issues.push({
      icon: <TrendingDown size={11} />,
      text: t('import.column.nanCount', {
        count: col.null_count.toLocaleString(),
        pct: col.null_percentage.toFixed(1),
      }),
      // Thresholds come from the original: >20% is critical, >5% worth a warning.
      color:
        col.null_percentage > 20
          ? 'var(--error)'
          : col.null_percentage > 5
            ? 'var(--warning)'
            : 'var(--text-4)',
    })
  }

  if (col.outlier_count > 0) {
    issues.push({
      icon: <TriangleAlert size={11} />,
      text: t('import.column.outlierCount', { count: col.outlier_count.toLocaleString() }),
      color: col.outlier_count > col.count * 0.05 ? 'var(--error)' : 'var(--warning)',
    })
  }

  if (col.is_timestamp && col.timestamp_gaps > 0) {
    issues.push({
      icon: <Clock size={11} />,
      text: t('import.column.disorderCount', { count: col.timestamp_gaps }),
      color: 'var(--error)',
    })
  }

  return (
    <div
      style={{
        padding: '12px 14px',
        borderRadius: 8,
        background: 'var(--bg-3)',
        border: `1px solid ${score < 60 ? 'var(--error-tint-20)' : 'var(--border-1)'}`,
        transition: 'border-color 0.15s',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 8,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 22,
              height: 22,
              borderRadius: 5,
              background: col.is_timestamp
                ? 'var(--info-tint-12)'
                : 'var(--accent-tint-10)',
              flexShrink: 0,
            }}
          >
            {col.is_timestamp ? (
              <Clock size={12} color="var(--info)" />
            ) : (
              <Hash size={12} color="var(--accent)" />
            )}
          </span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: 'var(--text-1)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {col.name}
          </span>
        </div>

        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            fontFamily: 'monospace',
            padding: '2px 8px',
            borderRadius: 5,
            background: scoreBg,
            color: scoreColor,
            flexShrink: 0,
          }}
        >
          {score.toFixed(0)}%
        </span>
      </div>

      <div
        style={{
          width: '100%',
          height: 4,
          borderRadius: 2,
          background: 'var(--border-1)',
          marginBottom: 8,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${Math.min(100, score)}%`,
            height: '100%',
            borderRadius: 2,
            background: scoreColor,
            transition: 'width 0.6s ease',
          }}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        <span
          style={{
            fontSize: 10,
            color: 'var(--text-5)',
            padding: '2px 6px',
            background: 'var(--bg-4)',
            borderRadius: 4,
            border: '1px solid var(--border-2)',
            fontFamily: 'monospace',
            textTransform: 'lowercase',
          }}
        >
          {col.dtype.replace(/"/g, '')}
        </span>

        {issues.length === 0 && (
          <span style={{ fontSize: 11, color: 'var(--border-3)' }}>
            {t('import.column.noIssue')}
          </span>
        )}

        {issues.map((issue, i) => (
          <span
            key={i}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
              color: issue.color,
            }}
          >
            {issue.icon}
            {issue.text}
          </span>
        ))}
      </div>
    </div>
  )
}
