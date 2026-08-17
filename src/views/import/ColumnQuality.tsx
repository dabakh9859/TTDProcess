import { ShieldAlert, ShieldCheck, ShieldX, Sparkles, TriangleAlert } from 'lucide-react'

import ColumnCard from './ColumnCard'
import Skeleton, { card } from '../../components/Skeleton'
import { useT } from '../../i18n'
import { useAppStore } from '../../store/useAppStore'
import type { ColumnStats } from '../../lib/types'

interface ColumnQualityProps {
  stats: ColumnStats[] | null
  loading: boolean
}

/** Placeholder grid shown while `get_column_stats` is still running. */
function QualitySkeleton() {
  return (
    <div style={{ ...card, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <Skeleton width={18} height={18} round />
        <Skeleton width={180} height={16} />
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 10,
        }}
      >
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            style={{
              padding: 14,
              borderRadius: 8,
              background: 'var(--bg-3)',
              border: '1px solid var(--border-1)',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Skeleton width={100} height={14} delay={i * 50} />
              <Skeleton width={40} height={14} delay={i * 50 + 20} />
            </div>
            <Skeleton width={250} height={6} delay={i * 50 + 40} />
            <div style={{ display: 'flex', gap: 8 }}>
              <Skeleton width={70} height={11} delay={i * 50 + 60} />
              <Skeleton width={70} height={11} delay={i * 50 + 80} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/**
 * Per-column quality report shown after a file is loaded.
 *
 * Columns are bucketed on their score: >=90 fine, 60–89 worth a look, <60 a
 * problem. When the average drops below 90 — or any column is under 60 — a
 * banner offers a shortcut to the cleaning screen.
 */
export default function ColumnQuality({ stats, loading }: ColumnQualityProps) {
  const { t } = useT()
  const setCurrentTab = useAppStore((s) => s.setCurrentTab)

  if (loading) return <QualitySkeleton />
  if (!stats || stats.length === 0) return null

  const good = stats.filter((c) => c.quality_score >= 90).length
  const warn = stats.filter((c) => c.quality_score >= 60 && c.quality_score < 90).length
  const bad = stats.filter((c) => c.quality_score < 60).length
  const average = stats.reduce((sum, c) => sum + c.quality_score, 0) / stats.length

  const critical = average < 60

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {(average < 90 || bad > 0) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '12px 16px',
            background: critical ? 'var(--error-tint-10)' : 'var(--warning-tint-10)',
            border: `1px solid ${critical ? 'var(--error)' : 'var(--warning)'}`,
            borderRadius: 8,
          }}
        >
          <TriangleAlert
            size={18}
            style={{
              color: critical ? 'var(--error)' : 'var(--warning)',
              flexShrink: 0,
            }}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <p
              style={{
                fontSize: 12,
                fontWeight: 600,
                margin: 0,
                color: critical ? 'var(--error)' : 'var(--warning)',
              }}
            >
              {critical
                ? t('import.quality.lowDetected')
                : t('import.quality.someIssues')}
            </p>
            <p
              style={{
                fontSize: 11,
                color: 'var(--text-3)',
                margin: '2px 0 0',
                lineHeight: 1.4,
              }}
            >
              {bad > 0 && <>{t('import.quality.colsLowScore', { count: bad })} </>}
              {warn > 0 && <>{t('import.quality.withNanOutliers', { count: warn })} </>}
              {t('import.quality.goToCleaning.before')}
              <strong style={{ color: 'var(--text-1)' }}>
                {t('import.quality.dataCleaningName')}
              </strong>
              {t('import.quality.goToCleaning.after')}
            </p>
          </div>
          <button
            onClick={() => setCurrentTab('detection')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '7px 12px',
              background: 'var(--accent)',
              color: '#fff',
              border: '1px solid var(--accent)',
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 500,
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <Sparkles size={12} /> {t('import.quality.cleanNow')}
          </button>
        </div>
      )}

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
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background:
                  average >= 90
                    ? 'var(--success-tint-12)'
                    : average >= 60
                      ? 'var(--warning-tint-12)'
                      : 'var(--error-tint-12)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {average >= 90 ? (
                <ShieldCheck size={16} color="var(--success)" />
              ) : average >= 60 ? (
                <ShieldAlert size={16} color="var(--warning)" />
              ) : (
                <ShieldX size={16} color="var(--error)" />
              )}
            </div>
            <div>
              <h2
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: 'var(--text-1)',
                  margin: 0,
                }}
              >
                {t('import.quality.title')}
              </h2>
              <p style={{ fontSize: 12, color: 'var(--text-4)', margin: 0 }}>
                {t('import.quality.averageScore', { score: average.toFixed(1) })}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            {good > 0 && (
              <Pill
                tone="success"
                text={t('import.quality.pillOk', { count: good })}
              />
            )}
            {warn > 0 && (
              <Pill
                tone="warning"
                text={t('import.quality.pillWarning', { count: warn })}
              />
            )}
            {bad > 0 && (
              <Pill tone="error" text={t('import.quality.pillIssues', { count: bad })} />
            )}
          </div>
        </div>

        <div
          style={{
            padding: 16,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 10,
          }}
        >
          {stats.map((col) => (
            <ColumnCard key={col.name} col={col} />
          ))}
        </div>
      </div>
    </div>
  )
}

function Pill({ tone, text }: { tone: 'success' | 'warning' | 'error'; text: string }) {
  return (
    <span
      style={{
        fontSize: 11,
        fontWeight: 500,
        padding: '4px 10px',
        borderRadius: 6,
        background: `var(--${tone}-tint-10)`,
        color: `var(--${tone})`,
        border: `1px solid var(--${tone}-tint-20)`,
      }}
    >
      {text}
    </span>
  )
}
