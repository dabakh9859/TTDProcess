import { CircleCheck, CircleX, Info, ScrollText, TriangleAlert } from 'lucide-react'

import { useT } from '../i18n'
import { useAppStore } from '../store/useAppStore'
import type { LogEntry } from '../store/useAppStore'

/** Icon for the most recent log line, colour-coded by level. */
function LevelIcon({ entry }: { entry: LogEntry | undefined }) {
  if (!entry) return <Info size={12} color="var(--text-4)" />

  switch (entry.level) {
    case 'Success':
      return <CircleCheck size={12} color="var(--success)" />
    case 'Warning':
      return <TriangleAlert size={12} color="var(--warning)" />
    case 'Error':
      return <CircleX size={12} color="var(--error)" />
    default:
      return <Info size={12} color="var(--info)" />
  }
}

export default function StatusBar() {
  const logs = useAppStore((s) => s.logs)
  const setCurrentTab = useAppStore((s) => s.setCurrentTab)
  const { t } = useT()

  const latest = logs[0]

  return (
    <footer
      style={{
        height: 32,
        background: 'var(--bg-2)',
        borderTop: '1px solid var(--border-2)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 12px',
        flexShrink: 0,
      }}
    >
      <button
        onClick={() => setCurrentTab('journal')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text-3)',
          fontSize: 11,
        }}
      >
        <ScrollText size={13} color="var(--text-4)" />
        <span>{t('layout.bottombar.activityLog')}</span>
        {logs.length > 0 && (
          <span
            style={{
              padding: '1px 6px',
              fontSize: 10,
              fontWeight: 700,
              background: 'var(--accent)',
              color: '#fff',
              borderRadius: 10,
              minWidth: 18,
              textAlign: 'center',
            }}
          >
            {logs.length}
          </span>
        )}
      </button>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          maxWidth: '60%',
          overflow: 'hidden',
        }}
      >
        {latest && (
          <>
            <LevelIcon entry={latest} />
            <span
              style={{
                fontSize: 11,
                color: 'var(--text-4)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {latest.message}
            </span>
          </>
        )}
      </div>

      <div style={{ fontSize: 10, color: 'var(--text-5)' }}>
        {t('layout.bottombar.developer')}
      </div>
    </footer>
  )
}
