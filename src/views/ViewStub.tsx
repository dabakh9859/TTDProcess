import { useT } from '../i18n'
import type { TKey } from '../i18n'
import type { Command } from '../lib/commands'

interface ViewStubProps {
  /** Sidebar label key — reuses the recovered translations. */
  labelKey: TKey
  /** Tab id, matching the routing table and the i18n namespace. */
  tab: string
  /** i18n namespace whose keys belong to this view. */
  namespace: string
  /** How many recovered translation keys live under that namespace. */
  keyCount: number
  /** Backend commands this view called in the previous build. */
  commands: Command[]
  /** Number of ECharts instances the previous build rendered here. */
  charts?: number
}

/**
 * Placeholder for a view that has not been rewritten yet.
 *
 * It deliberately carries the view's spec — recovered from the old bundle —
 * so the work left to do is visible in the running app rather than buried in
 * a document. Delete the stub when you implement the real screen.
 */
export default function ViewStub({
  labelKey,
  tab,
  namespace,
  keyCount,
  commands,
  charts = 0,
}: ViewStubProps) {
  const { t } = useT()

  return (
    <div style={{ maxWidth: 820, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div
          style={{
            fontSize: 11,
            fontFamily: 'monospace',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--text-5)',
          }}
        >
          À réécrire
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 600, color: 'var(--text-1)' }}>
          {t(labelKey)}
        </h1>
        <div style={{ fontSize: 13, color: 'var(--text-4)', fontFamily: 'monospace' }}>
          {tab}
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 1,
          background: 'var(--border-1)',
          border: '1px solid var(--border-1)',
          borderRadius: 6,
          overflow: 'hidden',
        }}
      >
        {[
          { n: keyCount, l: `clés i18n · ${namespace}` },
          { n: commands.length, l: 'commandes backend' },
          { n: charts, l: 'graphes ECharts' },
        ].map((s) => (
          <div key={s.l} style={{ background: 'var(--bg-4)', padding: '12px 14px' }}>
            <div
              style={{
                fontSize: 22,
                fontFamily: 'monospace',
                color: 'var(--text-1)',
                lineHeight: 1.1,
              }}
            >
              {s.n}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-4)', marginTop: 2 }}>{s.l}</div>
          </div>
        ))}
      </div>

      {commands.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 13, color: 'var(--text-3)' }}>
            Contrat backend de cet écran :
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {commands.map((c) => (
              <span
                key={c}
                style={{
                  fontFamily: 'monospace',
                  fontSize: 11.5,
                  padding: '3px 8px',
                  borderRadius: 4,
                  background: 'var(--bg-4)',
                  border: '1px solid var(--border-2)',
                  color: 'var(--text-3)',
                }}
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
