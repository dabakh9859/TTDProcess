import { Languages, Moon, Settings, Sun } from 'lucide-react'
import type { CSSProperties, MouseEvent } from 'react'

import { useT } from '../i18n'
import { useAppStore } from '../store/useAppStore'

/** Shared styling for the three icon buttons on the right. */
const iconButton: CSSProperties = {
  padding: 6,
  borderRadius: 6,
  border: 'none',
  background: 'transparent',
  color: 'var(--text-4)',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

function hoverIn(e: MouseEvent<HTMLButtonElement>) {
  e.currentTarget.style.background = 'var(--border-2)'
  e.currentTarget.style.color = 'var(--text-1)'
}

function hoverOut(e: MouseEvent<HTMLButtonElement>, resting: string) {
  e.currentTarget.style.background = 'transparent'
  e.currentTarget.style.color = resting
}

export default function TopBar() {
  const dataLoaded = useAppStore((s) => s.dataLoaded)
  const calculationsComplete = useAppStore((s) => s.calculationsComplete)
  const setCurrentTab = useAppStore((s) => s.setCurrentTab)
  const theme = useAppStore((s) => s.theme)
  const setTheme = useAppStore((s) => s.setTheme)
  const language = useAppStore((s) => s.language)
  const setLanguage = useAppStore((s) => s.setLanguage)
  const { t } = useT()

  const isDark = theme === 'dark'
  const nextLang = language === 'fr' ? 'en' : 'fr'

  return (
    <header
      style={{
        height: 48,
        background: 'var(--bg-2)',
        borderBottom: '1px solid var(--border-2)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 16px',
        flexShrink: 0,
        gap: 10,
      }}
    >
      <span style={{ fontSize: 13, color: 'var(--text-4)' }}>{t('topbar.subtitle')}</span>

      <span
        style={{
          fontSize: 11,
          fontFamily: 'monospace',
          background: 'var(--bg-4)',
          border: '1px solid var(--border-2)',
          borderRadius: 6,
          padding: '2px 8px',
          color: 'var(--text-3)',
        }}
      >
        v2.7.0
      </span>

      {dataLoaded && (
        <span
          style={{
            fontSize: 11,
            fontWeight: 500,
            background: 'var(--success-tint-15)',
            color: 'var(--success)',
            border: '1px solid var(--success-tint-30)',
            borderRadius: 6,
            padding: '2px 10px',
          }}
        >
          {t('topbar.dataLoaded')}
        </span>
      )}

      {calculationsComplete && (
        <span
          style={{
            fontSize: 11,
            fontWeight: 500,
            background: 'var(--accent-tint-15)',
            color: 'var(--accent)',
            border: '1px solid var(--accent-tint-30)',
            borderRadius: 6,
            padding: '2px 10px',
          }}
        >
          {t('topbar.calculationsDone')}
        </span>
      )}

      <button
        onClick={() => setLanguage(nextLang)}
        title={t(nextLang === 'fr' ? 'topbar.langToFR' : 'topbar.langToEN')}
        style={{
          padding: '4px 8px',
          borderRadius: 6,
          border: '1px solid var(--border-2)',
          background: 'transparent',
          color: 'var(--text-3)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
        }}
        onMouseEnter={hoverIn}
        onMouseLeave={(e) => hoverOut(e, 'var(--text-3)')}
      >
        <Languages size={13} />
        {language.toUpperCase()}
      </button>

      <button
        onClick={() => setTheme(isDark ? 'light' : 'dark')}
        title={t(isDark ? 'topbar.themeLight' : 'topbar.themeDark')}
        style={iconButton}
        onMouseEnter={hoverIn}
        onMouseLeave={(e) => hoverOut(e, 'var(--text-4)')}
      >
        {isDark ? <Sun size={16} /> : <Moon size={16} />}
      </button>

      <button
        onClick={() => setCurrentTab('parametres')}
        title={t('topbar.settings')}
        style={iconButton}
        onMouseEnter={hoverIn}
        onMouseLeave={(e) => hoverOut(e, 'var(--text-4)')}
      >
        <Settings size={16} />
      </button>
    </header>
  )
}
