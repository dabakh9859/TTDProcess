import {
  BookOpen,
  Brain,
  Calculator,
  ChartLine,
  CloudRain,
  Download,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  ScrollText,
  Settings,
  Sparkles,
  Table2,
  TrendingUp,
  Upload,
  Wrench,
  type LucideIcon,
} from 'lucide-react'

import { useT } from '../i18n'
import { useAppStore, type Tab } from '../store/useAppStore'
import type { TKey } from '../i18n'

interface NavItem {
  id: Tab
  labelKey: TKey
  icon: LucideIcon
}

interface NavSection {
  titleKey: TKey
  items: NavItem[]
}

/**
 * Note the ordering: within NETTOYAGE the sidebar lists aiTraining before
 * detection, which is *not* the order the views are declared in. Keep it —
 * it follows the workflow, not the code.
 */
const SECTIONS: NavSection[] = [
  {
    titleKey: 'sidebar.section.data',
    items: [
      { id: 'importation', labelKey: 'sidebar.import', icon: Upload },
      { id: 'envdata', labelKey: 'sidebar.envdata', icon: CloudRain },
      { id: 'tableau', labelKey: 'sidebar.table', icon: Table2 },
    ],
  },
  {
    titleKey: 'sidebar.section.analysis',
    items: [
      { id: 'calculs', labelKey: 'sidebar.calculsRaw', icon: Calculator },
      { id: 'calculsAdvanced', labelKey: 'sidebar.calculsAdvanced', icon: Calculator },
      { id: 'agregation', labelKey: 'sidebar.aggregation', icon: TrendingUp },
    ],
  },
  {
    titleKey: 'sidebar.section.cleaning',
    items: [
      { id: 'aiTraining', labelKey: 'sidebar.aiTraining', icon: Brain },
      { id: 'detection', labelKey: 'sidebar.dataCleaning', icon: Sparkles },
      { id: 'gapfilling', labelKey: 'sidebar.gapFilling', icon: Wrench },
    ],
  },
  {
    titleKey: 'sidebar.section.results',
    items: [
      { id: 'scenarios', labelKey: 'sidebar.scenarios', icon: LayoutGrid },
      { id: 'visualisation', labelKey: 'sidebar.visualization', icon: ChartLine },
    ],
  },
  {
    titleKey: 'sidebar.section.management',
    items: [
      { id: 'export', labelKey: 'sidebar.export', icon: Download },
      { id: 'journal', labelKey: 'sidebar.log', icon: ScrollText },
      { id: 'explications', labelKey: 'sidebar.docs', icon: BookOpen },
      { id: 'parametres', labelKey: 'sidebar.settings', icon: Settings },
    ],
  },
]

export default function Sidebar() {
  const currentTab = useAppStore((s) => s.currentTab)
  const setCurrentTab = useAppStore((s) => s.setCurrentTab)
  const collapsed = useAppStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)
  const { t } = useT()

  return (
    <aside
      style={{
        width: collapsed ? 52 : 220,
        height: '100%',
        background: 'var(--bg-1)',
        borderRight: '1px solid var(--border-2)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        transition: 'width 0.2s ease',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: collapsed ? '16px 10px' : '16px 16px',
          borderBottom: '1px solid var(--border-2)',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          minHeight: 65,
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'var(--accent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <TrendingUp size={18} color="#fff" />
        </div>
        {!collapsed && (
          <div>
            <div
              style={{
                color: 'var(--text-1)',
                fontWeight: 700,
                fontSize: 14,
                whiteSpace: 'nowrap',
              }}
            >
              TTDProcess
            </div>
            <div style={{ color: 'var(--text-4)', fontSize: 11 }}>v2.7.0</div>
          </div>
        )}
      </div>

      <nav style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '8px 0' }}>
        {SECTIONS.map((section) => (
          <div key={section.titleKey} style={{ marginBottom: 4 }}>
            {!collapsed && (
              <div
                style={{
                  padding: '8px 16px 4px',
                  fontSize: 11,
                  fontWeight: 600,
                  color: 'var(--text-5)',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                  whiteSpace: 'nowrap',
                }}
              >
                {t(section.titleKey)}
              </div>
            )}
            {collapsed && <div style={{ height: 6 }} />}

            {section.items.map((item) => {
              const Icon = item.icon
              const active = currentTab === item.id
              const label = t(item.labelKey)

              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  title={collapsed ? label : undefined}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: collapsed ? '8px 0' : '8px 16px',
                    justifyContent: collapsed ? 'center' : 'flex-start',
                    fontSize: 13,
                    fontWeight: active ? 500 : 400,
                    color: active ? 'var(--text-1)' : 'var(--text-3)',
                    background: active ? 'var(--border-2)' : 'transparent',
                    border: 'none',
                    borderLeft: active
                      ? '2px solid var(--accent)'
                      : '2px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    textAlign: 'left',
                    borderRadius: 0,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                  }}
                  onMouseEnter={(e) => {
                    if (active) return
                    e.currentTarget.style.background = 'var(--bg-5)'
                    e.currentTarget.style.color = 'var(--text-1)'
                  }}
                  onMouseLeave={(e) => {
                    if (active) return
                    e.currentTarget.style.background = 'transparent'
                    e.currentTarget.style.color = 'var(--text-3)'
                  }}
                >
                  <Icon size={16} style={{ flexShrink: 0 }} />
                  {!collapsed && <span>{label}</span>}
                </button>
              )
            })}
          </div>
        ))}
      </nav>

      <div
        style={{
          padding: collapsed ? '12px 0' : '12px 16px',
          borderTop: '1px solid var(--border-2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start',
        }}
      >
        <button
          onClick={toggleSidebar}
          title={collapsed ? 'Ouvrir le menu' : 'Réduire le menu'}
          style={{
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            color: 'var(--text-5)',
            fontSize: 11,
            borderRadius: 4,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-5)'
          }}
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          {!collapsed && <span>Réduire</span>}
        </button>
      </div>
    </aside>
  )
}
