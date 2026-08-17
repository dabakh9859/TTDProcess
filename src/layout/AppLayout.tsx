import Sidebar from './Sidebar'
import StatusBar from './StatusBar'
import TopBar from './TopBar'
import ThemeProvider from '../theme/ThemeProvider'
import { useAppStore } from '../store/useAppStore'
import { VIEWS } from '../views'

/**
 * Every view stays mounted for the whole session; switching tabs only flips
 * `display`. That is what lets each view keep its local state — scroll
 * position, form inputs, loaded chart data — without pushing any of it into
 * the store. Swapping this for a router would silently reset all of it.
 */
export default function AppLayout() {
  const currentTab = useAppStore((s) => s.currentTab)

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--bg-2)',
        color: 'var(--text-1)',
      }}
    >
      <ThemeProvider />
      <Sidebar />

      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <TopBar />

        {VIEWS.map(({ key, Component }) => (
          <main
            key={key}
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: 24,
              background: 'var(--bg-2)',
              display: currentTab === key ? 'block' : 'none',
            }}
          >
            <Component />
          </main>
        ))}

        <StatusBar />
      </div>
    </div>
  )
}
