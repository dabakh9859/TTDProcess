import { useEffect } from 'react'

import AppLayout from './layout/AppLayout'
import { invoke } from './lib/tauri'
import type { AppStatus } from './lib/types'
import { useAppStore } from './store/useAppStore'

/** Strip a full path down to its file name, whatever the separator. */
function baseName(path: string | null): string | null {
  return (path ?? '').split(/[\\/]/).pop() || null
}

/**
 * On mount, ask the backend whether a session survived the last run and
 * mirror it into the store. Failures are swallowed: a cold start with no
 * session is the normal case, not an error worth showing.
 */
export default function App() {
  const setDataLoaded = useAppStore((s) => s.setDataLoaded)
  const setFileName = useAppStore((s) => s.setFileName)
  const setTotalRows = useAppStore((s) => s.setTotalRows)
  const setTotalColumns = useAppStore((s) => s.setTotalColumns)
  const setColumns = useAppStore((s) => s.setColumns)
  const setCalculationsComplete = useAppStore((s) => s.setCalculationsComplete)
  const addLog = useAppStore((s) => s.addLog)

  useEffect(() => {
    void (async () => {
      try {
        const status = await invoke<AppStatus>('get_app_status')

        if (status.data_loaded) {
          setDataLoaded(true)
          setFileName(baseName(status.file_path))
          setTotalRows(status.n_rows)
          setTotalColumns(status.n_columns)
          setColumns(status.columns)

          if (status.session_restored) {
            addLog({
              level: 'Info',
              message: `Session restaurée : ${baseName(status.file_path) ?? 'session'} (${status.n_rows.toLocaleString()} lignes)`,
              timestamp: new Date().toISOString(),
            })
          }
        }
        setCalculationsComplete(status.calculations_complete)
      } catch {
        /* no session to restore */
      }
    })()
    // Runs once on mount, as in the original.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <AppLayout />
}
