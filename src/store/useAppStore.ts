// Reconstruit à l'identique depuis le bundle (`Gt` dans index-0nLOWM5B.js).
// Clé de persistance, valeurs par défaut et `partialize` sont ceux d'origine :
// ne les change pas si tu veux que les sessions déjà enregistrées côté
// localStorage soient relues correctement.
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export type Tab =
  | 'importation'
  | 'envdata'
  | 'tableau'
  | 'calculs'
  | 'calculsAdvanced'
  | 'agregation'
  | 'detection'
  | 'gapfilling'
  | 'aiTraining'
  | 'scenarios'
  | 'visualisation'
  | 'export'
  | 'journal'
  | 'explications'
  | 'parametres'

export type LogLevel = 'Info' | 'Success' | 'Warning' | 'Error'

export interface LogEntry {
  level: LogLevel
  message: string
  timestamp: string
}

interface AppState {
  currentTab: Tab
  setCurrentTab: (tab: Tab) => void

  dataLoaded: boolean
  setDataLoaded: (v: boolean) => void

  calculationsComplete: boolean
  setCalculationsComplete: (v: boolean) => void

  fileName: string | null
  setFileName: (v: string | null) => void

  totalRows: number
  setTotalRows: (v: number) => void

  totalColumns: number
  setTotalColumns: (v: number) => void

  columns: string[]
  setColumns: (v: string[]) => void

  logs: LogEntry[]
  addLog: (entry: LogEntry) => void
  clearLogs: () => void

  theme: 'dark' | 'light'
  setTheme: (v: 'dark' | 'light') => void

  language: 'fr' | 'en'
  setLanguage: (v: 'fr' | 'en') => void

  tableTargetDataset: string | null
  setTableTargetDataset: (v: string | null) => void

  detectionValidated: boolean
  setDetectionValidated: (v: boolean) => void

  detectedColumns: string[]
  setDetectedColumns: (v: string[]) => void

  totalOutliers: number
  setTotalOutliers: (v: number) => void

  sidebarCollapsed: boolean
  toggleSidebar: () => void

  // Incrémenté après toute mutation backend ; les vues s'en servent comme
  // dépendance d'effet pour se recharger.
  dataVersion: number
  bumpDataVersion: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      currentTab: 'importation',
      setCurrentTab: (currentTab) => set({ currentTab }),

      dataLoaded: false,
      setDataLoaded: (dataLoaded) => set({ dataLoaded }),

      calculationsComplete: false,
      setCalculationsComplete: (calculationsComplete) => set({ calculationsComplete }),

      fileName: null,
      setFileName: (fileName) => set({ fileName }),

      totalRows: 0,
      setTotalRows: (totalRows) => set({ totalRows }),

      totalColumns: 0,
      setTotalColumns: (totalColumns) => set({ totalColumns }),

      columns: [],
      setColumns: (columns) => set({ columns }),

      logs: [],
      // Les entrées les plus récentes en tête, plafonnées à 100.
      addLog: (entry) => set((s) => ({ logs: [entry, ...s.logs].slice(0, 100) })),
      clearLogs: () => set({ logs: [] }),

      theme:
        (typeof window !== 'undefined' &&
          (localStorage.getItem('ttd-theme') as 'dark' | 'light')) ||
        'dark',
      setTheme: (theme) => set({ theme }),

      language:
        (typeof window !== 'undefined' &&
          (localStorage.getItem('ttd-lang') as 'fr' | 'en')) ||
        'fr',
      setLanguage: (language) => set({ language }),

      tableTargetDataset: null,
      setTableTargetDataset: (tableTargetDataset) => set({ tableTargetDataset }),

      detectionValidated: false,
      setDetectionValidated: (detectionValidated) => set({ detectionValidated }),

      detectedColumns: [],
      setDetectedColumns: (detectedColumns) => set({ detectedColumns }),

      totalOutliers: 0,
      setTotalOutliers: (totalOutliers) => set({ totalOutliers }),

      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

      dataVersion: 0,
      bumpDataVersion: () => set((s) => ({ dataVersion: s.dataVersion + 1 })),
    }),
    {
      name: 'ttd-app-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        currentTab: s.currentTab,
        theme: s.theme,
        language: s.language,
        tableTargetDataset: s.tableTargetDataset,
        sidebarCollapsed: s.sidebarCollapsed,
      }),
      version: 1,
    },
  ),
)
