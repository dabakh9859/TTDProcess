import type { ComponentType } from 'react'

import type { Tab } from '../store/useAppStore'
import ImportView from './import'
import EnvDataView from './EnvDataView'
import TableView from './TableView'
import CalculationsView from './CalculationsView'
import AdvancedCalculationsView from './AdvancedCalculationsView'
import AggregationView from './AggregationView'
import AiTrainingView from './AiTrainingView'
import DetectionView from './DetectionView'
import GapFillingView from './GapFillingView'
import ScenariosView from './ScenariosView'
import VisualizationView from './VisualizationView'
import ExportView from './ExportView'
import JournalView from './JournalView'
import DocsView from './DocsView'
import SettingsView from './SettingsView'

/**
 * Every view is mounted at once by AppLayout and toggled with display,
 * so this list is the whole routing mechanism. Adding a screen means adding
 * a Tab to the store and an entry here.
 */
export const VIEWS: { key: Tab; Component: ComponentType }[] = [
  { key: 'importation', Component: ImportView },
  { key: 'envdata', Component: EnvDataView },
  { key: 'tableau', Component: TableView },
  { key: 'calculs', Component: CalculationsView },
  { key: 'calculsAdvanced', Component: AdvancedCalculationsView },
  { key: 'agregation', Component: AggregationView },
  { key: 'aiTraining', Component: AiTrainingView },
  { key: 'detection', Component: DetectionView },
  { key: 'gapfilling', Component: GapFillingView },
  { key: 'scenarios', Component: ScenariosView },
  { key: 'visualisation', Component: VisualizationView },
  { key: 'export', Component: ExportView },
  { key: 'journal', Component: JournalView },
  { key: 'explications', Component: DocsView },
  { key: 'parametres', Component: SettingsView },
]
