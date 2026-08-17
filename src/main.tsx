import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import './index.css'
import App from './App'
import DetachedChartWindow from './detached/DetachedChartWindow'

// A detached chart window is the same bundle loaded with ?view=detached.
// It renders only the chart, never the app shell.
const isDetached = new URLSearchParams(window.location.search).get('view') === 'detached'

createRoot(document.getElementById('root')!).render(
  <StrictMode>{isDetached ? <DetachedChartWindow /> : <App />}</StrictMode>,
)
