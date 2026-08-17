import { useEffect, useMemo, useState } from 'react'

import ThemeProvider from '../theme/ThemeProvider'

/** What the main window writes to localStorage before opening this window. */
interface DetachedPayload {
  option: Record<string, unknown>
  title: string
}

/**
 * Rework a chart option for the detached window: drop any fixed axis bounds so
 * the chart auto-scales, then give it zoom on both axes plus room for the
 * sliders. Recovered from the previous build — keep it when you wire ECharts
 * back in.
 */
export function toDetachedOption(
  option: Record<string, unknown>,
  tooltip: unknown,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...option, tooltip }

  const normaliseAxes = (axis: unknown) =>
    (Array.isArray(axis) ? axis : axis ? [axis] : []).map((a) => {
      const next = { ...(a as Record<string, unknown>) }
      delete next.min
      delete next.max
      if (next.type === 'value') next.scale = true
      return next
    })

  const xAxes = normaliseAxes(option.xAxis)
  const yAxes = normaliseAxes(option.yAxis)
  out.xAxis = xAxes.length <= 1 ? (xAxes[0] ?? option.xAxis) : xAxes
  out.yAxis = yAxes.length <= 1 ? (yAxes[0] ?? option.yAxis) : yAxes

  const xIndex = xAxes.map((_, i) => i)
  const yIndex = yAxes.map((_, i) => i)

  out.dataZoom = [
    {
      type: 'inside',
      xAxisIndex: xIndex,
      filterMode: 'none',
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
      moveOnMouseWheel: false,
    },
    {
      type: 'inside',
      yAxisIndex: yIndex,
      filterMode: 'none',
      zoomOnMouseWheel: 'shift',
      moveOnMouseMove: true,
      moveOnMouseWheel: false,
    },
    { type: 'slider', xAxisIndex: xIndex, filterMode: 'none', bottom: 6, height: 14 },
    { type: 'slider', yAxisIndex: yIndex, filterMode: 'none', right: 6, width: 14 },
  ]

  out.grid = {
    ...((option.grid as Record<string, unknown>) ?? {}),
    top: 40,
    left: 56,
    right: 48,
    bottom: 52,
    containLabel: true,
  }

  return out
}

/**
 * Standalone window for one detached chart.
 *
 * The main window stashes the chart option in localStorage under
 * `ttd-detached-<id>` and passes that key on the query string; nothing is sent
 * through Tauri IPC.
 */
export default function DetachedChartWindow() {
  const key = new URLSearchParams(window.location.search).get('key') ?? ''
  const [payload, setPayload] = useState<DetachedPayload | null>(null)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key)
      if (raw) setPayload(JSON.parse(raw) as DetachedPayload)
    } catch {
      /* malformed or missing payload — handled by the empty state below */
    }
  }, [key])

  const title = payload?.title ?? 'Graphique détaché'

  useEffect(() => {
    document.title = title
  }, [title])

  const seriesCount = useMemo(() => {
    const series = payload?.option?.series
    return Array.isArray(series) ? series.length : series ? 1 : 0
  }, [payload])

  return (
    <>
      <ThemeProvider />
      <div
        style={{
          height: '100vh',
          width: '100vw',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--bg-1)',
        }}
      >
        <div
          style={{
            padding: '8px 14px',
            borderBottom: '1px solid var(--border-2)',
            flexShrink: 0,
            display: 'flex',
            alignItems: 'baseline',
            gap: 12,
            overflow: 'hidden',
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>
            {title}
          </span>
          {payload && (
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>
              {seriesCount} série{seriesCount > 1 ? 's' : ''}
            </span>
          )}
        </div>

        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 24,
            textAlign: 'center',
            fontSize: 13,
            color: 'var(--text-4)',
          }}
        >
          {payload ? (
            // TODO: render with ECharts using toDetachedOption(payload.option, tooltip)
            <span>Rendu du graphe à brancher avec ECharts.</span>
          ) : (
            <span>
              Aucune donnée de graphique à afficher. Ferme cette fenêtre et relance
              «&nbsp;Détacher&nbsp;».
            </span>
          )}
        </div>
      </div>
    </>
  )
}
