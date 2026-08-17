import { open } from '@tauri-apps/plugin-dialog'
import {
  ArrowRight,
  Calculator,
  ChevronDown,
  CircleAlert,
  CircleCheckBig,
  FileSpreadsheet,
  LoaderCircle,
  Sparkles,
  Upload,
} from 'lucide-react'
import { useEffect, useState } from 'react'

import ColumnQuality from './ColumnQuality'
import PreviewSkeleton from './PreviewSkeleton'
import { card } from '../../components/Skeleton'
import { useT } from '../../i18n'
import { invoke } from '../../lib/tauri'
import { LOAD_TARGETS } from '../../lib/types'
import type { AppStatus, ColumnStats, LoadDataResult } from '../../lib/types'
import { useAppStore } from '../../store/useAppStore'

/** Last path segment, whichever separator the OS used. */
function baseName(path: string): string {
  return path.replace(/\\/g, '/').split('/').pop() ?? path
}

export default function ImportView() {
  const { t } = useT()
  const setDataLoaded = useAppStore((s) => s.setDataLoaded)
  const setFileName = useAppStore((s) => s.setFileName)
  const setTotalRows = useAppStore((s) => s.setTotalRows)
  const setTotalColumns = useAppStore((s) => s.setTotalColumns)
  const setColumns = useAppStore((s) => s.setColumns)
  const addLog = useAppStore((s) => s.addLog)
  const setCurrentTab = useAppStore((s) => s.setCurrentTab)
  const bumpDataVersion = useAppStore((s) => s.bumpDataVersion)

  const [filePath, setFilePath] = useState<string | null>(null)
  const [fileName, setLocalFileName] = useState('')
  const [sheetNames, setSheetNames] = useState<string[]>([])
  const [sheet, setSheet] = useState('')
  const [preview, setPreview] = useState<string[][] | null>(null)
  const [headerRow, setHeaderRow] = useState(0)
  const [dataStartRow, setDataStartRow] = useState(1)
  const [loadingSheets, setLoadingSheets] = useState(false)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [loadingData, setLoadingData] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ rows: number; cols: number } | null>(null)
  const [target, setTarget] = useState('raw')
  const [stats, setStats] = useState<ColumnStats[] | null>(null)
  const [loadingStats, setLoadingStats] = useState(false)

  // Reflect a restored session so the screen doesn't look empty after a restart.
  useEffect(() => {
    void (async () => {
      try {
        const status = await invoke<AppStatus>('get_app_status')
        if (!status.data_loaded) return

        setFilePath(status.file_path)
        setLocalFileName(baseName(status.file_path ?? ''))
        setLoaded(true)
        setResult({ rows: status.n_rows, cols: status.n_columns })

        setLoadingStats(true)
        try {
          setStats(await invoke<ColumnStats[]>('get_column_stats'))
        } catch {
          /* stats are a nicety, not a blocker */
        } finally {
          setLoadingStats(false)
        }
      } catch {
        /* nothing loaded */
      }
    })()
  }, [])

  async function loadPreview(path: string, sheetName: string) {
    setLoadingPreview(true)
    try {
      setPreview(
        await invoke<string[][]>('preview_file', { filePath: path, sheetName }),
      )
      setHeaderRow(0)
      setDataStartRow(1)
    } catch (e) {
      setError(t('import.error.preview', { error: String(e) }))
    } finally {
      setLoadingPreview(false)
    }
  }

  async function pickFile() {
    try {
      const picked = await open({
        filters: [
          {
            name: t('import.fileFilter.excelCsv'),
            extensions: ['xlsx', 'xls', 'csv', 'dat'],
          },
        ],
        multiple: false,
      })
      if (!picked) return

      const path = typeof picked === 'string' ? picked : (picked as { path: string }).path
      setFilePath(path)
      setLocalFileName(baseName(path))
      setLoaded(false)
      setPreview(null)
      setError(null)
      setResult(null)
      setLoadingSheets(true)

      try {
        const sheets = await invoke<string[]>('get_sheet_names', { filePath: path })
        setSheetNames(sheets)
        const first = sheets[0] ?? ''
        setSheet(first)
        if (first) await loadPreview(path, first)
      } catch (e) {
        setError(t('import.error.cannotRead', { error: String(e) }))
      } finally {
        setLoadingSheets(false)
      }
    } catch {
      /* dialog dismissed */
    }
  }

  async function onSheetChange(next: string) {
    setSheet(next)
    if (filePath) await loadPreview(filePath, next)
  }

  /**
   * Clicking a preview row retargets whichever marker makes sense: clicking the
   * current header row pushes the data start down to it, clicking at or above
   * the data start moves the header, anything below moves the data start.
   */
  function onRowClick(index: number) {
    if (index === headerRow) setDataStartRow(index)
    else if (index === dataStartRow || index < dataStartRow) setHeaderRow(index)
    else setDataStartRow(index)
  }

  async function loadData() {
    if (!filePath || !sheet) return
    setLoadingData(true)
    setError(null)

    try {
      const res = await invoke<LoadDataResult>('load_data', {
        filePath,
        sheetName: sheet,
        headerRow,
        dataStartRow,
        target,
      })

      setResult({ rows: res.num_rows, cols: res.num_cols })
      setLoaded(true)
      setFileName(fileName)
      setTotalRows(res.num_rows)
      setTotalColumns(res.num_cols)
      setColumns(res.columns)
      setDataLoaded(true)
      bumpDataVersion()
      addLog({
        level: 'Success',
        message: t('import.log.loaded', {
          name: fileName,
          rows: res.num_rows.toLocaleString(),
          cols: res.num_cols,
        }),
        timestamp: new Date().toISOString(),
      })

      setLoadingStats(true)
      try {
        setStats(await invoke<ColumnStats[]>('get_column_stats'))
      } catch {
        /* stats are a nicety, not a blocker */
      } finally {
        setLoadingStats(false)
      }
    } catch (e) {
      setError(t('import.error.loading', { error: String(e) }))
      addLog({
        level: 'Error',
        message: t('import.log.loadError', { name: fileName, error: String(e) }),
        timestamp: new Date().toISOString(),
      })
    } finally {
      setLoadingData(false)
    }
  }

  const canLoad = Boolean(filePath && sheet) && !loadingData && !loadingPreview

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h1 style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-1)', margin: 0 }}>
          {t('import.title')}
        </h1>
        <p style={{ fontSize: 14, color: 'var(--text-4)', marginTop: 4 }}>
          {t('import.subtitle')}
        </p>
      </div>

      <div
        onClick={pickFile}
        style={{
          ...card,
          borderStyle: 'dashed',
          borderWidth: 2,
          borderColor: filePath ? 'var(--success)' : 'var(--border-3)',
          background: filePath ? 'var(--success-tint-06)' : 'var(--bg-4)',
          borderRadius: 12,
          padding: '48px 24px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'border-color 0.2s, background 0.2s',
        }}
        onMouseEnter={(e) => {
          if (filePath) return
          e.currentTarget.style.borderColor = 'var(--accent)'
          e.currentTarget.style.background = 'var(--accent-tint-06)'
        }}
        onMouseLeave={(e) => {
          if (filePath) return
          e.currentTarget.style.borderColor = 'var(--border-3)'
          e.currentTarget.style.background = 'var(--bg-4)'
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 12,
          }}
        >
          {filePath ? (
            <>
              <FileSpreadsheet size={44} color="var(--success)" />
              <div>
                <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-1)' }}>
                  {fileName}
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-4)', marginTop: 4 }}>
                  {t('import.dropzone.changeFile')}
                </p>
              </div>
            </>
          ) : (
            <>
              <Upload size={44} color="var(--accent)" />
              <div>
                <p style={{ fontSize: 14, color: 'var(--text-3)' }}>
                  {t('import.dropzone.selectFile')}
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-5)', marginTop: 4 }}>
                  {t('import.dropzone.supportedFormats')}
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {filePath && (
        <div style={{ ...card, padding: 20 }}>
          <h2
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--text-1)',
              marginBottom: 16,
            }}
          >
            {t('import.options.title')}
          </h2>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: 16,
              alignItems: 'end',
            }}
          >
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 12,
                  color: 'var(--text-4)',
                  marginBottom: 6,
                }}
              >
                {t('import.options.excelSheet')}
              </label>
              <div style={{ position: 'relative' }}>
                <select
                  value={sheet}
                  onChange={(e) => void onSheetChange(e.target.value)}
                  disabled={loadingSheets || sheetNames.length === 0}
                  style={{
                    width: '100%',
                    background: 'var(--bg-2)',
                    border: '1px solid var(--border-2)',
                    borderRadius: 6,
                    padding: '8px 32px 8px 12px',
                    fontSize: 13,
                    color: 'var(--text-1)',
                    appearance: 'none',
                    cursor: 'pointer',
                    outline: 'none',
                  }}
                >
                  {loadingSheets ? (
                    <option>{t('import.loading')}</option>
                  ) : (
                    sheetNames.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))
                  )}
                </select>
                <ChevronDown
                  size={14}
                  color="var(--text-4)"
                  style={{
                    position: 'absolute',
                    right: 8,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    pointerEvents: 'none',
                  }}
                />
              </div>
            </div>

            <div
              style={{
                fontSize: 12,
                color: 'var(--text-4)',
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              <p>
                <Dot color="var(--success)" />
                {t('import.options.headerRow', { row: headerRow + 1 })}
              </p>
              <p>
                <Dot color="var(--info)" />
                {t('import.options.dataStartRow', { row: dataStartRow + 1 })}
              </p>
              <p style={{ color: 'var(--text-5)' }}>{t('import.options.clickToAdjust')}</p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <label style={{ fontSize: 11, color: 'var(--text-4)' }}>Charger comme</label>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                style={{
                  padding: '7px 10px',
                  fontSize: 12,
                  borderRadius: 6,
                  background: 'var(--bg-2)',
                  color: 'var(--text-1)',
                  border: '1px solid var(--border-2)',
                }}
              >
                {Object.entries(LOAD_TARGETS).map(([label, value]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: 10.5, color: 'var(--text-5)', lineHeight: 1.45 }}>
                {target === 'raw'
                  ? 'Import classique : remplace les données de travail et efface les calculs.'
                  : 'Le fichier devient cette étape du pipeline. Le reste de la session est conservé — tu peux ensuite entraîner, combler et recalculer à partir de là.'}
              </span>
            </div>

            <div>
              <button
                onClick={loadData}
                disabled={!canLoad}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  background: 'var(--accent)',
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: 13,
                  padding: '10px 20px',
                  border: 'none',
                  borderRadius: 8,
                  cursor: loadingData ? 'wait' : 'pointer',
                  opacity: canLoad ? 1 : 0.5,
                  transition: 'background 0.15s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'var(--accent-hover)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'var(--accent)'
                }}
              >
                {loadingData ? (
                  <>
                    <LoaderCircle
                      size={16}
                      style={{ animation: 'spin 1s linear infinite' }}
                    />
                    {t('import.loading')}
                  </>
                ) : (
                  <>
                    <Upload size={16} />
                    {t('import.loadData')}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            ...card,
            borderColor: 'var(--error-tint-30)',
            background: 'var(--error-tint-08)',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
          }}
        >
          <CircleAlert
            size={18}
            color="var(--error)"
            style={{ flexShrink: 0, marginTop: 2 }}
          />
          <p style={{ fontSize: 13, color: 'var(--error)' }}>{error}</p>
        </div>
      )}

      {loaded && result && (
        <div
          style={{
            ...card,
            borderColor: 'var(--success-tint-30)',
            background: 'var(--success-tint-08)',
            padding: '14px 18px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
          }}
        >
          <CircleCheckBig size={22} color="var(--success)" style={{ flexShrink: 0 }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <p
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: 'var(--success)',
                margin: 0,
              }}
            >
              {t('import.success.title')}
            </p>
            <p style={{ fontSize: 12, color: 'var(--text-3)', margin: '2px 0 0' }}>
              <strong>{result.rows.toLocaleString()}</strong> {t('import.success.rows')} ·{' '}
              <strong>{result.cols}</strong> {t('import.success.colsReady')}
            </p>
          </div>
          <button
            onClick={() => setCurrentTab('calculs')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 14px',
              background: 'var(--accent)',
              color: '#fff',
              border: '1px solid var(--accent)',
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 500,
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <Calculator size={13} /> {t('import.success.goToCalculs')} <ArrowRight size={12} />
          </button>
          <button
            onClick={() => setCurrentTab('detection')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 14px',
              background: 'transparent',
              color: 'var(--text-2)',
              border: '1px solid var(--border-2)',
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 500,
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <Sparkles size={13} /> {t('import.success.dataCleaning')}
          </button>
        </div>
      )}

      {loaded && (loadingStats || stats) && (
        <ColumnQuality stats={stats} loading={loadingStats} />
      )}

      {loadingPreview && !preview && <PreviewSkeleton />}

      {preview && preview.length > 0 && (
        <div style={{ ...card, overflow: 'hidden', position: 'relative' }}>
          {loadingPreview && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                zIndex: 20,
                background: 'var(--modal-backdrop)',
                backdropFilter: 'blur(2px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 12,
                  padding: '24px 32px',
                  background: 'var(--bg-4)',
                  border: '1px solid var(--border-2)',
                  borderRadius: 12,
                  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                }}
              >
                <LoaderCircle
                  size={22}
                  color="var(--accent)"
                  style={{ animation: 'spin 1s linear infinite' }}
                />
                <span style={{ fontSize: 13, color: 'var(--text-3)' }}>
                  {t('import.preview.updating')}
                </span>
              </div>
            </div>
          )}

          <div
            style={{
              padding: '12px 20px',
              borderBottom: '1px solid var(--border-2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <h2
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: 'var(--text-1)',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              {t('import.preview.title')}
            </h2>
            <p style={{ fontSize: 12, color: 'var(--text-4)' }}>
              {t('import.preview.clickRow')}
            </p>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {preview.slice(0, 25).map((row, i) => {
                  const isHeader = i === headerRow
                  const isStart = i === dataStartRow
                  return (
                    <tr
                      key={i}
                      onClick={() => onRowClick(i)}
                      style={{
                        borderBottom: '1px solid var(--row-divider)',
                        cursor: 'pointer',
                        background: isHeader
                          ? 'var(--success-tint-30)'
                          : isStart
                            ? 'var(--info-tint-30)'
                            : 'transparent',
                        borderLeft: isHeader
                          ? '4px solid var(--success)'
                          : isStart
                            ? '4px solid var(--info)'
                            : '4px solid transparent',
                        fontWeight: isHeader || isStart ? 600 : 400,
                        transition: 'background 0.1s',
                      }}
                      onMouseEnter={(e) => {
                        if (isHeader || isStart) return
                        e.currentTarget.style.background = 'var(--row-divider)'
                      }}
                      onMouseLeave={(e) => {
                        if (isHeader || isStart) return
                        e.currentTarget.style.background = 'transparent'
                      }}
                    >
                      <td
                        style={{
                          padding: '6px 12px',
                          color: 'var(--text-4)',
                          fontFamily: 'monospace',
                          width: 56,
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span
                            style={{
                              width: isHeader || isStart ? 12 : 8,
                              height: isHeader || isStart ? 12 : 8,
                              borderRadius: '50%',
                              flexShrink: 0,
                              background: isHeader
                                ? 'var(--success)'
                                : isStart
                                  ? 'var(--info)'
                                  : 'transparent',
                              boxShadow: isHeader
                                ? '0 0 0 2px var(--success-tint-25)'
                                : isStart
                                  ? '0 0 0 2px var(--info-tint-25)'
                                  : 'none',
                            }}
                          />
                          {i + 1}
                        </div>
                      </td>

                      {row.slice(0, 10).map((cell, c) => (
                        <td
                          key={c}
                          style={{
                            padding: '6px 12px',
                            color: 'var(--text-1)',
                            whiteSpace: 'nowrap',
                            maxWidth: 180,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {cell || (
                            <span
                              style={{
                                color: 'var(--muted-placeholder)',
                                fontStyle: 'italic',
                              }}
                            >
                              {t('import.preview.empty')}
                            </span>
                          )}
                        </td>
                      ))}

                      {row.length > 10 && (
                        <td
                          style={{
                            padding: '6px 12px',
                            color: 'var(--muted-placeholder)',
                            fontSize: 10,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {t('import.preview.moreCols', { count: row.length - 10 })}
                        </td>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div
            style={{
              padding: '10px 20px',
              borderTop: '1px solid var(--border-2)',
              display: 'flex',
              alignItems: 'center',
              gap: 24,
              fontSize: 12,
              color: 'var(--text-4)',
            }}
          >
            <Legend
              color="var(--success)"
              text={t('import.preview.headerLegend', { row: headerRow + 1 })}
            />
            <Legend
              color="var(--info)"
              text={t('import.preview.dataStartLegend', { row: dataStartRow + 1 })}
            />
            <span style={{ marginLeft: 'auto' }}>
              {preview.length > 25
                ? t('import.preview.showing25', { total: preview.length })
                : t('import.preview.rowsShown', { count: preview.length })}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function Dot({ color }: { color: string }) {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: color,
        marginRight: 6,
        verticalAlign: 'middle',
      }}
    />
  )
}

function Legend({ color, text }: { color: string; text: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
      {text}
    </span>
  )
}
