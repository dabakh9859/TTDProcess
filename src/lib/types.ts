/** Shapes returned by the Rust backend. Mirrors src-tauri/src/core/types.rs. */

export interface ColumnStats {
  name: string
  dtype: string
  count: number
  null_count: number
  null_percentage: number
  mean: number | null
  std: number | null
  min: number | null
  max: number | null
  median: number | null
  /** Outliers detected with the IQR method. */
  outlier_count: number
  is_timestamp: boolean
  /** Timestamp ordering violations (non-monotonic rows). */
  timestamp_gaps: number
  /** 0–100. */
  quality_score: number
}

/** `get_app_status` — what survived from the previous session. */
export interface AppStatus {
  data_loaded: boolean
  file_path: string | null
  sheet_name: string | null
  n_rows: number
  n_columns: number
  columns: string[]
  calculations_complete: boolean
  session_restored: boolean
  env_loaded: boolean
  ml_loaded: boolean
  ml_targets: string[]
  ml_predictors: string[]
}

/** `load_data` result. */
export interface LoadDataResult {
  num_rows: number
  num_cols: number
  columns: string[]
  /** Only present when the file was loaded as a derived pipeline stage. */
  target?: string
}

/**
 * Which pipeline slot an imported file becomes.
 *
 * `raw` is a plain import: it replaces the working data and clears every
 * derived result. Any other key loads the file *as* that stage instead, which
 * is how you bring back a T600 cleaned elsewhere and carry on from there
 * without owning the original logger file.
 */
export const LOAD_TARGETS: Record<string, string> = {
  'Raw Data (°C)': 'raw',
  'Cleaned Data': 'cleaned',
  Tslope: 'tslope',
  Baseline: 'baseline',
  'Delta-T': 'delta_t',
  T600: 't600',
  T0: 'tm',
  sT0: 'stm',
  T0i: 'tmi',
  K: 'k',
  'Sap Flow': 'sap_flow',
}
