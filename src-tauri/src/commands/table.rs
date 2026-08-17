use polars::prelude::*;
use tauri::State;

use crate::state::AppState;
use crate::utils::dataframe_serde::dataframe_page_to_json;
use crate::utils::logger;

#[tauri::command]
pub async fn get_table_page(
    state: State<'_, AppState>,
    dataset: String,
    page: usize,
    page_size: usize,
    search: Option<String>,
    // Chart downsample target: when page_size signals "give me everything"
    // (>= 50k), stride-downsample to ~this many points. None => 8k (crash-safe
    // default for detection/gap-fill charts). Visualisation passes a higher
    // value so zoomed-in curves stay detailed.
    max_rows: Option<usize>,
    // Optional time window (epoch ms) — keeps only rows whose TIMESTAMP falls
    // inside [ts_start_ms, ts_end_ms] BEFORE downsampling. Lets the Visualisation
    // tab load the zoomed-in window at FULL resolution (level-of-detail).
    ts_start_ms: Option<i64>,
    ts_end_ms: Option<i64>,
    // Optional column projection (+ TIMESTAMP/DATE auto-kept). The Visualisation
    // tab passes ONLY the plotted columns so even a 280k-row raw source weighs a
    // few MB — letting it load EVERY point without a downsample limit.
    select_columns: Option<Vec<String>>,
    // When true (and `select_columns` names the plotted columns), downsampling
    // keeps each bucket's min and max instead of every Nth row, so spikes stay
    // visible. The cleaning charts NEED this: the user selects outliers by
    // hand, and plain stride hid ~93% of them.
    keep_peaks: Option<bool>,
) -> Result<serde_json::Value, String> {
    // Saved-aggregation source: dataset key "agg_<uuid>" — read the JSON
    // result from disk and return it as a table page directly (no DataFrame
    // round-trip). This lets the Visualisation tab pull saved aggregations
    // exactly like any other dataset.
    if let Some(agg_id) = dataset.strip_prefix("agg_") {
        return read_aggregation_as_table_page(agg_id, page, page_size, search.as_deref())
            .await;
    }

    // Ad-hoc visualisation file: dataset key "file:<id>" — same pagination
    // path as everything else, the frame just lives in `viz_files`.
    if let Some(file_id) = dataset.strip_prefix("file:") {
        let (df, columns) = {
            let data = state.inner.lock().map_err(|e| e.to_string())?;
            let vf = data.viz_files.get(file_id)
                .ok_or_else(|| format!("Fichier '{}' non chargé", file_id))?;
            let columns: Vec<String> = vf.df.get_column_names().iter().map(|s| s.to_string()).collect();
            (vf.df.clone(), columns)
        };
        let df = project_columns(df, &select_columns);
        let df = match (ts_start_ms, ts_end_ms) {
            (Some(a), Some(b)) => filter_by_ms(&df, a, b),
            _ => df,
        };
        let source_rows = df.height();
        let df = if page_size >= 50_000 { downsample_for(&df, max_rows.unwrap_or(8_000), &select_columns, keep_peaks) } else { df };
        return tokio::task::spawn_blocking(move || {
            process_table_page(&df, &columns, page, page_size, search.as_deref(), source_rows)
        })
        .await
        .map_err(|e| e.to_string())?;
    }

    // Scenario overlay: dataset key "overlay:<scenarioId>:<dataset>" —
    // resolve the DataFrame from the in-memory overlay map and proceed
    // through the standard pagination path.
    if let Some(rest) = dataset.strip_prefix("overlay:") {
        let (overlay_id, inner_key) = match rest.split_once(':') {
            Some(pair) => pair,
            None => return Err(format!("Clé overlay mal formée : {}", dataset)),
        };
        let (df, columns) = {
            let data = state.inner.lock().map_err(|e| e.to_string())?;
            let overlay = data.loaded_scenario_overlays.get(overlay_id)
                .ok_or_else(|| format!("Overlay '{}' non chargé", overlay_id))?;
            let df = overlay.dataset(inner_key)
                .ok_or_else(|| format!("Dataset '{}' absent de l'overlay '{}'", inner_key, overlay_id))?;
            let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
            (df.clone(), columns)
        };
        let df = project_columns(df, &select_columns);
        let df = match (ts_start_ms, ts_end_ms) {
            (Some(a), Some(b)) => filter_by_ms(&df, a, b),
            _ => df,
        };
        let source_rows = df.height();
        let df = if page_size >= 50_000 { downsample_for(&df, max_rows.unwrap_or(8_000), &select_columns, keep_peaks) } else { df };
        return tokio::task::spawn_blocking(move || {
            process_table_page(&df, &columns, page, page_size, search.as_deref(), source_rows)
        })
        .await
        .map_err(|e| e.to_string())?;
    }

    // Clone the DataFrame out of the lock
    let (df, columns) = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = match dataset.as_str() {
            "raw" => data.raw_data.as_ref(),
            "cleaned" => data.cleaned_data.as_ref(),
            // Environmental data — same DataFrame the Tm/VPD-PAR method
            // and the Voie B ML cleaner use. Surfaces in the visualisation
            // page so the user can plot env vs. sap-flow on the same chart.
            "env" => data.env_data.as_ref(),
            "tslope" => data.results.tslope.as_ref(),
            "baseline" => data.results.baseline.as_ref(),
            "delta_t" => data.results.delta_t.as_ref(),
            "t600" => data.results.t600.as_ref(),
            "tm" => data.results.tm.as_ref(),
            "stm" => data.results.stm.as_ref(),
            "tmi" => data.results.tmi.as_ref(),
            "k" => data.results.k.as_ref(),
            "sap_flow" => data.results.sap_flow.as_ref(),
            "ttdplus_fourier" => data.results.ttdplus_fourier.as_ref(),
            "ttdplus_refs" => data.results.ttdplus_refs.as_ref(),
            "ttdplus_sap_flow" => data.results.ttdplus_sap_flow.as_ref(),
            // RegressionDiurne (T0) step-by-step verification tables
            "rd_regression" => data.results.rd_regression.as_ref(),
            "rd_result" => data.results.rd_result.as_ref(),
            "jh" => data.results.jh.as_ref(),
            "jhp" => data.results.jhp.as_ref(),
            "qh" => data.results.qh.as_ref(),
            "qd" => data.results.qd.as_ref(),
            _ => return Err(format!("Unknown dataset: {}", dataset)),
        }
        .ok_or_else(|| format!("Dataset '{}' is not available", dataset))?;

        let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        (df.clone(), columns)
    };
    // Lock is released here

    // A chart asking for "everything" (huge page_size) on an annual file
    // (~280k rows) would ship ~240 MB of JSON and crash the WebView. When the
    // caller wants a full-range view, stride-downsample to ~8k points (first &
    // last kept). Calculations are unaffected — they read the full backend data.
    let df = project_columns(df, &select_columns);
    let df = match (ts_start_ms, ts_end_ms) {
        (Some(a), Some(b)) => filter_by_ms(&df, a, b),
        _ => df,
    };
    let source_rows = df.height();
    let df = if page_size >= 50_000 { downsample_for(&df, max_rows.unwrap_or(8_000), &select_columns, keep_peaks) } else { df };

    // Heavy work (filter + paginate) on a blocking thread
    tokio::task::spawn_blocking(move || {
        process_table_page(&df, &columns, page, page_size, search.as_deref(), source_rows)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn process_table_page(
    df: &DataFrame,
    columns: &[String],
    page: usize,
    page_size: usize,
    search: Option<&str>,
    // Row count BEFORE downsampling. `total_rows` counts what we actually
    // return, so a chart could never tell whether it was looking at the whole
    // series or a 1-in-14 sample. The cleaning UI needs the difference to warn
    // "N rows hidden — zoom in to see them all".
    source_rows: usize,
) -> Result<serde_json::Value, String> {
    // Apply search filter on the full dataframe BEFORE pagination
    let working_df = match search {
        Some(term) if !term.is_empty() => {
            filter_dataframe(df, term).unwrap_or_else(|_| df.clone())
        }
        _ => df.clone(),
    };

    let total_rows = working_df.height();
    let total_pages = if total_rows == 0 {
        0
    } else {
        (total_rows + page_size - 1) / page_size
    };

    let start = page * page_size;
    let end = (start + page_size).min(total_rows);

    if start >= total_rows {
        return Ok(serde_json::json!({
            "columns": columns,
            "rows": [],
            "total_rows": total_rows,
            "total_pages": total_pages,
            "current_page": page,
            "source_rows": source_rows,
        }));
    }

    let page_df = working_df.slice(start as i64, end - start);
    let rows = dataframe_page_to_json(&page_df).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "current_page": page,
        "source_rows": source_rows,
    }))
}

/// Stride-downsample a DataFrame to ~`max_rows` rows (always keeping the first
/// and last so the time RANGE is preserved). Used to let a chart show the whole
/// span of a huge dataset (annual files ≈280k rows) without shipping every row
/// to the WebView, which crashes it. Display-only; the backend keeps the full
/// data for every calculation.
/// Stride-downsample to at most `max_rows` rows, keeping the first and last so
/// the time RANGE is preserved.
///
/// Deliberately a plain stride and NOT "prefer a row that carries a hole".
/// That variant was tried to make cleaning more visible at full zoom-out, and
/// it destroys the signal: with a stride of 9, letting one null win its window
/// discards up to 8 real values, so a column with a few thousand gaps loses its
/// diurnal envelope and its peaks (a T600 series topping at 58 rendered as if
/// it capped at 12). Fidelity of the values that remain matters more than
/// advertising the ones that were removed — the level-of-detail reload shows
/// every gap exactly as soon as the user zooms in.
/// Pick the downsampling strategy: peak-preserving when the caller asked for it
/// AND told us which columns are plotted, plain stride otherwise.
fn downsample_for(
    df: &DataFrame,
    max_rows: usize,
    select_columns: &Option<Vec<String>>,
    keep_peaks: Option<bool>,
) -> DataFrame {
    match (keep_peaks.unwrap_or(false), select_columns) {
        (true, Some(cols)) if !cols.is_empty() => {
            downsample_rows_keep_peaks(df, max_rows, cols)
        }
        _ => downsample_rows(df, max_rows),
    }
}

/// Downsample while KEEPING THE EXTREMES of `cols`.
///
/// Plain stride sampling silently hides spikes: with 105 000 rows shown as
/// 7 500 points (stride 14) a T600 column whose real maximum is 58 renders as
/// if it topped out at 11.5, and a sap-flow column peaking at 4575 shows 22.
/// On the cleaning charts that is not cosmetic — the user is asked to SELECT
/// the outliers, and 93% of them were never drawn.
///
/// Instead: split the frame into buckets and keep, per bucket, the rows holding
/// the min and the max of every requested column (plus the first and last row,
/// so the time range is preserved). Extremes therefore always survive, and the
/// point count stays bounded by `max_rows`.
///
/// `cols` should be the handful of columns actually being plotted — the bucket
/// count is divided by their number to stay within budget.
pub(crate) fn downsample_rows_keep_peaks(
    df: &DataFrame,
    max_rows: usize,
    cols: &[String],
) -> DataFrame {
    let n = df.height();
    if max_rows == 0 || n <= max_rows || cols.is_empty() {
        return downsample_rows(df, max_rows);
    }

    // Pull each column once as f64 (skip non-numeric — nothing to extremise).
    let series: Vec<Vec<Option<f64>>> = cols
        .iter()
        .filter_map(|c| df.column(c).ok())
        .filter(|s| s.dtype().is_numeric())
        .filter_map(|s| s.cast(&DataType::Float64).ok())
        .filter_map(|s| s.f64().ok().map(|ca| ca.into_iter().collect()))
        .collect();
    if series.is_empty() {
        return downsample_rows(df, max_rows);
    }

    // Each bucket yields up to 2 points per column (its min and its max).
    let per_bucket = 2 * series.len();
    let buckets = (max_rows / per_bucket).max(1);
    let bucket_len = n.div_ceil(buckets).max(1);

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    for b in 0..buckets {
        let start = b * bucket_len;
        if start >= n { break; }
        let end = ((b + 1) * bucket_len).min(n);
        for vals in &series {
            let mut lo_i: Option<usize> = None;
            let mut hi_i: Option<usize> = None;
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for i in start..end {
                if let Some(v) = vals[i] {
                    if !v.is_finite() { continue; }
                    if v < lo { lo = v; lo_i = Some(i); }
                    if v > hi { hi = v; hi_i = Some(i); }
                }
            }
            if let Some(i) = lo_i { keep[i] = true; }
            if let Some(i) = hi_i { keep[i] = true; }
        }
    }

    let mask = BooleanChunked::from_slice("mask".into(), &keep);
    df.filter(&mask).unwrap_or_else(|_| df.clone())
}

pub(crate) fn downsample_rows(df: &DataFrame, max_rows: usize) -> DataFrame {
    let n = df.height();
    if max_rows == 0 || n <= max_rows {
        return df.clone();
    }
    let stride = n.div_ceil(max_rows).max(1);
    let mask: Vec<bool> = (0..n).map(|i| i % stride == 0 || i == n - 1).collect();
    let mask_ca = BooleanChunked::from_slice("mask".into(), &mask);
    df.filter(&mask_ca).unwrap_or_else(|_| df.clone())
}

/// Project the DataFrame to TIMESTAMP/DATE + the requested columns (those that
/// exist). Shrinks the JSON payload so the Visualisation tab can load every row
/// of a wide raw source without shipping all 48 columns.
fn project_columns(df: DataFrame, sel: &Option<Vec<String>>) -> DataFrame {
    let Some(cols) = sel else { return df; };
    if cols.is_empty() { return df; }
    let existing: std::collections::HashSet<String> =
        df.get_column_names().iter().map(|s| s.to_string()).collect();
    let mut keep: Vec<String> = Vec::new();
    for tcol in ["TIMESTAMP", "DATE"] {
        if existing.contains(tcol) { keep.push(tcol.to_string()); }
    }
    for c in cols {
        if existing.contains(c) && !keep.contains(c) { keep.push(c.clone()); }
    }
    // Always project to what matched. If only TIMESTAMP matched (the requested
    // column doesn't exist), return that ALONE — never fall back to the full
    // frame, which on a 280k×48 raw source would ship ~240 MB and crash the UI.
    if keep.is_empty() { return df; }
    df.select(keep).unwrap_or(df)
}

/// Keep only rows whose TIMESTAMP is within [t0, t1] (epoch ms). Used for the
/// Visualisation level-of-detail: a zoomed-in window is re-fetched at full
/// resolution. Returns the df unchanged if there's no TIMESTAMP column.
fn filter_by_ms(df: &DataFrame, t0: i64, t1: i64) -> DataFrame {
    let Ok(ts_col) = df.column("TIMESTAMP") else { return df.clone(); };
    let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    let n = df.height();
    // Build a mask of EXACTLY n rows (one per df row). Null / unparseable
    // timestamps → false. Critical: ts_col_to_datetimes SKIPS those, yielding a
    // shorter Vec → df.filter length-mismatch → it returned the WHOLE frame
    // (annual files have blank-timestamp rows). We iterate aligned instead.
    let mut mask = vec![false; n];
    match ts_col.dtype() {
        DataType::String => {
            if let Ok(ca) = ts_col.str() {
                for (i, opt) in ca.into_iter().enumerate() {
                    if let Some(s) = opt {
                        if let Some(dt) = crate::core::timestamp_utils::parse_string_to_datetime(s) {
                            let m = dt.timestamp_millis();
                            if m >= lo && m <= hi { mask[i] = true; }
                        }
                    }
                }
            }
        }
        DataType::Datetime(tu, _) => {
            if let Ok(ca) = ts_col.datetime() {
                let to_ms: fn(i64) -> i64 = match tu {
                    TimeUnit::Nanoseconds => |t| t / 1_000_000,
                    TimeUnit::Microseconds => |t| t / 1_000,
                    TimeUnit::Milliseconds => |t| t,
                };
                for (i, opt) in ca.into_iter().enumerate() {
                    if let Some(t) = opt {
                        let m = to_ms(t);
                        if m >= lo && m <= hi { mask[i] = true; }
                    }
                }
            }
        }
        DataType::Int64 => {                    // microseconds, per ts_col_to_datetimes
            if let Ok(ca) = ts_col.i64() {
                for (i, opt) in ca.into_iter().enumerate() {
                    if let Some(t) = opt {
                        let m = t / 1_000;
                        if m >= lo && m <= hi { mask[i] = true; }
                    }
                }
            }
        }
        _ => return df.clone(),
    }
    let mask_ca = BooleanChunked::from_slice("mask".into(), &mask);
    df.filter(&mask_ca).unwrap_or_else(|_| df.clone())
}

/// Search across all columns: format each value as display string and check contains (case-insensitive)
fn filter_dataframe(df: &DataFrame, term: &str) -> PolarsResult<DataFrame> {
    let lower_term = term.to_lowercase();
    let height = df.height();

    let mut mask_vec = vec![false; height];

    for col in df.get_columns() {
        for i in 0..height {
            if mask_vec[i] {
                continue;
            }
            let val = col.get(i).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            let s = format!("{}", val);
            if s.to_lowercase().contains(&lower_term) {
                mask_vec[i] = true;
            }
        }
    }

    let mask = BooleanChunked::from_slice("mask".into(), &mask_vec);
    df.filter(&mask)
}

#[tauri::command]
pub async fn update_cell(
    state: State<'_, AppState>,
    dataset: String,
    row_index: usize,
    column: String,
    value: serde_json::Value,
) -> Result<(), String> {
    if column.to_uppercase() == "TIMESTAMP" {
        return Err("La colonne TIMESTAMP ne peut pas être modifiée.".to_string());
    }
    if !(dataset == "raw" || dataset == "cleaned") {
        return Err("Seules les tables Brutes et Nettoyées sont modifiables.".to_string());
    }

    let mut data = state.inner.lock().map_err(|e| e.to_string())?;

    let df_slot: &mut Option<DataFrame> = match dataset.as_str() {
        "raw" => &mut data.raw_data,
        "cleaned" => &mut data.cleaned_data,
        _ => unreachable!(),
    };
    let df = df_slot
        .as_mut()
        .ok_or_else(|| format!("Dataset '{}' non disponible.", dataset))?;

    if row_index >= df.height() {
        return Err(format!(
            "Ligne {} hors limites (taille: {}).",
            row_index,
            df.height()
        ));
    }

    let col = df
        .column(&column)
        .map_err(|e| format!("Colonne '{}' introuvable: {}", column, e))?;
    let dtype = col.dtype().clone();

    let new_series = match dtype {
        DataType::Float64 => {
            let parsed = parse_optional_f64(&value)
                .map_err(|e| format!("Valeur invalide pour {} (Float64): {}", column, e))?;
            let mut v: Vec<Option<f64>> = col
                .f64()
                .map_err(|e| e.to_string())?
                .into_iter()
                .collect();
            v[row_index] = parsed;
            Series::new(column.as_str().into(), v)
        }
        DataType::Float32 => {
            let parsed = parse_optional_f64(&value)
                .map_err(|e| format!("Valeur invalide pour {} (Float32): {}", column, e))?
                .map(|f| f as f32);
            let mut v: Vec<Option<f32>> = col
                .f32()
                .map_err(|e| e.to_string())?
                .into_iter()
                .collect();
            v[row_index] = parsed;
            Series::new(column.as_str().into(), v)
        }
        DataType::Int64 => {
            let parsed = parse_optional_i64(&value)
                .map_err(|e| format!("Valeur invalide pour {} (Int64): {}", column, e))?;
            let mut v: Vec<Option<i64>> = col
                .i64()
                .map_err(|e| e.to_string())?
                .into_iter()
                .collect();
            v[row_index] = parsed;
            Series::new(column.as_str().into(), v)
        }
        DataType::Int32 => {
            let parsed = parse_optional_i64(&value)
                .map_err(|e| format!("Valeur invalide pour {} (Int32): {}", column, e))?
                .map(|i| i as i32);
            let mut v: Vec<Option<i32>> = col
                .i32()
                .map_err(|e| e.to_string())?
                .into_iter()
                .collect();
            v[row_index] = parsed;
            Series::new(column.as_str().into(), v)
        }
        DataType::String => {
            let parsed = parse_optional_string(&value);
            let mut v: Vec<Option<String>> = col
                .str()
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|opt| opt.map(|s| s.to_string()))
                .collect();
            v[row_index] = parsed;
            Series::new(column.as_str().into(), v)
        }
        other => {
            return Err(format!(
                "Type de colonne non pris en charge pour l'édition: {:?}",
                other
            ));
        }
    };

    df.replace(&column, new_series)
        .map_err(|e| format!("Échec de la mise à jour: {}", e))?;

    Ok(())
}

fn parse_optional_f64(value: &serde_json::Value) -> Result<Option<f64>, String> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n.as_f64().map(Some).ok_or_else(|| "nombre invalide".into()),
        serde_json::Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Ok(None)
            } else {
                t.replace(',', ".")
                    .parse::<f64>()
                    .map(Some)
                    .map_err(|e| e.to_string())
            }
        }
        _ => Err("type de valeur non supporté".into()),
    }
}

fn parse_optional_i64(value: &serde_json::Value) -> Result<Option<i64>, String> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(Some)
            .ok_or_else(|| "entier invalide".into()),
        serde_json::Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Ok(None)
            } else {
                t.parse::<i64>().map(Some).map_err(|e| e.to_string())
            }
        }
        _ => Err("type de valeur non supporté".into()),
    }
}

fn parse_optional_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) if s.is_empty() => None,
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// Visualisation-specific source listing. Returns one entry per dataset
/// CURRENTLY LOADED in memory (skipping unloaded slots), with its column
/// names so the viz page can populate cascading source → column dropdowns
/// without making 15 separate `get_table_page` calls just to get headers.
#[tauri::command]
pub fn viz_list_sources(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;
    let mk = |key: &str, df: &Option<DataFrame>| -> Option<serde_json::Value> {
        df.as_ref().map(|d| {
            let cols: Vec<String> = d.get_column_names().iter()
                .map(|s| s.to_string())
                // Hide the InvT600_* intermediates (1/T600) from the
                // visualisation pickers — they're a calc step, not a curve.
                .filter(|c| !c.to_uppercase().starts_with("INVT600"))
                .collect();
            serde_json::json!({
                "key": key,
                "n_rows": d.height(),
                "columns": cols,
            })
        })
    };
    let sources: Vec<serde_json::Value> = [
        mk("raw", &data.raw_data),
        mk("cleaned", &data.cleaned_data),
        mk("env", &data.env_data),
        mk("tslope", &data.results.tslope),
        mk("baseline", &data.results.baseline),
        mk("delta_t", &data.results.delta_t),
        mk("t600", &data.results.t600),
        mk("tm", &data.results.tm),
        mk("stm", &data.results.stm),
        mk("tmi", &data.results.tmi),
        mk("k", &data.results.k),
        mk("sap_flow", &data.results.sap_flow),
        mk("ttdplus_fourier", &data.results.ttdplus_fourier),
        mk("ttdplus_refs", &data.results.ttdplus_refs),
        mk("ttdplus_sap_flow", &data.results.ttdplus_sap_flow),
        mk("rd_regression", &data.results.rd_regression),
        mk("rd_result", &data.results.rd_result),
        mk("jh", &data.results.jh),
        mk("jhp", &data.results.jhp),
        mk("qh", &data.results.qh),
        mk("qd", &data.results.qd),
    ]
    .into_iter()
    .flatten()
    .collect();

    // Ad-hoc files the user opened just to look at them. Listed after the
    // pipeline datasets so they never displace the "real" sources.
    let mut sources = sources;
    for (id, vf) in &data.viz_files {
        let cols: Vec<String> = vf.df.get_column_names().iter()
            .map(|s| s.to_string())
            .filter(|c| !c.to_uppercase().starts_with("INVT600"))
            .collect();
        sources.push(serde_json::json!({
            "key": format!("file:{}", id),
            "n_rows": vf.df.height(),
            "columns": cols,
            "friendly_label": vf.label,
            "origin": vf.path,
        }));
    }

    // Saved aggregations live on disk (not in memory) — list them here so
    // the Visualisation tab sees them alongside the calc DataFrames. Each
    // agg becomes a source with key "agg_<id>"; columns = period + columns,
    // n_rows = n_periods. The name + dataset go in the friendly_label /
    // origin fields so the frontend can render a human label.
    if let Ok(aggs) = crate::commands::aggregation::list_aggregations() {
        for m in aggs {
            let mut cols: Vec<String> = vec!["period".to_string()];
            cols.extend(m.columns.iter().cloned());
            sources.push(serde_json::json!({
                "key": format!("agg_{}", m.id),
                "n_rows": m.n_periods,
                "columns": cols,
                "friendly_label": m.name,
                "origin": format!("{} · {} · {}", m.dataset, m.period, m.operation),
            }));
        }
    }

    // Scenario overlays — every overlay contributes one source PER dataset
    // it carries. Keyed `overlay:<id>:<dataset_key>` and labelled with the
    // scenario name + dataset label so the user can pick e.g. the T600 of
    // scenario A side-by-side with the live T600.
    let overlay_dataset_labels: &[(&str, &str)] = &[
        ("raw", "Brut"), ("cleaned", "Nettoyé"), ("env", "Env."),
        ("tslope", "Tslope"), ("baseline", "Baseline"), ("delta_t", "Delta-T"),
        ("t600", "T600"), ("tm", "T0"), ("stm", "sT0"), ("tmi", "T0i"),
        ("k", "K"), ("sap_flow", "Fd"),
        ("ttdplus_fourier", "TTD+ Fourier"), ("ttdplus_refs", "TTD+ Refs"),
        ("ttdplus_sap_flow", "TTD+ Flux"),
        ("rd_regression", "rd_regression"), ("rd_result", "rd_result"),
        ("jh", "Jh"), ("jhp", "Jhp"), ("qh", "Qh"), ("qd", "Qd"),
    ];
    for (overlay_id, overlay) in data.loaded_scenario_overlays.iter() {
        for (key, pretty) in overlay_dataset_labels {
            if let Some(df) = overlay.dataset(key) {
                let cols: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
                sources.push(serde_json::json!({
                    "key": format!("overlay:{}:{}", overlay_id, key),
                    "n_rows": df.height(),
                    "columns": cols,
                    "friendly_label": format!("[{}] {}", overlay.name, pretty),
                    "origin": format!("Scénario · {}", overlay.created_at),
                }));
            }
        }
    }

    Ok(serde_json::json!({ "sources": sources }))
}

/// Open an arbitrary file (CSV / Excel) as an EXTRA visualisation source.
///
/// Deliberately does NOT touch `raw_data` or any pipeline result: the point is
/// to look at an exported T600 / sap-flow file — or another station's export —
/// next to the current session, not to replace it. The frame is parsed with the
/// same loader as the normal import, so anything the import accepts works here.
#[tauri::command]
pub async fn viz_load_file(
    state: State<'_, AppState>,
    file_path: String,
    sheet_name: Option<String>,
    header_row: Option<usize>,
    data_start_row: Option<usize>,
) -> Result<serde_json::Value, String> {
    let fp = file_path.clone();
    let sheet = sheet_name.unwrap_or_default();
    let hr = header_row.unwrap_or(0);
    let dsr = data_start_row.unwrap_or(hr + 1);

    let df = tokio::task::spawn_blocking(move || {
        crate::core::data_loader::load_excel_file(&fp, &sheet, hr, dsr)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    if df.height() == 0 {
        return Err("Le fichier ne contient aucune ligne de données.".to_string());
    }

    let label = std::path::Path::new(&file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("fichier")
        .to_string();
    let id = uuid::Uuid::new_v4().to_string();
    let n_rows = df.height();
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.viz_files.insert(id.clone(), crate::state::VizFile {
            label: label.clone(),
            path: file_path.clone(),
            df,
        });
        logger::add_log(
            &mut data.logs,
            crate::core::types::LogLevel::Success,
            format!("Fichier ouvert en visualisation : {} ({} lignes)", label, n_rows),
        );
    }

    Ok(serde_json::json!({
        "id": id,
        "key": format!("file:{}", id),
        "label": label,
        "n_rows": n_rows,
        "columns": columns,
    }))
}

/// Drop a file previously opened with `viz_load_file`.
#[tauri::command]
pub fn viz_unload_file(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    // Accept either the bare id or the "file:<id>" source key.
    let key = id.strip_prefix("file:").unwrap_or(&id).to_string();
    if let Some(vf) = data.viz_files.remove(&key) {
        logger::add_log(
            &mut data.logs,
            crate::core::types::LogLevel::Info,
            format!("Fichier retiré de la visualisation : {}", vf.label),
        );
    }
    Ok(())
}

#[tauri::command]
pub fn get_datasets_info(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;

    let dataset_info = |name: &str, df: &Option<DataFrame>| -> serde_json::Value {
        match df {
            Some(df) => serde_json::json!({
                "name": name,
                "available": true,
                "rows": df.height(),
                "columns": df.width(),
            }),
            None => serde_json::json!({
                "name": name,
                "available": false,
                "rows": 0,
                "columns": 0,
            }),
        }
    };

    let datasets = vec![
        dataset_info("raw", &data.raw_data),
        dataset_info("cleaned", &data.cleaned_data),
        dataset_info("tslope", &data.results.tslope),
        dataset_info("baseline", &data.results.baseline),
        dataset_info("delta_t", &data.results.delta_t),
        dataset_info("t600", &data.results.t600),
        dataset_info("tm", &data.results.tm),
        dataset_info("stm", &data.results.stm),
        dataset_info("tmi", &data.results.tmi),
        dataset_info("k", &data.results.k),
        dataset_info("sap_flow", &data.results.sap_flow),
        dataset_info("ttdplus_fourier", &data.results.ttdplus_fourier),
        dataset_info("ttdplus_refs", &data.results.ttdplus_refs),
        dataset_info("ttdplus_sap_flow", &data.results.ttdplus_sap_flow),
        dataset_info("rd_regression", &data.results.rd_regression),
        dataset_info("rd_result", &data.results.rd_result),
        dataset_info("jh", &data.results.jh),
        dataset_info("jhp", &data.results.jhp),
        dataset_info("qh", &data.results.qh),
        dataset_info("qd", &data.results.qd),
    ];

    Ok(serde_json::json!({ "datasets": datasets }))
}

/// Streams a saved aggregation off disk as a `get_table_page`-shaped JSON
/// response (so the Visualisation + Tableau pages don't need a special
/// loader). Sliced + filtered in pure JSON, no DataFrame round-trip — the
/// rows are small and already typed.
async fn read_aggregation_as_table_page(
    agg_id: &str,
    page: usize,
    page_size: usize,
    search: Option<&str>,
) -> Result<serde_json::Value, String> {
    let id = agg_id.to_string();
    let search = search.map(|s| s.to_lowercase());

    tokio::task::spawn_blocking(move || {
        let dir = crate::commands::aggregation::aggregations_root().join(&id);
        let result_path = dir.join("result.json");
        let meta_path = dir.join("meta.json");
        if !result_path.exists() {
            return Err(format!("Agrégation introuvable : {}", id));
        }
        let result_str = std::fs::read_to_string(&result_path).map_err(|e| e.to_string())?;
        let result: serde_json::Value =
            serde_json::from_str(&result_str).map_err(|e| e.to_string())?;
        let rows_arr = result
            .get("rows")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let raw_cols: Vec<String> = result
            .get("columns")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let mut columns: Vec<String> = vec!["period".to_string()];
        columns.extend(raw_cols);

        // Best-effort case-insensitive substring search across all cell
        // values — same semantics as the DataFrame path.
        let filtered: Vec<serde_json::Value> = match &search {
            Some(q) if !q.is_empty() => rows_arr
                .into_iter()
                .filter(|row| {
                    row.as_object()
                        .map(|obj| {
                            obj.values().any(|v| {
                                v.as_str()
                                    .map(|s| s.to_lowercase().contains(q))
                                    .unwrap_or_else(|| v.to_string().to_lowercase().contains(q))
                            })
                        })
                        .unwrap_or(false)
                })
                .collect(),
            _ => rows_arr,
        };

        let total_rows = filtered.len();
        let total_pages = total_rows.div_ceil(page_size.max(1)).max(1);
        let start = page.saturating_mul(page_size).min(total_rows);
        let end = start.saturating_add(page_size).min(total_rows);
        let page_rows: Vec<serde_json::Value> = filtered[start..end].to_vec();

        // Re-load meta just to surface the name in case future callers want it.
        let _meta = if meta_path.exists() {
            std::fs::read_to_string(&meta_path).ok()
        } else { None };

        Ok(serde_json::json!({
            "columns": columns,
            "rows": page_rows,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "current_page": page,
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(vals: Vec<Option<f64>>) -> DataFrame {
        let n = vals.len() as i64;
        DataFrame::new(vec![
            Series::new("TIMESTAMP".into(), (0..n).collect::<Vec<i64>>()),
            Series::new("T600_S1".into(), vals),
        ])
        .unwrap()
    }

    /// The point budget must not change — we only pick a different delegate
    /// inside each stride window. One delegate per window plus the final row,
    /// which is always kept so the time RANGE stays intact.
    #[test]
    fn downsample_keeps_the_row_budget() {
        let df = frame((0..100).map(|i| Some(i as f64)).collect());
        let out = downsample_rows(&df, 10);
        assert_eq!(out.height(), 11, "10 window delegates + the preserved last row");
    }

    /// Every kept value must be a real one, never a substitute chosen to make
    /// something else visible.
    ///
    /// This pins the decision NOT to bias the stride towards rows carrying
    /// holes. Doing so hid up to `stride - 1` genuine values per window, which
    /// flattened the diurnal envelope and clipped the peaks (a series topping
    /// at 58 was drawn as if it capped at 12).
    #[test]
    fn downsample_never_substitutes_a_hole_for_real_values() {
        // Stride 10, with a hole at index 7 — inside the first window but not
        // on its boundary.
        let mut vals: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
        vals[7] = None;
        let out = downsample_rows(&frame(vals), 10);

        let ts = out.column("TIMESTAMP").unwrap().i64().unwrap();
        assert_eq!(ts.get(0), Some(0), "window 0..10 keeps its own first row");

        let col = out.column("T600_S1").unwrap().f64().unwrap();
        assert_eq!(
            col.into_iter().filter(|v| v.is_none()).count(),
            0,
            "a hole between strides is simply not sampled — it shows on zoom-in",
        );
    }

    /// A hole that DOES land on the stride is kept as a hole, so gaps still
    /// render wherever the decimation happens to sample one.
    #[test]
    fn downsample_keeps_a_hole_that_lands_on_the_stride() {
        let mut vals: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
        vals[20] = None;
        let out = downsample_rows(&frame(vals), 10);
        let col = out.column("T600_S1").unwrap().f64().unwrap();
        assert_eq!(col.into_iter().filter(|v| v.is_none()).count(), 1);
    }

    /// With nothing missing, spacing stays exactly as before the change.
    #[test]
    fn downsample_keeps_even_spacing_when_nothing_is_missing() {
        let df = frame((0..50).map(|i| Some(i as f64)).collect());
        let out = downsample_rows(&df, 5);
        let ts = out.column("TIMESTAMP").unwrap().i64().unwrap();
        let picked: Vec<i64> = ts.into_no_null_iter().collect();
        assert_eq!(picked, vec![0, 10, 20, 30, 40, 49]);
    }
}
