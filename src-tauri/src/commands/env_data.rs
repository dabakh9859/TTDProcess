//! Commands for environmental data (VPD, PAR, …) used by the Tm DataEnv method.
//!
//! The env file is loaded separately from the main raw_data and stored in a
//! dedicated slot on AppState. Conditions/thresholds are carried alongside the
//! Tm method parameters when Run Pipeline is triggered — they're not persisted
//! server-side so the UI remains the source of truth.

use tauri::State;

use crate::core::data_loader;
use crate::core::types::LogLevel;
use crate::state::AppState;
use crate::utils::logger;

/// Resample env data onto a clean grid of round timestamps. Many flux
/// stations export rows offset from the round hour (e.g. 12h15, 12h45,
/// 13h15…) — we want values aligned to xx:00 / xx:30 so the downstream
/// vpd_par lookup matches T600 timestamps cleanly.
///
/// Algorithm: detect the source step (median of consecutive diffs),
/// build a target grid at multiples of that step from epoch, and linear-
/// interpolate every numeric column at each target timestamp. For
/// equidistant brackets (the common case here) linear interp degenerates
/// to the mean of the two surrounding points, which is what the user's
/// thesis advisor asked for.
///
/// No-op when timestamps are already aligned to round multiples of the
/// step (no offset detected).
/// Result of a resample pass. `realigned` tells the caller whether the
/// timestamps were actually shifted (= info worth surfacing to the user
/// in the journal), as opposed to a no-op when they were already aligned.
pub struct ResampleResult {
    pub df: polars::prelude::DataFrame,
    pub realigned: bool,
    pub detected_step_secs: i64,
    pub detected_offset_secs: i64,
}

fn resample_env_to_round_grid(
    df: &polars::prelude::DataFrame,
    ts_col: &str,
) -> anyhow::Result<ResampleResult> {
    use polars::prelude::*;
    use rayon::prelude::*;

    if df.height() < 2 {
        return Ok(ResampleResult {
            df: df.clone(),
            realigned: false,
            detected_step_secs: 0,
            detected_offset_secs: 0,
        });
    }

    // Drop rows where the timestamp is null FIRST so every downstream
    // index access (ts_secs[i] vs column_values[name][i]) refers to the
    // same logical row. ts_col_to_datetimes silently filters nulls, which
    // used to misalign ts vs values when even a single TS cell was empty
    // — the interpolator then read garbage values for every target.
    let df_owned: DataFrame;
    let df: &DataFrame = {
        let ts_series = df.column(ts_col)?;
        let null_count = ts_series.null_count();
        if null_count == 0 {
            df
        } else {
            df_owned = df.filter(&ts_series.is_not_null())?;
            &df_owned
        }
    };

    let timestamps = crate::core::timestamp_utils::ts_col_to_datetimes(df.column(ts_col)?)?;
    let bail_noop = |df: &DataFrame| ResampleResult {
        df: df.clone(),
        realigned: false,
        detected_step_secs: 0,
        detected_offset_secs: 0,
    };
    if timestamps.len() < 2 {
        return Ok(bail_noop(df));
    }
    // Sanity check: after the null-filter, ts_col_to_datetimes should
    // return exactly df.height() entries. If something else (string
    // parse failures, etc.) trims them further we bail to the original
    // DataFrame rather than producing misaligned output.
    if timestamps.len() != df.height() {
        return Ok(bail_noop(df));
    }
    let ts_secs: Vec<i64> = timestamps.iter().map(|d| d.timestamp()).collect();

    // Detect step from the median of positive consecutive diffs.
    let mut diffs: Vec<i64> = ts_secs.windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0)
        .collect();
    if diffs.is_empty() {
        return Ok(bail_noop(df));
    }
    diffs.sort_unstable();
    let step = diffs[diffs.len() / 2];
    if step <= 0 {
        return Ok(bail_noop(df));
    }

    // Already aligned? Skip the work.
    let already_aligned = ts_secs.iter().all(|t| t.rem_euclid(step) == 0);
    if already_aligned {
        return Ok(ResampleResult {
            df: df.clone(),
            realigned: false,
            detected_step_secs: step,
            detected_offset_secs: 0,
        });
    }
    let detected_offset = ts_secs[0].rem_euclid(step);

    // Build target grid: from floor(first/step)*step to ceil(last/step)*step.
    let first = ts_secs[0];
    let last = *ts_secs.last().unwrap();
    let start = first - first.rem_euclid(step);
    let last_aligned = last - last.rem_euclid(step);
    let end = if last_aligned < last { last_aligned + step } else { last_aligned };
    if end <= start {
        return Ok(bail_noop(df));
    }
    let n_target = ((end - start) / step + 1) as usize;
    let target: Vec<i64> = (0..n_target).map(|i| start + i as i64 * step).collect();

    // Parallel-interpolate every numeric column. The TIMESTAMP column is
    // rebuilt from the target grid; non-numeric columns are dropped (the
    // Tm pipeline only reads numeric env columns anyway).
    let column_names: Vec<String> = df.get_column_names()
        .iter()
        .filter(|n| **n != ts_col)
        .map(|s| s.to_string())
        .collect();

    // Pre-extract values once (sequential, cheap) so the parallel loop
    // doesn't need to lock the DataFrame. We try to cast EVERY non-ts
    // column to f64 — Polars handles `String → Float64` by parsing each
    // cell and inserting null for unparseable rows. The previous
    // `is_numeric()` gate dropped any column the loader stored as String
    // (CSV files with "NA"/empty cells trigger this), which produced an
    // env DataFrame with only TIMESTAMP and looked like an empty table.
    let column_values: Vec<(String, Vec<Option<f64>>)> = column_names.iter()
        .filter_map(|name| {
            let col = df.column(name).ok()?;
            let cast = col.cast(&DataType::Float64).ok()?;
            let vals: Vec<Option<f64>> = cast.f64().ok()?.into_iter().collect();
            Some((name.clone(), vals))
        })
        .collect();

    // Defensive fallback: if for any reason we couldn't extract any data
    // column, keep the original DataFrame intact. Better to ship the user
    // unaligned timestamps than to overwrite their data with timestamps
    // alone.
    if column_values.is_empty() {
        return Ok(bail_noop(df));
    }

    // Each column interpolated independently → trivially parallelizable.
    let interpolated_columns: Vec<Series> = column_values.par_iter()
        .map(|(name, vals)| {
            let interp: Vec<Option<f64>> = target.iter()
                .map(|&t| linear_interp_at(t, &ts_secs, vals))
                .collect();
            Series::new(name.as_str().into(), interp)
        })
        .collect();

    // Build the new TIMESTAMP series in microseconds (Polars Datetime us).
    let ts_us: Vec<i64> = target.iter().map(|s| s * 1_000_000).collect();
    let ts_series = Series::new(ts_col.into(), ts_us)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;

    let mut out: Vec<Series> = Vec::with_capacity(1 + interpolated_columns.len());
    out.push(ts_series);
    out.extend(interpolated_columns);

    Ok(ResampleResult {
        df: DataFrame::new(out)?,
        realigned: true,
        detected_step_secs: step,
        detected_offset_secs: detected_offset,
    })
}

/// Linear interpolation of `source_vals` at the target timestamp. Clamps
/// at edges (returns the closest source value when target is outside the
/// source range). Returns None only if both bracketing values are None.
fn linear_interp_at(
    target: i64,
    source_ts: &[i64],
    source_vals: &[Option<f64>],
) -> Option<f64> {
    if source_ts.is_empty() { return None; }
    let last_idx = source_ts.len() - 1;
    if target <= source_ts[0] { return source_vals[0]; }
    if target >= source_ts[last_idx] { return source_vals[last_idx]; }
    // Find idx such that source_ts[idx-1] <= target < source_ts[idx].
    let idx = source_ts.partition_point(|&t| t <= target);
    if idx == 0 || idx > last_idx { return None; }
    let t_lo = source_ts[idx - 1];
    let t_hi = source_ts[idx];
    if t_hi == t_lo { return source_vals[idx - 1]; }
    let v_lo = source_vals[idx - 1]?;
    let v_hi = source_vals[idx]?;
    let alpha = (target - t_lo) as f64 / (t_hi - t_lo) as f64;
    Some(v_lo + alpha * (v_hi - v_lo))
}

/// Gap-aware variant used when compiling several files into one continuous
/// series. Same linear interpolation as `linear_interp_at`, with two
/// differences that matter when stitching multi-year data:
///   - Outside the source range it returns None (no edge clamping) so the
///     padding before the first / after the last sample stays empty.
///   - When the two bracketing source samples are more than `max_gap` seconds
///     apart, it returns None — a genuine data gap (a missing month between
///     yearly files) is left as NaN instead of being bridged by a straight
///     line of invented values. An exact hit on a source timestamp always
///     returns that sample's value, even right before a gap.
fn linear_interp_at_capped(
    target: i64,
    source_ts: &[i64],
    source_vals: &[Option<f64>],
    max_gap: i64,
) -> Option<f64> {
    if source_ts.is_empty() { return None; }
    let last_idx = source_ts.len() - 1;
    if target < source_ts[0] || target > source_ts[last_idx] { return None; }
    if target == source_ts[last_idx] { return source_vals[last_idx]; }
    let idx = source_ts.partition_point(|&t| t <= target);
    if idx == 0 || idx > last_idx { return None; }
    let t_lo = source_ts[idx - 1];
    let t_hi = source_ts[idx];
    if t_lo == target { return source_vals[idx - 1]; } // exact sample → always valid
    if t_hi - t_lo > max_gap { return None; }          // strictly inside a real gap → NaN
    if t_hi == t_lo { return source_vals[idx - 1]; }
    let v_lo = source_vals[idx - 1]?;
    let v_hi = source_vals[idx]?;
    let alpha = (target - t_lo) as f64 / (t_hi - t_lo) as f64;
    Some(v_lo + alpha * (v_hi - v_lo))
}

#[tauri::command]
pub async fn load_env_data(
    state: State<'_, AppState>,
    file_path: String,
    sheet_name: String,
    header_row: usize,
    data_start_row: usize,
) -> Result<serde_json::Value, String> {
    let fp = file_path.clone();
    let sn = sheet_name.clone();

    let df = tokio::task::spawn_blocking(move || {
        data_loader::load_excel_file(&fp, &sn, header_row, data_start_row)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    let num_rows = df.height();
    let num_cols = df.width();
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.env_file_path = Some(file_path.clone());
        data.env_sheet_name = Some(sheet_name.clone());
        data.env_data = Some(df);

        logger::add_log(
            &mut data.logs,
            LogLevel::Success,
            format!(
                "Env file loaded: {} rows × {} columns from {}",
                num_rows, num_cols, file_path
            ),
        );
    }

    Ok(serde_json::json!({
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
    }))
}

#[tauri::command]
pub async fn get_env_info(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;

    let Some(df) = data.env_data.as_ref() else {
        return Ok(serde_json::json!({ "loaded": false }));
    };

    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    Ok(serde_json::json!({
        "loaded": true,
        "file_path": data.env_file_path,
        "sheet_name": data.env_sheet_name,
        "num_rows": df.height(),
        "num_cols": df.width(),
        "columns": columns,
    }))
}

#[tauri::command]
pub async fn clear_env_data(state: State<'_, AppState>) -> Result<(), String> {
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    data.env_data = None;
    data.env_file_path = None;
    data.env_sheet_name = None;
    logger::add_log(
        &mut data.logs,
        LogLevel::Info,
        "Env data cleared".to_string(),
    );
    Ok(())
}

/// Preview how many rows match the given conditions (for live UI feedback).
/// Conditions: `[{column: "VPD", operator: "<", threshold: 0.5}, ...]`
#[tauri::command]
pub async fn preview_env_conditions(
    state: State<'_, AppState>,
    conditions: Vec<EnvConditionInput>,
) -> Result<serde_json::Value, String> {
    let df = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        data.env_data
            .as_ref()
            .ok_or_else(|| "No env file loaded".to_string())?
            .clone()
    };

    tokio::task::spawn_blocking(move || {
        use polars::prelude::*;

        let total = df.height();
        if total == 0 || conditions.is_empty() {
            return Ok(serde_json::json!({
                "total": total,
                "matched": 0,
            }));
        }

        // AND-combine all conditions into a single boolean mask
        let mut mask = vec![true; total];
        for cond in &conditions {
            let col = df
                .column(&cond.column)
                .map_err(|e| format!("Column '{}' not found: {}", cond.column, e))?;
            let cast = col
                .cast(&DataType::Float64)
                .map_err(|e| format!("Cannot cast '{}' to f64: {}", cond.column, e))?;
            let values = cast.f64().map_err(|e| e.to_string())?;

            for (i, v) in values.into_iter().enumerate() {
                if !mask[i] {
                    continue;
                }
                let Some(x) = v else {
                    mask[i] = false;
                    continue;
                };
                let pass = match cond.operator.as_str() {
                    "<" => x < cond.threshold,
                    "<=" | "≤" => x <= cond.threshold,
                    ">" => x > cond.threshold,
                    ">=" | "≥" => x >= cond.threshold,
                    "==" | "=" => (x - cond.threshold).abs() < f64::EPSILON,
                    _ => false,
                };
                mask[i] = pass;
            }
        }

        let matched = mask.iter().filter(|&&b| b).count();

        Ok::<serde_json::Value, String>(serde_json::json!({
            "total": total,
            "matched": matched,
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct EnvConditionInput {
    pub column: String,
    pub operator: String,
    pub threshold: f64,
}

/// Quick min/max/p10/p90 for one or more columns in the env DataFrame.
/// Used by the Calculs UI to show the actual data range under each
/// threshold slider — so a user with VPD in hPa rather than kPa can see
/// at a glance that their values are 10× larger than the default
/// threshold expects, and adjust accordingly.
#[tauri::command]
pub async fn env_column_quick_stats(
    state: State<'_, AppState>,
    columns: Vec<String>,
) -> Result<serde_json::Value, String> {
    let df = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        data.env_data
            .as_ref()
            .ok_or_else(|| "No env file loaded".to_string())?
            .clone()
    };
    tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        use polars::prelude::*;
        let mut out: Vec<serde_json::Value> = Vec::with_capacity(columns.len());
        for col_name in &columns {
            if col_name.is_empty() { continue; }
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(_) => {
                    out.push(serde_json::json!({ "column": col_name, "available": false }));
                    continue;
                }
            };
            let cast = match col.cast(&DataType::Float64) {
                Ok(c) => c,
                Err(_) => {
                    out.push(serde_json::json!({ "column": col_name, "available": false }));
                    continue;
                }
            };
            let values = cast.f64().map_err(|e| e.to_string())?;
            let mut finite: Vec<f64> = values
                .into_iter()
                .filter_map(|v| v.filter(|x| x.is_finite()))
                .collect();
            if finite.is_empty() {
                out.push(serde_json::json!({ "column": col_name, "available": true, "n_finite": 0 }));
                continue;
            }
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = finite.len();
            let min = finite[0];
            let max = finite[n - 1];
            // 10th / 90th percentile — uses nearest-rank, fine for a hint.
            let p10 = finite[(n as f64 * 0.10).floor() as usize];
            let p90 = finite[(n as f64 * 0.90).floor() as usize];
            out.push(serde_json::json!({
                "column": col_name,
                "available": true,
                "n_finite": n,
                "min": min,
                "max": max,
                "p10": p10,
                "p90": p90,
            }));
        }
        Ok(serde_json::json!({ "stats": out }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Time-range filter mode for sliced env-data import. `auto` reads from the
/// already-loaded raw_data timestamps; `custom` uses the dates the user typed
/// in the UI; `all` skips filtering entirely (for small files).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum EnvTimeRangeMode {
    Auto,
    Custom { start: String, end: String },
    All,
}

/// Load an env file with three pre-filters in one go: header_row /
/// data_start_row (parse-time) + column projection (drop unused) + timestamp
/// slice (drop rows outside the analysis window). The result lands in the
/// same `env_data` slot the rest of the app already uses, so existing
/// callers (Calculs T0 method, Detection Voie B) just see "env data is now
/// loaded" without any code change.
#[tauri::command]
pub async fn load_env_data_sliced(
    state: State<'_, AppState>,
    file_path: String,
    sheet_name: String,
    header_row: usize,
    data_start_row: usize,
    selected_columns: Vec<String>,
    timestamp_column: String,
    time_range: EnvTimeRangeMode,
) -> Result<serde_json::Value, String> {
    // NOTE: do NOT `use polars::prelude::*` here — it shadows our enum
    // variants `Auto` and `All` (Polars exports types with the same names).
    // The prelude is imported inside the spawn_blocking closure where it's
    // actually needed.

    // Resolve the time window BEFORE we hit the file — auto mode needs to
    // read the main raw_data timestamps which are already in state.
    let (range_start, range_end): (Option<chrono::DateTime<chrono::Utc>>, Option<chrono::DateTime<chrono::Utc>>) = match &time_range {
        &EnvTimeRangeMode::All => (None, None),
        &EnvTimeRangeMode::Custom { ref start, ref end } => {
            let parse = |s: &str| -> Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::NaiveDateTime::parse_from_str(&format!("{} 00:00:00", s.trim()), "%Y-%m-%d %H:%M:%S")
                    .or_else(|_| chrono::NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S"))
                    .map(|n| n.and_utc())
                    .map_err(|e| format!("date invalide '{}': {}", s, e))
            };
            (Some(parse(start)?), Some(parse(end)?))
        }
        &EnvTimeRangeMode::Auto => {
            let app = state.inner.lock().map_err(|e| e.to_string())?;
            let df = app.raw_data.as_ref()
                .ok_or_else(|| "Aucune donnée principale chargée — utilise 'Manuel' ou 'Tout charger'".to_string())?;
            let ts_col = df.column("TIMESTAMP")
                .map_err(|_| "Colonne TIMESTAMP introuvable dans les données principales".to_string())?;
            let ts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
                .map_err(|e| e.to_string())?;
            let min = ts.iter().min().copied();
            let max = ts.iter().max().copied();
            // Exact alignment to the raw data range — no buffer. Earlier
            // versions added a ±1-day padding to give stability windows in
            // the VpdPar method some context at the edges, but visually it
            // looked like the env curves "overflowed" the sensor data range
            // when overlaid in the Visualisation tab, which confused users.
            // The VpdPar stability windows near edges have always tolerated
            // shorter contexts, so the buffer wasn't strictly required.
            (min, max)
        }
    };

    let fp = file_path.clone();
    let sn = sheet_name.clone();
    let ts_name = timestamp_column.clone();
    // Renamed to avoid shadowing polars::prelude::cols (a function in the
    // lazy expression API that we'd otherwise pull in below).
    let selected_cols = selected_columns.clone();

    // Heavy work off the async runtime.
    let (df, total_rows_in_file, resample) = tokio::task::spawn_blocking(move || -> Result<(polars::prelude::DataFrame, usize, ResampleResult), String> {
        use polars::prelude::*;
        // 1. Full parse using the existing loader (header_row + data_start_row aware).
        let mut df = crate::core::data_loader::load_excel_file(&fp, &sn, header_row, data_start_row)
            .map_err(|e| e.to_string())?;
        let total_in_file = df.height();

        // 2. Project columns: keep timestamp + selected.
        let mut keep: Vec<String> = Vec::with_capacity(selected_cols.len() + 1);
        if !ts_name.is_empty() && !keep.iter().any(|c| c == &ts_name) {
            keep.push(ts_name.clone());
        }
        for c in &selected_cols {
            if c != &ts_name && !keep.iter().any(|k| k == c) {
                keep.push(c.clone());
            }
        }
        if !keep.is_empty() {
            df = df.select(keep.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                .map_err(|e| format!("Sélection de colonnes : {}", e))?;
        }

        // 3. Time-slice if a range is requested.
        if let (Some(start), Some(end)) = (range_start, range_end) {
            let ts_col = df.column(&ts_name)
                .map_err(|_| format!("Colonne timestamp '{}' introuvable", ts_name))?;
            let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
                .map_err(|e| e.to_string())?;
            let mask_vec: Vec<bool> = datetimes.iter()
                .map(|t| *t >= start && *t <= end)
                .collect();
            let bool_chunked = BooleanChunked::from_iter_values("mask".into(), mask_vec.into_iter());
            df = df.filter(&bool_chunked)
                .map_err(|e| format!("Filtre temporel : {}", e))?;
        }

        // 4. Normalize the timestamp column to "TIMESTAMP" so downstream
        //    consumers (build_env_lookup, vpd_par, ML training) work
        //    regardless of how the column was named in the source file
        //    (DateSemih, Date_heure, TIMESTAMP_END, etc.).
        if !ts_name.is_empty() && ts_name != "TIMESTAMP" {
            df.rename(&ts_name, "TIMESTAMP".into())
                .map_err(|e| format!("Renommage timestamp : {}", e))?;
        }

        // 5. Resample to round timestamps via linear interpolation. Many
        //    flux-station exports are offset from xx:00 / xx:30 — this
        //    realigns them so downstream lookups against the T600 grid
        //    match cleanly. No-op when timestamps are already aligned.
        let resample = resample_env_to_round_grid(&df, "TIMESTAMP")
            .map_err(|e| format!("Recalage des timestamps : {}", e))?;
        let df = resample.df.clone();

        Ok((df, total_in_file, resample))
    })
    .await
    .map_err(|e| e.to_string())??;

    let num_rows = df.height();
    let num_cols = df.width();
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.env_file_path = Some(file_path.clone());
        data.env_sheet_name = Some(sheet_name.clone());
        data.env_data = Some(df);
        let realign_note = if resample.realigned {
            format!(
                " — recalage automatique : pas détecté {}s, décalage {}s → grille HH:00/HH:30 par moyenne arithmétique",
                resample.detected_step_secs, resample.detected_offset_secs
            )
        } else {
            String::new()
        };
        logger::add_log(
            &mut data.logs,
            LogLevel::Success,
            format!(
                "Env file (slice) chargé : {} / {} lignes × {} colonnes — {}{}",
                num_rows, total_rows_in_file, num_cols, file_path, realign_note
            ),
        );
        crate::utils::session_persist::save(&data);
    }

    let range_iso = match (range_start, range_end) {
        (Some(s), Some(e)) => Some(format!("{} → {}", s.format("%Y-%m-%d"), e.format("%Y-%m-%d"))),
        _ => None,
    };

    Ok(serde_json::json!({
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_rows_in_file": total_rows_in_file,
        "columns": columns,
        "range": range_iso,
        "realigned": resample.realigned,
        "detected_step_secs": resample.detected_step_secs,
        "detected_offset_secs": resample.detected_offset_secs,
    }))
}

/// Compile SEVERAL env files (e.g. one per year) into a single continuous
/// series. All files are assumed to share the same format (same header row,
/// timestamp column and columns) so one config applies to all.
///
/// Pipeline: parse + project + normalize each file → merge every row into a
/// timestamp-keyed BTreeMap (this sorts chronologically and drops duplicate
/// timestamps at the year junctions, keeping the first) → optional time slice
/// → project everything onto a REGULAR continuous time grid at the detected
/// step. Inside continuous data the grid is filled by linear interpolation
/// (which also realigns offset timestamps to round marks); genuine missing
/// periods between/within files are left as NaN — never invented. The result
/// lands in the same `env_data` slot the single-file path uses.
#[tauri::command]
pub async fn load_env_data_multi(
    state: State<'_, AppState>,
    file_paths: Vec<String>,
    sheet_name: String,
    header_row: usize,
    data_start_row: usize,
    selected_columns: Vec<String>,
    timestamp_column: String,
    time_range: EnvTimeRangeMode,
) -> Result<serde_json::Value, String> {
    if file_paths.is_empty() {
        return Err("Aucun fichier sélectionné.".to_string());
    }

    // Resolve the time window before touching the files (Auto reads raw_data).
    let (range_start, range_end): (Option<chrono::DateTime<chrono::Utc>>, Option<chrono::DateTime<chrono::Utc>>) = match &time_range {
        EnvTimeRangeMode::All => (None, None),
        EnvTimeRangeMode::Custom { start, end } => {
            let parse = |s: &str| -> Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::NaiveDateTime::parse_from_str(&format!("{} 00:00:00", s.trim()), "%Y-%m-%d %H:%M:%S")
                    .or_else(|_| chrono::NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S"))
                    .map(|n| n.and_utc())
                    .map_err(|e| format!("date invalide '{}': {}", s, e))
            };
            (Some(parse(start)?), Some(parse(end)?))
        }
        EnvTimeRangeMode::Auto => {
            let app = state.inner.lock().map_err(|e| e.to_string())?;
            let df = app.raw_data.as_ref()
                .ok_or_else(|| "Aucune donnée principale chargée — utilise 'Manuel' ou 'Tout charger'".to_string())?;
            let ts_col = df.column("TIMESTAMP")
                .map_err(|_| "Colonne TIMESTAMP introuvable dans les données principales".to_string())?;
            let ts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col).map_err(|e| e.to_string())?;
            (ts.iter().min().copied(), ts.iter().max().copied())
        }
    };

    let files = file_paths.clone();
    let sn = sheet_name.clone();
    let ts_name = timestamp_column.clone();
    let selected = selected_columns.clone();

    let (df, per_file, step_secs, n_gap_rows) = tokio::task::spawn_blocking(
        move || -> Result<(polars::prelude::DataFrame, Vec<serde_json::Value>, i64, usize), String> {
            use polars::prelude::*;
            use std::collections::btree_map::{BTreeMap, Entry};

            // Columns to keep: timestamp + selected (dedup, preserve order).
            let mut keep: Vec<String> = Vec::new();
            if !ts_name.is_empty() { keep.push(ts_name.clone()); }
            for c in &selected {
                if c != &ts_name && !keep.contains(c) { keep.push(c.clone()); }
            }

            let mut per_file: Vec<serde_json::Value> = Vec::with_capacity(files.len());
            let mut col_set: Vec<String> = Vec::new(); // non-timestamp columns, from first file
            let mut merged: BTreeMap<i64, Vec<Option<f64>>> = BTreeMap::new();

            for fp in &files {
                let mut df = crate::core::data_loader::load_excel_file(fp, &sn, header_row, data_start_row)
                    .map_err(|e| format!("Fichier '{}' : {}", fp, e))?;
                let rows_in_file = df.height();

                // Project to kept columns present in this file.
                if !keep.is_empty() {
                    let names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
                    let avail: Vec<&str> = keep.iter()
                        .filter(|k| names.iter().any(|n| n == *k))
                        .map(|s| s.as_str())
                        .collect();
                    if !avail.is_empty() {
                        df = df.select(avail)
                            .map_err(|e| format!("Fichier '{}' (sélection colonnes) : {}", fp, e))?;
                    }
                }
                // Normalize the timestamp column name to TIMESTAMP.
                let names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
                if !ts_name.is_empty() && ts_name != "TIMESTAMP" && names.iter().any(|n| n == &ts_name) {
                    df.rename(&ts_name, "TIMESTAMP".into())
                        .map_err(|e| format!("Fichier '{}' (renommage timestamp) : {}", fp, e))?;
                }
                // Drop null-timestamp rows so the ts index aligns with values.
                {
                    let ts_series = df.column("TIMESTAMP")
                        .map_err(|_| format!("Fichier '{}' : colonne timestamp introuvable", fp))?;
                    if ts_series.null_count() > 0 {
                        df = df.filter(&ts_series.is_not_null()).map_err(|e| e.to_string())?;
                    }
                }
                if col_set.is_empty() {
                    col_set = df.get_column_names().iter()
                        .filter(|n| **n != "TIMESTAMP").map(|s| s.to_string()).collect();
                }

                let ts = crate::core::timestamp_utils::ts_col_to_datetimes(df.column("TIMESTAMP").unwrap())
                    .map_err(|e| e.to_string())?;
                let colvals: Vec<Vec<Option<f64>>> = col_set.iter().map(|c| {
                    df.column(c).ok()
                        .and_then(|col| col.cast(&DataType::Float64).ok())
                        .and_then(|cast| cast.f64().ok().map(|ca| ca.into_iter().collect::<Vec<Option<f64>>>()))
                        .unwrap_or_else(|| vec![None; df.height()])
                }).collect();

                for i in 0..ts.len() {
                    let sec = ts[i].timestamp();
                    if let Entry::Vacant(slot) = merged.entry(sec) {
                        let row: Vec<Option<f64>> = (0..col_set.len())
                            .map(|ci| colvals[ci].get(i).copied().flatten())
                            .collect();
                        slot.insert(row);
                    }
                }
                per_file.push(serde_json::json!({ "file": fp, "rows": rows_in_file }));
            }

            if merged.is_empty() {
                return Err("Aucune donnée valide trouvée dans les fichiers sélectionnés.".to_string());
            }

            // Optional time slice on the merged source.
            if let (Some(start), Some(end)) = (range_start, range_end) {
                let (s, e) = (start.timestamp(), end.timestamp());
                merged.retain(|&sec, _| sec >= s && sec <= e);
                if merged.is_empty() {
                    return Err("Aucune ligne dans la fenêtre temporelle demandée.".to_string());
                }
            }

            let source_ts: Vec<i64> = merged.keys().copied().collect();
            let source_cols: Vec<Vec<Option<f64>>> = (0..col_set.len())
                .map(|ci| merged.values().map(|row| row[ci]).collect())
                .collect();

            // Detect the step (median of positive consecutive diffs).
            let mut diffs: Vec<i64> = source_ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0).collect();
            if diffs.is_empty() {
                return Err("Impossible de déterminer le pas temporel (un seul horodatage).".to_string());
            }
            diffs.sort_unstable();
            let step = diffs[diffs.len() / 2].max(1);
            let max_gap = step * 2; // bracket wider than this = real gap → NaN

            // Continuous grid aligned to multiples of step.
            let first = source_ts[0];
            let last = *source_ts.last().unwrap();
            let start_g = first - first.rem_euclid(step);
            let last_aligned = last - last.rem_euclid(step);
            let end_g = if last_aligned < last { last_aligned + step } else { last_aligned };
            let n_target = ((end_g - start_g) / step + 1).max(1) as usize;
            let grid: Vec<i64> = (0..n_target).map(|i| start_g + i as i64 * step).collect();

            // Interpolate each column onto the grid (gap-aware → NaN in real gaps).
            use rayon::prelude::*;
            let interp_vals: Vec<Vec<Option<f64>>> = (0..col_set.len()).into_par_iter().map(|ci| {
                grid.iter().map(|&t| linear_interp_at_capped(t, &source_ts, &source_cols[ci], max_gap)).collect()
            }).collect();

            // Count fully-empty grid rows (the real gaps now filled with NaN).
            let n_gap = (0..grid.len())
                .filter(|&r| (0..col_set.len()).all(|ci| interp_vals[ci][r].is_none()))
                .count();

            let ts_us: Vec<i64> = grid.iter().map(|s| s * 1_000_000).collect();
            let ts_series = Series::new("TIMESTAMP".into(), ts_us)
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None)).map_err(|e| e.to_string())?;
            let mut out_cols: Vec<Series> = Vec::with_capacity(1 + col_set.len());
            out_cols.push(ts_series);
            for (ci, name) in col_set.iter().enumerate() {
                out_cols.push(Series::new(name.as_str().into(), interp_vals[ci].clone()));
            }
            let out_df = DataFrame::new(out_cols).map_err(|e| e.to_string())?;

            Ok((out_df, per_file, step, n_gap))
        },
    )
    .await
    .map_err(|e| e.to_string())??;

    let num_rows = df.height();
    let num_cols = df.width();
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.env_file_path = Some(format!("{} fichiers compilés", file_paths.len()));
        data.env_sheet_name = Some(sheet_name.clone());
        data.env_data = Some(df);
        logger::add_log(
            &mut data.logs,
            LogLevel::Success,
            format!(
                "Env multi-fichiers compilé : {} fichiers → {} lignes (grille continue, pas {}s) × {} colonnes, {} trous (NaN)",
                file_paths.len(), num_rows, step_secs, num_cols, n_gap_rows
            ),
        );
        crate::utils::session_persist::save(&data);
    }

    let range_iso = match (range_start, range_end) {
        (Some(s), Some(e)) => Some(format!("{} → {}", s.format("%Y-%m-%d"), e.format("%Y-%m-%d"))),
        _ => None,
    };

    Ok(serde_json::json!({
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
        "per_file": per_file,
        "n_files": file_paths.len(),
        "step_secs": step_secs,
        "n_gap_rows": n_gap_rows,
        "range": range_iso,
    }))
}
