use polars::prelude::{ChunkedArray, DataType, Int64Type, IntoSeries, Series};
use tauri::State;

use crate::core::data_loader;
use crate::core::types::ColumnStats;
use crate::state::AppState;
use crate::utils::logger;

#[tauri::command]
pub async fn get_sheet_names(file_path: String) -> Result<Vec<String>, String> {
    tokio::task::spawn_blocking(move || {
        data_loader::get_sheet_names(&file_path).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn preview_file(file_path: String, sheet_name: String) -> Result<Vec<Vec<String>>, String> {
    tokio::task::spawn_blocking(move || {
        data_loader::preview_file(&file_path, &sheet_name, 30).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn load_data(
    state: State<'_, AppState>,
    file_path: String,
    sheet_name: String,
    header_row: usize,
    data_start_row: usize,
    // Which slot the file becomes. `None`/"raw" keeps the historical behaviour
    // (fresh import: replaces raw and clears every derived result). A derived
    // key ("t600", "tm", …) loads the file AS that pipeline stage instead,
    // which is what lets a user bring back a T600 they cleaned elsewhere and
    // carry on — train on it, fill its gaps, recompute the sap flow from it —
    // without owning the original logger file.
    target: Option<String>,
) -> Result<serde_json::Value, String> {
    let fp = file_path.clone();
    let sn = sheet_name.clone();

    // Heavy I/O: load Excel file on a blocking thread
    let df = tokio::task::spawn_blocking(move || {
        data_loader::load_excel_file(&fp, &sn, header_row, data_start_row)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    // Drop entirely-empty rows that carry no TIMESTAMP (some exports interleave
    // blank rows). They have no time, so they're useless for analysis AND they
    // desync the calculation pipeline (compacted datetime index vs full frame).
    // Strip them here so every consumer — table, viz, pipeline — sees clean data.
    let n_before = df.height();
    let (df, dropped_blank) = match crate::core::timestamp_utils::drop_null_timestamp_rows(&df) {
        Ok((clean, n)) => (clean, n),
        Err(_) => (df, 0),
    };

    // Refuse an import that would land EMPTY. This happens when the TIMESTAMP
    // column exists but no cell parses (unknown date format) — every row is
    // then "blank" and gets dropped. Loading it anyway used to replace
    // raw_data with an empty frame AND clear every pipeline result, so a
    // single bad import silently destroyed the user's cleaning + calculations.
    // Bail BEFORE touching the state so the session survives.
    if df.height() == 0 && n_before > 0 {
        return Err(format!(
            "Aucune ligne exploitable : les {} lignes du fichier ont un horodatage vide ou dans un format non reconnu. \
             Vérifie la colonne TIMESTAMP et la ligne d'en-tête choisie. Les données actuelles n'ont pas été modifiées.",
            n_before
        ));
    }

    // Harmonise legacy sensor column names (e.g. SF_4-TA-1 → SF_4a-TA-1)
    // so data from files with different naming conventions forms continuous series.
    let (df, merged_cols) = data_loader::merge_legacy_sensor_columns(df);

    // Drop columns that are entirely null (empty trailing Excel columns, often
    // auto-named _0, _1, _2… by the loader). They carry no data and can confuse
    // type inference or column matching downstream.
    let (df, dropped_cols) = {
        let all_null: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|c| df.column(c).map(|s| s.null_count() == s.len()).unwrap_or(false))
            .map(|s| s.to_string())
            .collect();
        let n = all_null.len();
        if n > 0 { (df.drop_many(&all_null), n) } else { (df, 0) }
    };

    let num_rows = df.height();
    let num_cols = df.width();
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    let target_key = target.unwrap_or_else(|| "raw".to_string());

    // Store in state (fast, just moves the DataFrame)
    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;

        if target_key != "raw" {
            // Loading a file AS a pipeline stage. Everything else is left
            // alone: wiping the session here would defeat the purpose (the user
            // is assembling a state, not starting over).
            match target_key.as_str() {
                "cleaned"  => data.cleaned_data = Some(df.clone()),
                "tslope"   => data.results.tslope = Some(df.clone()),
                "baseline" => data.results.baseline = Some(df.clone()),
                "delta_t"  => data.results.delta_t = Some(df.clone()),
                "t600"     => data.results.t600 = Some(df.clone()),
                "tm"       => data.results.tm = Some(df.clone()),
                "stm"      => data.results.stm = Some(df.clone()),
                "tmi"      => data.results.tmi = Some(df.clone()),
                "k"        => data.results.k = Some(df.clone()),
                "sap_flow" => data.results.sap_flow = Some(df.clone()),
                other => return Err(format!("Destination inconnue : '{}'", other)),
            }
            // Remember that this stage came from a file. `run_pipeline` reads
            // this to resume from here instead of rebuilding the stage from
            // raw data that belongs to a different (older) import.
            if target_key != "cleaned" && !data.imported_stages.iter().any(|s| *s == target_key) {
                data.imported_stages.push(target_key.clone());
            }
            data.last_import_target = Some(target_key.clone());
            // A file loaded AS this stage replaces its contents wholesale, so a
            // pre-cleaning snapshot taken on the previous contents — possibly
            // another station's — no longer describes anything. Keeping it
            // would make the origin flags read every value as reconstructed.
            data.cleaning_pre_snapshots.remove(&target_key);
            // Several downstream commands bail with "Aucune donnée chargée" when
            // raw_data is empty, even when they never read it (recompute_from_
            // stage only uses it for the steps BEFORE the one it restarts from).
            // Seeding it with this file keeps that path open for someone who
            // only owns the derived export.
            if data.raw_data.is_none() {
                data.raw_data = Some(df);
                data.columns = columns.clone();
                data.file_path = Some(file_path.clone());
                data.sheet_name = Some(sheet_name.clone());
            }
            logger::add_log(
                &mut data.logs,
                crate::core::types::LogLevel::Success,
                format!(
                    "{} chargé comme « {} » : {} lignes x {} colonnes",
                    file_path, target_key, num_rows, num_cols
                ),
            );
            crate::utils::session_persist::save(&data);
            return Ok(serde_json::json!({
                "num_rows": num_rows,
                "num_cols": num_cols,
                "columns": columns,
                "target": target_key,
            }));
        }

        data.columns = columns.clone();
        data.file_path = Some(file_path.clone());
        data.sheet_name = Some(sheet_name.clone());
        data.raw_data = Some(df);
        data.cleaned_data = None;
        data.results = crate::core::types::CalculationResults::default();
        // A new import is a fresh context — clear any cleaning provenance from a
        // previous file/session so stale "T600 cleaned" CTAs don't linger.
        data.cleaning_source_dataset = None;
        data.cleaning_method_label = None;
        data.cleaning_path = None;
        data.cleaning_target_columns.clear();
        data.cleaning_locked_stages.clear();
        data.cleaning_pre_snapshots.clear();
        // Stages imported against the PREVIOUS file no longer describe anything.
        data.imported_stages.clear();
        data.last_import_target = None;

        logger::add_log(
            &mut data.logs,
            crate::core::types::LogLevel::Success,
            format!(
                "Loaded {} rows x {} columns from {}",
                num_rows, num_cols, file_path
            ),
        );
        if dropped_blank > 0 {
            logger::add_log(
                &mut data.logs,
                crate::core::types::LogLevel::Info,
                format!("{} lignes vides (sans horodatage) retirées au chargement", dropped_blank),
            );
        }
        if dropped_cols > 0 {
            logger::add_log(
                &mut data.logs,
                crate::core::types::LogLevel::Info,
                format!("{} colonnes entièrement vides retirées au chargement", dropped_cols),
            );
        }
        if merged_cols > 0 {
            logger::add_log(
                &mut data.logs,
                crate::core::types::LogLevel::Info,
                format!("{} colonnes capteurs harmonisées (nommage legacy unifié)", merged_cols),
            );
        }
        // Persist the new raw data + cleared results to disk so the
        // session survives a webview reload / Windows sleep.
        crate::utils::session_persist::save(&data);
    }

    Ok(serde_json::json!({
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
    }))
}

#[tauri::command]
pub async fn get_column_stats(state: State<'_, AppState>) -> Result<Vec<ColumnStats>, String> {
    // The import page calls this with no argument — once on mount and once
    // right after a load — so it has to resolve the dataset itself. It must be
    // the file the user actually just imported: after a load AS a stage,
    // `raw_data` still holds the previous, unrelated import, and reporting on
    // that made the page show logger columns (RECORD, U_Bat, …) for a T600
    // file, as if the import had silently loaded the old data.
    let df = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let staged = data.last_import_target.as_deref().and_then(|key| match key {
            "cleaned" => data.cleaned_data.as_ref(),
            "tslope" => data.results.tslope.as_ref(),
            "baseline" => data.results.baseline.as_ref(),
            "delta_t" => data.results.delta_t.as_ref(),
            "t600" => data.results.t600.as_ref(),
            "tm" => data.results.tm.as_ref(),
            "stm" => data.results.stm.as_ref(),
            "tmi" => data.results.tmi.as_ref(),
            "k" => data.results.k.as_ref(),
            "sap_flow" => data.results.sap_flow.as_ref(),
            _ => None,
        });
        // Fall back to raw when the slot was emptied since (reset, forced run).
        staged
            .or(data.raw_data.as_ref())
            .ok_or_else(|| "No data loaded".to_string())?
            .clone()
    };

    // Heavy computation on blocking thread
    tokio::task::spawn_blocking(move || compute_column_stats(&df))
        .await
        .map_err(|e| e.to_string())?
}

fn compute_column_stats(df: &polars::prelude::DataFrame) -> Result<Vec<ColumnStats>, String> {
    let mut stats_vec: Vec<ColumnStats> = Vec::new();

    for col_name in df.get_column_names() {
        let series = df.column(col_name).map_err(|e| e.to_string())?;
        let count = series.len();
        let null_count = series.null_count();
        let null_percentage = if count > 0 {
            (null_count as f64 / count as f64) * 100.0
        } else {
            0.0
        };

        let dtype_str = format!("{:?}", series.dtype());

        let is_timestamp = matches!(
            series.dtype(),
            DataType::Datetime(_, _) | DataType::Date | DataType::Time
        );

        let timestamp_gaps = if is_timestamp {
            let s: Series = series.clone().into_series();
            let casted: Series = match s.cast(&DataType::Int64) {
                Ok(c) => c,
                Err(_) => {
                    stats_vec.push(make_basic_stats(
                        col_name, &dtype_str, count, null_count, null_percentage, true, 0,
                    ));
                    continue;
                }
            };
            let ca: &ChunkedArray<Int64Type> = match casted.i64() {
                Ok(a) => a,
                Err(_) => {
                    stats_vec.push(make_basic_stats(
                        col_name, &dtype_str, count, null_count, null_percentage, true, 0,
                    ));
                    continue;
                }
            };
            let mut gaps = 0usize;
            let mut prev: Option<i64> = None;
            for i in 0..ca.len() {
                let opt_val: Option<i64> = ca.get(i);
                if let (Some(p), Some(v)) = (prev, opt_val) {
                    if v <= p {
                        gaps += 1;
                    }
                }
                if opt_val.is_some() {
                    prev = opt_val;
                }
            }
            gaps
        } else {
            0
        };

        let (mean, std, min, max, median, outlier_count) = if let Ok(f64_series) = series.f64() {
            let valid: Vec<f64> = f64_series
                .into_iter()
                .flatten()
                .filter(|v| v.is_finite())
                .collect();
            if valid.is_empty() {
                (None, None, None, None, None, 0)
            } else {
                let n = valid.len() as f64;
                let mean_val = valid.iter().sum::<f64>() / n;
                let variance =
                    valid.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n;
                let std_val = variance.sqrt();

                let min_val = valid
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let max_val = valid
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();

                let mut sorted = valid.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_val = if sorted.len() % 2 == 0 {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                };

                // IQR-based outlier detection — very conservative (×6) so the
                // recap only flags truly aberrant values (sensor failure /
                // typo), not legitimate rare events. Equivalent to ~6-7σ on a
                // normal distribution.
                //
                // Skip degenerate distributions where Q1==Q3 (IQR==0): this
                // happens on zero-inflated columns like Rain_mm or any sensor
                // that's zero most of the time. With IQR=0 the fences collapse
                // and EVERY non-modal value gets flagged as outlier, which is
                // false signal noise — better to surface 0 outliers than 200
                // bogus ones.
                let q1_idx = sorted.len() / 4;
                let q3_idx = (sorted.len() * 3) / 4;
                let q1 = sorted[q1_idx];
                let q3 = sorted[q3_idx];
                let iqr = q3 - q1;
                let outliers = if iqr <= f64::EPSILON {
                    0
                } else {
                    let lower_fence = q1 - 6.0 * iqr;
                    let upper_fence = q3 + 6.0 * iqr;
                    valid.iter().filter(|&&v| v < lower_fence || v > upper_fence).count()
                };

                (
                    Some(mean_val),
                    Some(std_val),
                    Some(min_val),
                    Some(max_val),
                    Some(median_val),
                    outliers,
                )
            }
        } else {
            (None, None, None, None, None, 0)
        };

        let quality_score = compute_quality_score(
            count, null_count, outlier_count, is_timestamp, timestamp_gaps,
        );

        stats_vec.push(ColumnStats {
            name: col_name.to_string(),
            dtype: dtype_str,
            count,
            null_count,
            null_percentage,
            mean,
            std,
            min,
            max,
            median,
            outlier_count,
            is_timestamp,
            timestamp_gaps,
            quality_score,
        });
    }

    Ok(stats_vec)
}

fn make_basic_stats(
    name: &str,
    dtype: &str,
    count: usize,
    null_count: usize,
    null_pct: f64,
    is_ts: bool,
    ts_gaps: usize,
) -> ColumnStats {
    let qs = compute_quality_score(count, null_count, 0, is_ts, ts_gaps);
    ColumnStats {
        name: name.to_string(),
        dtype: dtype.to_string(),
        count,
        null_count,
        null_percentage: null_pct,
        mean: None,
        std: None,
        min: None,
        max: None,
        median: None,
        outlier_count: 0,
        is_timestamp: is_ts,
        timestamp_gaps: ts_gaps,
        quality_score: qs,
    }
}

fn compute_quality_score(
    count: usize,
    null_count: usize,
    outlier_count: usize,
    is_timestamp: bool,
    timestamp_gaps: usize,
) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let completeness = 1.0 - (null_count as f64 / count as f64);
    let valid = count - null_count;
    let outlier_ratio = if valid > 0 {
        1.0 - (outlier_count as f64 / valid as f64).min(1.0)
    } else {
        1.0
    };
    let order_score = if is_timestamp && count > 1 {
        1.0 - (timestamp_gaps as f64 / (count - 1) as f64).min(1.0)
    } else {
        1.0
    };
    let score = completeness * 50.0 + outlier_ratio * 30.0 + order_score * 20.0;
    (score * 100.0).round() / 100.0
}
