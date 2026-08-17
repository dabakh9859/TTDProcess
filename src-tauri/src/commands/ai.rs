//! Tauri commands for the AI sidecar.
//!
//! Thin wrappers that forward calls to the long-lived Python child. The
//! frontend never talks to Python directly; it goes through these.
//!
//! `cleaning_apply_saits` is the one exception that does real work in
//! Rust: it runs `predict` on the sidecar, then merges the returned series
//! back into the live Polars DataFrame so the rest of the app picks it up
//! transparently from the Gap Filling tab.

use std::collections::HashMap;

use serde_json::Value;
use tauri::{AppHandle, State};

use crate::ai::AiSidecar;
use crate::state::AppState;
use crate::core::types::LogLevel;
use crate::utils::logger;

#[tauri::command]
pub async fn ai_health(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
) -> Result<Value, String> {
    sidecar.call(&app, "health", Value::Object(Default::default())).await
}

#[tauri::command]
pub async fn ai_train(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    state: State<'_, AppState>,
    spec: Value,
) -> Result<Value, String> {
    let mut spec = spec;
    // Train on a CALCULATED in-memory dataset (e.g. "t600") instead of xlsx:
    // export it to a temp Parquet and tell the sidecar to load it generically.
    if let Some(ds) = spec.get("dataset").and_then(|v| v.as_str()).map(str::to_string) {
        let (path, num_cols) = export_dataset_for_training(&state, &ds)?;
        let obj = spec.as_object_mut().ok_or("spec doit être un objet")?;
        obj.insert("files".into(), serde_json::json!([path]));
        obj.insert("generic_table".into(), serde_json::json!(true));
        if !obj.contains_key("feature_columns") && !obj.contains_key("selected_columns") {
            obj.insert("feature_columns".into(), serde_json::json!(num_cols));
        }
        obj.remove("dataset");
    }
    // Env features: the sidecar can't read arbitrary env file formats
    // (.xls/.csv) via openpyxl. The app already parsed the env file into memory
    // (Rust/calamine), so export THAT to a temp Parquet and hand the sidecar a
    // format it always reads. Avoids the "openpyxl does not support file
    // format" error and works regardless of the original env file type.
    let wants_env = spec
        .get("env_columns")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if wants_env {
        match export_env_for_training(&state) {
            Ok(env_path) => {
                if let Some(obj) = spec.as_object_mut() {
                    obj.insert("env_file".into(), serde_json::json!(env_path));
                }
            }
            Err(e) => return Err(format!("Données environnementales : {}", e)),
        }
    }
    sidecar.call(&app, "train", spec).await
}

/// Export the in-memory environmental data to a temp Parquet so the sidecar
/// can read it generically (the original file may be .xls/.csv which openpyxl
/// rejects). The env DataFrame already has a normalized "TIMESTAMP" column.
fn export_env_for_training(state: &State<'_, AppState>) -> Result<String, String> {
    use polars::prelude::*;
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let df = app
        .env_data
        .as_ref()
        .ok_or("aucune donnée environnementale chargée (onglet « Données env »)")?;
    // Unique filename per export: the path is stored in model.meta.env_file and
    // re-read at inference. A fixed name would let a second model (trained with
    // DIFFERENT env data) overwrite this file, so the first model would then
    // rebuild its VPD/PAR features from the wrong data — silently wrong
    // predictions. A uuid suffix keeps each model's env snapshot distinct.
    let mut path = std::env::temp_dir();
    path.push(format!("ttd_train_env_{}.parquet", uuid::Uuid::new_v4()));
    let file = std::fs::File::create(&path).map_err(|e| format!("création env temp: {}", e))?;
    let mut df_clone = df.clone();
    ParquetWriter::new(file)
        .finish(&mut df_clone)
        .map_err(|e| format!("écriture env parquet: {}", e))?;
    Ok(path.to_string_lossy().to_string())
}

/// Export an in-memory dataset to a temp Parquet for SAITS training, returning
/// the path and the numeric (non-TIMESTAMP) column names to use as features.
fn export_dataset_for_training(
    state: &State<'_, AppState>,
    dataset: &str,
) -> Result<(String, Vec<String>), String> {
    use polars::prelude::*;
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let df = dataset_ref(&app, dataset)?;

    let mut num_cols: Vec<String> = Vec::new();
    for s in df.get_columns() {
        let name = s.name().to_string();
        if name.eq_ignore_ascii_case("TIMESTAMP") { continue; }
        if matches!(s.dtype(), DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32) {
            num_cols.push(name);
        }
    }
    if num_cols.is_empty() {
        return Err(format!("Le dataset '{}' n'a aucune colonne numérique à entraîner.", dataset));
    }

    // Unique per export (see export_env_for_training): the path is recorded in
    // model.meta.files and reloaded at inference, so two trainings on the same
    // dataset key must not clobber each other's snapshot.
    let mut path = std::env::temp_dir();
    path.push(format!("ttd_train_{}_{}.parquet", dataset, uuid::Uuid::new_v4()));
    let file = std::fs::File::create(&path).map_err(|e| format!("création du fichier temp: {}", e))?;
    let mut df_clone = df.clone();
    ParquetWriter::new(file).finish(&mut df_clone).map_err(|e| format!("écriture parquet: {}", e))?;
    Ok((path.to_string_lossy().to_string(), num_cols))
}

/// Write an arbitrary DataFrame to a fresh temp parquet — used to feed SAITS the
/// LIVE in-memory data (current working slot) instead of requiring the user to
/// (re)import the original xlsx files.
fn export_df_temp(df: &polars::prelude::DataFrame, tag: &str) -> Result<String, String> {
    use polars::prelude::*;
    let mut path = std::env::temp_dir();
    path.push(format!("ttd_saits_{}_{}.parquet", tag, uuid::Uuid::new_v4()));
    let file = std::fs::File::create(&path).map_err(|e| format!("création temp: {}", e))?;
    let mut c = df.clone();
    ParquetWriter::new(file).finish(&mut c).map_err(|e| format!("écriture parquet: {}", e))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn ai_predict(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    spec: Value,
) -> Result<Value, String> {
    sidecar.call(&app, "predict", spec).await
}

#[tauri::command]
pub async fn ai_model_save(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    model_id: String,
    path: String,
) -> Result<Value, String> {
    let params = serde_json::json!({ "model_id": model_id, "path": path });
    sidecar.call(&app, "model_save", params).await
}

#[tauri::command]
pub async fn ai_model_load(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    path: String,
) -> Result<Value, String> {
    let params = serde_json::json!({ "path": path });
    sidecar.call(&app, "model_load", params).await
}

#[tauri::command]
pub async fn ai_inspect_files(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    spec: Value,
) -> Result<Value, String> {
    sidecar.call(&app, "inspect_files", spec).await
}

#[tauri::command]
pub async fn ai_list_models(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
) -> Result<Value, String> {
    sidecar.call(&app, "list_models", Value::Object(Default::default())).await
}

#[tauri::command]
pub async fn ai_list_env_columns(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    path: String,
) -> Result<Value, String> {
    let params = serde_json::json!({ "path": path });
    sidecar.call(&app, "list_env_columns", params).await
}

/// Run SAITS prediction over `gap_start..gap_end` and merge the imputed values
/// back into the dataset's target column. The sidecar still needs the original
/// xlsx files (the model only knows feat_cols + scaling) — we accept them as
/// `files` and fall back to whatever the model.meta recorded at training time.
///
/// The merge is point-by-point on TIMESTAMP: a row in the dataset gets the
/// imputed value iff its TIMESTAMP matches an entry the sidecar returned, the
/// target column is currently NaN, and the imputed value is finite.
#[tauri::command]
pub async fn cleaning_apply_saits(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    state: State<'_, AppState>,
    model_id: String,
    files: Vec<String>,
    target_column: String,
    gap_start: String,
    gap_end: String,
    dataset: Option<String>,
) -> Result<Value, String> {
    use polars::prelude::*;

    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // 1+2. Predict on the sidecar and parse into a millis→value map.
    let (by_ms, _) = saits_predict_map(
        &app, &sidecar, &model_id, &files, &target_column, &gap_start, &gap_end, true, None,
    ).await?;

    if by_ms.is_empty() {
        return Err("predict returned no finite values to merge".to_string());
    }

    // 3. Take the dataset, snapshot the pre-state, merge column.
    let (n_filled, target_col_clone) = {
        let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;

        let mut df = dataset_ref(&app_data, &dataset_key)?.clone();
        app_data.cleaning_pre_snapshots
            .entry(dataset_key.clone())
            .or_insert_with(|| df.clone());

        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;

        // Read the current target column (must be Float64).
        let tgt = df.column(&target_column)
            .map_err(|e| format!("Colonne cible '{}' introuvable: {}", target_column, e))?;
        let mut new_vals: Vec<Option<f64>> = match tgt.dtype() {
            DataType::Float64 => tgt.f64().map_err(|e| e.to_string())?
                .into_iter().collect(),
            DataType::Float32 => tgt.f32().map_err(|e| e.to_string())?
                .into_iter().map(|o| o.map(|x| x as f64)).collect(),
            other => return Err(format!(
                "La colonne '{}' a le type {:?}, attendu Float64", target_column, other
            )),
        };

        // Merge: only fill rows that are currently None/NaN AND fall in the map.
        let (pred_sorted, pred_tol) = sorted_preds(&by_ms);
        let mut n_filled = 0usize;
        for (i, dt) in datetimes.iter().enumerate() {
            let ms = dt.timestamp_millis();
            if let Some(v) = nearest_pred(&pred_sorted, ms, pred_tol) {
                let is_missing = match new_vals.get(i) {
                    Some(Some(x)) => !x.is_finite(),
                    Some(None) => true,
                    None => false,
                };
                if is_missing {
                    new_vals[i] = Some(v);
                    n_filled += 1;
                }
            }
        }

        if n_filled == 0 {
            return Err(format!(
                "Aucun trou comblé sur '{}' — la plage demandée n'a pas de NaN à imputer, ou les TIMESTAMP ne correspondent pas.",
                target_column,
            ));
        }

        let new_s = Series::new(target_column.as_str().into(), &new_vals);
        df.replace(&target_column, new_s)
            .map_err(|e| format!("replace column {}: {}", target_column, e))?;

        write_cleaned_to_slot(&mut app_data, &dataset_key, df)?;
        app_data.cleaning_source_dataset = Some(dataset_key.clone());
        app_data.cleaning_method_label = Some("SAITS".to_string());
        app_data.cleaning_path = Some("saits".to_string());
        if !app_data.cleaning_target_columns.iter().any(|x| x == &target_column) {
            app_data.cleaning_target_columns.push(target_column.clone());
        }
        logger::add_log(
            &mut app_data.logs,
            LogLevel::Success,
            format!(
                "Imputation SAITS appliquée : {} points comblés sur '{}' ({} → {})",
                n_filled, target_column, gap_start, gap_end,
            ),
        );
        crate::utils::session_persist::save(&app_data);
        (n_filled, target_column.clone())
    };

    Ok(serde_json::json!({
        "n_filled":      n_filled,
        "target_column": target_col_clone,
        "model_id":      model_id,
        "gap_start":     gap_start,
        "gap_end":       gap_end,
        "dataset":       dataset_key,
    }))
}

/// Parse a sidecar-emitted timestamp (RFC3339 or naive ISO) to UTC.
fn parse_imputed_ts(ts: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|d| d.with_timezone(&chrono::Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S")
                .or_else(|_| chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S%.f"))
                .map(|n| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(n, chrono::Utc))
        })
        .map_err(|e| format!("bad timestamp '{}': {}", ts, e))
}

/// Build a time-sorted (millis, value) list from a prediction map plus a match
/// tolerance (the model's prediction grid step, inferred as the median spacing
/// between consecutive predicted timestamps). Used for nearest-time merges so a
/// dataset at a finer/raw cadence than the model grid still aligns — exact-millis
/// matching would otherwise fill nothing ("Aucun trou comblé").
fn sorted_preds(map: &HashMap<i64, f64>) -> (Vec<(i64, f64)>, i64) {
    let mut v: Vec<(i64, f64)> = map.iter().map(|(&k, &val)| (k, val)).collect();
    v.sort_by_key(|&(k, _)| k);
    let tol = if v.len() >= 2 {
        let mut gaps: Vec<i64> = v.windows(2).map(|w| w[1].0 - w[0].0).filter(|&g| g > 0).collect();
        if gaps.is_empty() {
            i64::MAX / 4
        } else {
            gaps.sort_unstable();
            gaps[gaps.len() / 2]
        }
    } else {
        i64::MAX / 4
    };
    (v, tol)
}

/// Nearest predicted value to `ms` within `tol` millis (None if the closest
/// prediction is farther than one grid step). On an exact grid the distance is
/// 0, so this is a strict superset of the old exact-millis lookup.
fn nearest_pred(sorted: &[(i64, f64)], ms: i64, tol: i64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = sorted.partition_point(|&(k, _)| k < ms);
    let mut best: Option<(i64, f64)> = None;
    for cand in [idx.checked_sub(1), Some(idx)].into_iter().flatten() {
        if let Some(&(k, val)) = sorted.get(cand) {
            let d = (k - ms).abs();
            if d <= tol && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, val));
            }
        }
    }
    best.map(|(_, val)| val)
}

/// Run the sidecar `predict` for one column over a gap range and collect the
/// finite imputed values into a millis→value map.
async fn saits_predict_map(
    app: &AppHandle,
    sidecar: &AiSidecar,
    model_id: &str,
    files: &[String],
    target_column: &str,
    gap_start: &str,
    gap_end: &str,
    impute_existing: bool,
    env_file: Option<&str>,
) -> Result<(HashMap<i64, f64>, Option<f64>), String> {
    let mut params = serde_json::json!({
        "model_id":        model_id,
        "files":           files,
        "target_column":   target_column,
        "gap_start":       gap_start,
        "gap_end":         gap_end,
        "impute_existing": impute_existing,
    });
    // Fresh env snapshot overrides the model's (possibly stale) meta.env_file.
    if let Some(ef) = env_file {
        params["env_file"] = serde_json::json!(ef);
    }
    let pred = sidecar.call(app, "predict", params).await?;
    // The model's typical reconstruction error for this column (physical units),
    // used to calibrate detection. None for models trained before this existed.
    let recon_error = pred.get("recon_error").and_then(|v| v.as_f64()).filter(|v| v.is_finite());
    let imputed = pred.get("imputed")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "predict response missing 'imputed' array".to_string())?;

    let mut by_ms: HashMap<i64, f64> = HashMap::with_capacity(imputed.len());
    for item in imputed {
        let ts = item.get("timestamp").and_then(|v| v.as_str())
            .ok_or_else(|| "imputed row missing timestamp".to_string())?;
        let Some(val) = item.get("value").and_then(|v| v.as_f64()) else { continue };
        if !val.is_finite() { continue; }
        let dt = parse_imputed_ts(ts)?;
        by_ms.insert(dt.timestamp_millis(), val);
    }
    Ok((by_ms, recon_error))
}

/// Robust scale (MAD·1.4826) of a column's finite values — its natural spread.
fn robust_scale(vals: &[Option<f64>]) -> f64 {
    let mut finite: Vec<f64> = vals.iter().filter_map(|o| o.filter(|x| x.is_finite())).collect();
    if finite.len() < 3 { return 0.0; }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = |v: &[f64]| if v.len() % 2 == 0 {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    } else { v[v.len() / 2] };
    let med = median(&finite);
    let mut devs: Vec<f64> = finite.iter().map(|x| (x - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median(&devs) * 1.4826
}

/// Median of the finite values in `x[i-half ..= i+half]`, or None if < 12 pts.
fn rolling_median(x: &[Option<f64>], i: usize, half: usize) -> Option<f64> {
    let n = x.len();
    let lo = i.saturating_sub(half);
    let hi = (i + half + 1).min(n);
    let mut w: Vec<f64> = x[lo..hi].iter().filter_map(|o| o.filter(|v| v.is_finite())).collect();
    if w.len() < 12 { return None; }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(w[w.len() / 2])
}

/// Detection mask — a rolling ROBUST Z-SCORE on the residual `raw − pred`.
///
/// For each point we measure how far its residual sits from the LOCAL residual
/// profile: `dev = residual − rolling_median(residual)`, flagged when
/// `|dev| > threshold × max(rolling_MAD(dev), floor)`.
///
/// Why this and not a plain residual threshold: SAITS reconstructs some regions
/// structurally worse (the series START with no left-context, high-amplitude
/// seasons), producing a PERIODIC residual pattern at every diurnal trough. A
/// global threshold floods those normal points as false positives; a purely
/// local *magnitude* scale over-inflates and then MISSES real spikes inside
/// those regions. The robust z-score fixes both: the structured diurnal residual
/// forms the local distribution, and only a point that deviates from THAT
/// pattern (a genuine spike) is flagged — even in the high-error season.
///
/// `floor = max(recon error, signal MAD)` keeps near-constant channels (where
/// the residual spread collapses to ~0) from being carpet-flagged.
fn recon_outlier_mask(
    residuals: &[Option<f64>],
    vals: &[Option<f64>],
    recon_error: f64,
    threshold: f64,
) -> Vec<bool> {
    let floor = recon_error.max(robust_scale(vals)).max(1e-12);
    let n = residuals.len();
    let win = 192usize.min((n / 4).max(24));
    let half = win / 2;

    // 1. Deviation of each residual from its local (rolling-median) profile.
    let mut dev: Vec<Option<f64>> = vec![None; n];
    for i in 0..n {
        if let Some(ri) = residuals[i] {
            if ri.is_finite() {
                let med = rolling_median(residuals, i, half).unwrap_or(0.0);
                dev[i] = Some(ri - med);
            }
        }
    }

    // 2. Flag where |dev| exceeds threshold × local robust spread (MAD of dev).
    let abs_dev: Vec<Option<f64>> = dev.iter().map(|o| o.map(|v| v.abs())).collect();
    let mut mask = vec![false; n];
    for i in 0..n {
        let Some(di) = dev[i] else { continue };
        let lmad = rolling_median(&abs_dev, i, half).map(|m| m * 1.4826).unwrap_or(0.0);
        let scale = lmad.max(floor);
        if di.abs() > threshold * scale {
            mask[i] = true;
        }
    }
    mask
}

/// Classical outlier mask on a single column's values (NaN/None cells are
/// never flagged). `method` is "mad" (robust, default) or "zscore".
fn outlier_mask(vals: &[Option<f64>], method: &str, threshold: f64) -> Vec<bool> {
    let finite: Vec<f64> = vals.iter().filter_map(|o| o.filter(|x| x.is_finite())).collect();
    let n = finite.len();
    if n < 3 { return vec![false; vals.len()]; }

    let (center, scale) = if method == "zscore" {
        let mean = finite.iter().sum::<f64>() / n as f64;
        let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        (mean, var.sqrt())
    } else {
        // MAD (median absolute deviation), scaled to be std-comparable.
        let mut s = finite.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = |v: &[f64]| if v.len() % 2 == 0 {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        } else { v[v.len() / 2] };
        let med = median(&s);
        let mut devs: Vec<f64> = finite.iter().map(|x| (x - med).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (med, median(&devs) * 1.4826)
    };

    if scale <= 0.0 || !scale.is_finite() {
        return vec![false; vals.len()];
    }
    vals.iter()
        .map(|o| match o {
            Some(x) if x.is_finite() => ((x - center) / scale).abs() > threshold,
            _ => false,
        })
        .collect()
}

/// Read a numeric column as `Vec<Option<f64>>` (accepts Float64/Float32).
fn col_as_f64_opts(
    df: &polars::prelude::DataFrame,
    column: &str,
) -> Result<Vec<Option<f64>>, String> {
    use polars::prelude::*;
    let s = df.column(column)
        .map_err(|e| format!("Colonne '{}' introuvable: {}", column, e))?;
    match s.dtype() {
        DataType::Float64 => Ok(s.f64().map_err(|e| e.to_string())?.into_iter().collect()),
        DataType::Float32 => Ok(s.f32().map_err(|e| e.to_string())?
            .into_iter().map(|o| o.map(|x| x as f64)).collect()),
        other => Err(format!("La colonne '{}' a le type {:?}, attendu Float64", column, other)),
    }
}

/// Voie B — IA via SAITS: per target column, detect outliers (classical) → set
/// them NaN, then impute every NaN (outliers + pre-existing gaps) with the
/// trained SAITS model over the dataset's full time range. Both steps in one
/// pass; predict runs first (it rebuilds features from `files`, independent of
/// the live DataFrame's NaNs).
#[tauri::command]
pub async fn cleaning_apply_saits_clean(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    state: State<'_, AppState>,
    model_id: String,
    files: Vec<String>,
    target_columns: Vec<String>,
    detect_method: String,
    detect_threshold: f64,
    dataset: Option<String>,
) -> Result<Value, String> {
    use polars::prelude::*;

    if target_columns.is_empty() {
        return Err("Aucune colonne cible sélectionnée.".to_string());
    }
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // 0. Read the dataset's time range (short lock) → SAITS gap window.
    let (gap_start, gap_end) = {
        let app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = dataset_ref(&app_data, &dataset_key)?;
        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let dts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
        let min = dts.iter().min().ok_or("dataset vide")?;
        let max = dts.iter().max().ok_or("dataset vide")?;
        (min.format("%Y-%m-%dT%H:%M:%S").to_string(),
         max.format("%Y-%m-%dT%H:%M:%S").to_string())
    };

    // 1. Predict each target column on the sidecar (async, no lock held).
    let mut maps: Vec<(String, HashMap<i64, f64>)> = Vec::with_capacity(target_columns.len());
    for col in &target_columns {
        let (m, _) = saits_predict_map(&app, &sidecar, &model_id, &files, col, &gap_start, &gap_end, false, None).await?;
        maps.push((col.clone(), m));
    }

    // 2. Detect outliers → NaN, then merge imputed into NaN cells (one lock).
    let mut per_column: Vec<Value> = Vec::with_capacity(target_columns.len());
    let mut total_out = 0usize;
    let mut total_fill = 0usize;
    {
        let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let mut df = dataset_ref(&app_data, &dataset_key)?.clone();
        app_data.cleaning_pre_snapshots
            .entry(dataset_key.clone())
            .or_insert_with(|| df.clone());

        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;

        for (col, map) in &maps {
            let mut vals = col_as_f64_opts(&df, col)?;

            // a. Detect → NaN.
            let mask = outlier_mask(&vals, &detect_method, detect_threshold);
            let mut n_out = 0usize;
            for (i, flagged) in mask.iter().enumerate() {
                if *flagged { vals[i] = None; n_out += 1; }
            }

            // b. Impute every currently-missing cell that SAITS predicted.
            let mut n_fill = 0usize;
            if !map.is_empty() {
                let (pred_sorted, pred_tol) = sorted_preds(map);
                for (i, dt) in datetimes.iter().enumerate() {
                    let missing = match vals.get(i) {
                        Some(Some(x)) => !x.is_finite(),
                        Some(None) => true,
                        None => false,
                    };
                    if missing {
                        if let Some(v) = nearest_pred(&pred_sorted, dt.timestamp_millis(), pred_tol) {
                            vals[i] = Some(v);
                            n_fill += 1;
                        }
                    }
                }
            }

            let new_s = Series::new(col.as_str().into(), &vals);
            df.replace(col, new_s)
                .map_err(|e| format!("replace column {}: {}", col, e))?;

            total_out += n_out;
            total_fill += n_fill;
            per_column.push(serde_json::json!({
                "column": col, "n_outliers": n_out, "n_filled": n_fill,
            }));
        }

        write_cleaned_to_slot(&mut app_data, &dataset_key, df)?;
        app_data.cleaning_source_dataset = Some(dataset_key.clone());
        app_data.cleaning_method_label = Some("SAITS (détection+imputation)".to_string());
        app_data.cleaning_path = Some("saits".to_string());
        for col in &target_columns {
            if !app_data.cleaning_target_columns.iter().any(|x| x == col) {
                app_data.cleaning_target_columns.push(col.clone());
            }
        }
        logger::add_log(
            &mut app_data.logs,
            LogLevel::Success,
            format!(
                "Voie B SAITS : {} outliers retirés, {} points imputés sur {} colonne(s)",
                total_out, total_fill, target_columns.len(),
            ),
        );
        crate::utils::session_persist::save(&app_data);
    }

    Ok(serde_json::json!({
        "per_column":      per_column,
        "n_total_outliers": total_out,
        "n_total_filled":   total_fill,
        "model_id":        model_id,
        "gap_start":       gap_start,
        "gap_end":         gap_end,
        "dataset":         dataset_key,
    }))
}

/// Voie B — IA via SAITS, DETECT-ONLY: predict + find candidate points
/// (detected outliers and pre-existing gaps that SAITS can fill) WITHOUT
/// touching the dataset. The user validates/invalidates them in the UI, then
/// `cleaning_apply_cells` commits only the kept ones.
#[tauri::command]
pub async fn cleaning_detect_saits(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    state: State<'_, AppState>,
    model_id: String,
    files: Vec<String>,
    target_columns: Vec<String>,
    detect_method: String,
    detect_threshold: f64,
    dataset: Option<String>,
) -> Result<Value, String> {
    if target_columns.is_empty() {
        return Err("Aucune colonne cible sélectionnée.".to_string());
    }
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // 0. Dataset time range → SAITS gap window.
    let (gap_start, gap_end) = {
        let app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = dataset_ref(&app_data, &dataset_key)?;
        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let dts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
        let min = dts.iter().min().ok_or("dataset vide")?;
        let max = dts.iter().max().ok_or("dataset vide")?;
        (min.format("%Y-%m-%dT%H:%M:%S").to_string(),
         max.format("%Y-%m-%dT%H:%M:%S").to_string())
    };

    // Reconstruct from the LIVE in-memory dataset (re-exported to a temp parquet)
    // — no (re)import of the original xlsx is needed. Env data, if loaded, is
    // re-exported too so it overrides any stale model.meta.env_file. Falls back
    // to the caller-provided files only if the dataset can't be exported.
    let detect_files = match export_dataset_for_training(&state, &dataset_key) {
        Ok((path, _)) => vec![path],
        Err(_) => files.clone(),
    };
    let env_opt = export_env_for_training(&state).ok();

    // 1. Predict each column (async). Capture the model's typical reconstruction
    // error per column — the detector's threshold scale.
    let mut maps: Vec<(String, HashMap<i64, f64>, Option<f64>)> = Vec::with_capacity(target_columns.len());
    for col in &target_columns {
        // Detection: independent reconstruction (mask the whole column).
        let (m, recon) = saits_predict_map(&app, &sidecar, &model_id, &detect_files, col, &gap_start, &gap_end, false, env_opt.as_deref()).await?;
        maps.push((col.clone(), m, recon));
    }

    // 2. Build candidate lists (no mutation).
    let mut per_column: Vec<Value> = Vec::with_capacity(target_columns.len());
    {
        let app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = dataset_ref(&app_data, &dataset_key)?;
        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;

        for (col, map, recon) in &maps {
            let vals = col_as_f64_opts(df, col)?;
            let (pred_sorted, pred_tol) = sorted_preds(map);
            // Detection is driven BY THE MODEL: residual = raw − SAITS prediction
            // at each timestamp. A point is an outlier when that error far
            // exceeds the error the model NORMALLY makes (its per-column
            // reconstruction error, learned on good data) — NOT when it merely
            // lands in the tail of the residual distribution, which collapses on
            // near-constant channels and floods normal points as false
            // positives. Falls back to the residual MAD/z-score for models
            // trained before recon_error existed.
            let residuals: Vec<Option<f64>> = datetimes.iter().enumerate().map(|(i, dt)| {
                let ms = dt.timestamp_millis();
                match (vals[i], nearest_pred(&pred_sorted, ms, pred_tol)) {
                    (Some(raw), Some(pred)) if raw.is_finite() && pred.is_finite() => Some(raw - pred),
                    _ => None,
                }
            }).collect();
            let mask = match recon {
                Some(re) if *re > 0.0 => recon_outlier_mask(&residuals, &vals, *re, detect_threshold),
                _ => outlier_mask(&residuals, &detect_method, detect_threshold),
            };
            let mut candidates: Vec<Value> = Vec::new();
            for (i, dt) in datetimes.iter().enumerate() {
                // Voie B handles OUTLIERS only — pre-existing gaps (NaN) are left
                // for the Gap Filling "Compléter" step, so the two stages don't
                // overlap.
                if !mask[i] { continue; }
                let ms = dt.timestamp_millis();
                // Only actionable if SAITS proposes a finite replacement here.
                let Some(proposed) = nearest_pred(&pred_sorted, ms, pred_tol) else { continue };
                candidates.push(serde_json::json!({
                    "ts_millis": ms,
                    "kind": "outlier",
                    "raw": vals[i],
                    "proposed": proposed,
                }));
            }
            per_column.push(serde_json::json!({ "column": col, "candidates": candidates }));
        }
    }

    Ok(serde_json::json!({
        "per_column": per_column,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "dataset": dataset_key,
    }))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyCell {
    column: String,
    ts_millis: i64,
    value: Option<f64>,
}

/// Commit a set of validated cells into the cleaned dataset. Clones the SOURCE
/// dataset (so unvalidated outliers/gaps are left untouched), snapshots the
/// pre-state, sets each cell to its value, and writes the cleaned slot.
#[tauri::command]
pub async fn cleaning_apply_cells(
    state: State<'_, AppState>,
    dataset: Option<String>,
    cells: Vec<ApplyCell>,
) -> Result<Value, String> {
    use polars::prelude::*;

    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());
    if cells.is_empty() {
        return Err("Aucun point validé à appliquer.".to_string());
    }

    let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
    let mut df = dataset_ref(&app_data, &dataset_key)?.clone();
    app_data.cleaning_pre_snapshots
        .entry(dataset_key.clone())
        .or_insert_with(|| df.clone());

    let ts_col = df.column("TIMESTAMP")
        .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
    let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
        .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;

    // Group cells by column.
    let mut by_col: HashMap<String, HashMap<i64, Option<f64>>> = HashMap::new();
    for c in &cells {
        by_col.entry(c.column.clone()).or_default()
            .insert(c.ts_millis, c.value.filter(|x| x.is_finite()));
    }

    let mut total = 0usize;
    let mut cols_touched: Vec<String> = Vec::new();
    for (col, map) in &by_col {
        let mut vals = col_as_f64_opts(&df, col)?;
        for (i, dt) in datetimes.iter().enumerate() {
            if let Some(v) = map.get(&dt.timestamp_millis()) {
                vals[i] = *v;
                total += 1;
            }
        }
        df.replace(col, Series::new(col.as_str().into(), &vals))
            .map_err(|e| format!("replace column {}: {}", col, e))?;
        cols_touched.push(col.clone());
    }

    write_cleaned_to_slot(&mut app_data, &dataset_key, df)?;
    app_data.cleaning_source_dataset = Some(dataset_key.clone());
    app_data.cleaning_method_label = Some("SAITS (validé)".to_string());
    app_data.cleaning_path = Some("saits".to_string());
    for col in &cols_touched {
        if !app_data.cleaning_target_columns.iter().any(|x| x == col) {
            app_data.cleaning_target_columns.push(col.clone());
        }
    }
    logger::add_log(
        &mut app_data.logs,
        LogLevel::Success,
        format!("Voie B SAITS : {} point(s) validé(s) appliqué(s) sur {} colonne(s)", total, cols_touched.len()),
    );
    crate::utils::session_persist::save(&app_data);

    Ok(serde_json::json!({ "n_applied": total, "columns": cols_touched }))
}

/// Voie B → Gap Filling "Compléter": impute every REMAINING gap (NaN) in the
/// already-cleaned columns with SAITS, over the dataset's full range. Reads and
/// writes the CLEANED slot (continues from the Voie B result); falls back to
/// the source if nothing was cleaned yet. Imputation only — no outlier step.
#[tauri::command]
pub async fn cleaning_complete_saits(
    app: AppHandle,
    sidecar: State<'_, AiSidecar>,
    state: State<'_, AppState>,
    model_id: String,
    files: Vec<String>,
    target_columns: Vec<String>,
    dataset: Option<String>,
) -> Result<Value, String> {
    use polars::prelude::*;

    if target_columns.is_empty() {
        return Err("Aucune colonne à compléter.".to_string());
    }
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // Read the working df (cleaned slot if present, else the source).
    let read_working = |app_data: &crate::state::AppData| -> Result<DataFrame, String> {
        cleaned_slot_clone(app_data, &dataset_key)
            .or_else(|_| dataset_ref(app_data, &dataset_key).map(|d| d.clone()))
    };

    // 0. Time range.
    let (gap_start, gap_end) = {
        let app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = read_working(&app_data)?;
        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let dts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
        let min = dts.iter().min().ok_or("dataset vide")?;
        let max = dts.iter().max().ok_or("dataset vide")?;
        (min.format("%Y-%m-%dT%H:%M:%S").to_string(),
         max.format("%Y-%m-%dT%H:%M:%S").to_string())
    };

    // Reconstruct from the LIVE working data (re-exported) + fresh env snapshot —
    // no xlsx (re)import needed.
    let complete_files = {
        let df = {
            let app_data = state.inner.lock().map_err(|e| e.to_string())?;
            read_working(&app_data)?
        };
        export_df_temp(&df, "complete").map(|p| vec![p]).unwrap_or_else(|_| files.clone())
    };
    let env_opt = export_env_for_training(&state).ok();

    // 1. Predict each column.
    let mut maps: Vec<(String, HashMap<i64, f64>)> = Vec::with_capacity(target_columns.len());
    for col in &target_columns {
        // Gap filling: keep observed values, impute only the existing NaN so
        // fills use the sensor's own neighbouring context.
        let (m, _) = saits_predict_map(&app, &sidecar, &model_id, &complete_files, col, &gap_start, &gap_end, true, env_opt.as_deref()).await?;
        maps.push((col.clone(), m));
    }

    // 2. Fill remaining NaN — PREVIEW ONLY. Stash the filled frame as a pending
    //    completion; nothing is written to the dataset slot yet. The user then
    //    confirms (cleaning_commit_completion) or drops it (cleaning_discard_
    //    completion) from the Gap Filling page.
    let mut per_column: Vec<Value> = Vec::with_capacity(target_columns.len());
    let mut per_column_pairs: Vec<(String, usize)> = Vec::with_capacity(target_columns.len());
    let mut total = 0usize;
    let rows = {
        let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
        let mut df = read_working(&app_data)?;

        let ts_col = df.column("TIMESTAMP")
            .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
        let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
            .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;

        for (col, map) in &maps {
            let mut vals = col_as_f64_opts(&df, col)?;
            // Gaps (NaN) present in this column BEFORE filling — lets the UI
            // show "% of gaps filled" per sensor.
            let n_gaps = vals.iter().filter(|v| match v {
                Some(x) => !x.is_finite(),
                None => true,
            }).count();
            let mut n_fill = 0usize;
            if !map.is_empty() {
                let (pred_sorted, pred_tol) = sorted_preds(map);
                for (i, dt) in datetimes.iter().enumerate() {
                    let missing = match vals.get(i) {
                        Some(Some(x)) => !x.is_finite(),
                        Some(None) => true,
                        None => false,
                    };
                    if missing {
                        if let Some(v) = nearest_pred(&pred_sorted, dt.timestamp_millis(), pred_tol) {
                            vals[i] = Some(v);
                            n_fill += 1;
                        }
                    }
                }
            }
            df.replace(col, Series::new(col.as_str().into(), &vals))
                .map_err(|e| format!("replace column {}: {}", col, e))?;
            total += n_fill;
            per_column.push(serde_json::json!({ "column": col, "n_filled": n_fill, "n_gaps": n_gaps }));
            per_column_pairs.push((col.clone(), n_fill));
        }

        // Serialise the filled frame so the UI can show the "after" preview
        // WITHOUT it being committed. Downsample for transfer (annual files are
        // ~280k rows → would crash the WebView); the FULL frame is kept in
        // pending_completion for the commit.
        let rows = crate::utils::dataframe_serde::dataframe_page_to_json(
            &crate::commands::table::downsample_rows(&df, 8_000))
            .map_err(|e| e.to_string())?;
        app_data.pending_completion = Some(crate::state::PendingCompletion {
            dataset_key: dataset_key.clone(),
            df,
            per_column: per_column_pairs,
            n_total: total,
        });
        logger::add_log(
            &mut app_data.logs,
            LogLevel::Info,
            format!("Gap Filling SAITS : {} trou(s) comblé(s) — aperçu, en attente d'application", total),
        );
        rows
    };

    Ok(serde_json::json!({
        "per_column": per_column,
        "n_total_filled": total,
        "rows": rows,
        "pending": true,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "dataset": dataset_key,
    }))
}

/// Apply the pending SAITS completion to its dataset slot. Captures a one-shot
/// before-snapshot first (for the Voie B reset / before-after), then commits.
#[tauri::command]
pub async fn cleaning_commit_completion(state: State<'_, AppState>) -> Result<Value, String> {
    let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
    let pending = app_data.pending_completion.take()
        .ok_or("Aucune complétion en attente à appliquer.")?;
    let crate::state::PendingCompletion { dataset_key, df, per_column, n_total } = pending;

    // Snapshot the slot as it stands NOW (still the pre-fill state) so a later
    // reset can restore it, then overwrite with the completed frame.
    let before = cleaned_slot_clone(&app_data, &dataset_key)
        .or_else(|_| dataset_ref(&app_data, &dataset_key).map(|d| d.clone()))?;
    app_data.cleaning_pre_snapshots.entry(dataset_key.clone()).or_insert(before);
    write_cleaned_to_slot(&mut app_data, &dataset_key, df)?;
    app_data.cleaning_source_dataset = Some(dataset_key.clone());
    app_data.cleaning_method_label = Some("SAITS (complété)".to_string());
    app_data.cleaning_path = Some("saits".to_string());
    let cols: Vec<String> = per_column.iter().map(|(c, _)| c.clone()).collect();
    for c in &cols {
        if !app_data.cleaning_target_columns.iter().any(|x| x == c) {
            app_data.cleaning_target_columns.push(c.clone());
        }
    }
    logger::add_log(
        &mut app_data.logs,
        LogLevel::Success,
        format!("Gap Filling SAITS appliqué au fichier principal : {} trou(s) comblé(s) sur {} colonne(s)", n_total, cols.len()),
    );
    crate::utils::session_persist::save(&app_data);
    Ok(serde_json::json!({ "n_applied": n_total, "columns": cols }))
}

/// Drop the pending SAITS completion without touching the dataset.
#[tauri::command]
pub async fn cleaning_discard_completion(state: State<'_, AppState>) -> Result<Value, String> {
    let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
    let had = app_data.pending_completion.take().is_some();
    if had {
        logger::add_log(
            &mut app_data.logs,
            LogLevel::Info,
            "Gap Filling SAITS : complétion annulée (aperçu rejeté)".to_string(),
        );
    }
    Ok(serde_json::json!({ "discarded": had }))
}

// Local copies of cleaning_v2's dataset helpers. We avoid pulling them
// public to keep the cleaning module API surface tight.
fn dataset_ref<'a>(
    app: &'a crate::state::AppData,
    key: &str,
) -> Result<&'a polars::prelude::DataFrame, String> {
    let r = match key {
        "raw" => app.raw_data.as_ref(),
        "cleaned" => app.cleaned_data.as_ref(),
        "tslope"   => app.results.tslope.as_ref(),
        "baseline" => app.results.baseline.as_ref(),
        "delta_t"  => app.results.delta_t.as_ref(),
        "t600"     => app.results.t600.as_ref(),
        "tm"       => app.results.tm.as_ref(),
        "stm"      => app.results.stm.as_ref(),
        "tmi"      => app.results.tmi.as_ref(),
        "k"        => app.results.k.as_ref(),
        "sap_flow" => app.results.sap_flow.as_ref(),
        other => return Err(format!("dataset inconnu '{}'", other)),
    };
    r.ok_or_else(|| format!("dataset '{}' non chargé", key))
}

fn write_cleaned_to_slot(
    app: &mut crate::state::AppData,
    key: &str,
    df: polars::prelude::DataFrame,
) -> Result<(), String> {
    match key {
        "raw" | "cleaned" => { app.cleaned_data = Some(df); }
        "tslope"   => { app.results.tslope = Some(df); }
        "baseline" => { app.results.baseline = Some(df); }
        "delta_t"  => { app.results.delta_t = Some(df); }
        "t600"     => { app.results.t600 = Some(df); }
        "tm"       => { app.results.tm = Some(df); }
        "stm"      => { app.results.stm = Some(df); }
        "tmi"      => { app.results.tmi = Some(df); }
        "k"        => { app.results.k = Some(df); }
        "sap_flow" => { app.results.sap_flow = Some(df); }
        other => return Err(format!("dataset inconnu '{}'", other)),
    }
    Ok(())
}

/// Clone the dataframe currently sitting in the CLEANED slot for `key` (the
/// destination `write_cleaned_to_slot` targets), so a follow-up edit can read
/// the post-cleaning state rather than the original source.
fn cleaned_slot_clone(
    app: &crate::state::AppData,
    key: &str,
) -> Result<polars::prelude::DataFrame, String> {
    let r = match key {
        "raw" | "cleaned" => app.cleaned_data.as_ref(),
        "tslope"   => app.results.tslope.as_ref(),
        "baseline" => app.results.baseline.as_ref(),
        "delta_t"  => app.results.delta_t.as_ref(),
        "t600"     => app.results.t600.as_ref(),
        "tm"       => app.results.tm.as_ref(),
        "stm"      => app.results.stm.as_ref(),
        "tmi"      => app.results.tmi.as_ref(),
        "k"        => app.results.k.as_ref(),
        "sap_flow" => app.results.sap_flow.as_ref(),
        other => return Err(format!("dataset inconnu '{}'", other)),
    };
    r.cloned().ok_or_else(|| format!("dataset nettoyé '{}' non chargé", key))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RestoreCell {
    ts_millis: i64,
    value: Option<f64>,
}

/// Restore specific cells of a cleaned column to a caller-supplied value — used
/// by the Voie B SAITS chart to let the user "invalidate" a flagged point
/// (click a red dot → put back the original raw value, or null to undo an
/// imputed gap). Writes into the cleaned slot so the change persists.
#[tauri::command]
pub async fn cleaning_restore_cells(
    state: State<'_, AppState>,
    dataset: Option<String>,
    column: String,
    cells: Vec<RestoreCell>,
) -> Result<Value, String> {
    use polars::prelude::*;

    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());
    if cells.is_empty() {
        return Ok(serde_json::json!({ "restored": 0, "column": column }));
    }

    let mut app_data = state.inner.lock().map_err(|e| e.to_string())?;
    let mut df = cleaned_slot_clone(&app_data, &dataset_key)?;

    let ts_col = df.column("TIMESTAMP")
        .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
    let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
        .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
    let mut vals = col_as_f64_opts(&df, &column)?;

    let mut want: HashMap<i64, Option<f64>> = HashMap::with_capacity(cells.len());
    for c in &cells {
        want.insert(c.ts_millis, c.value.filter(|x| x.is_finite()));
    }

    let mut restored = 0usize;
    for (i, dt) in datetimes.iter().enumerate() {
        if let Some(v) = want.get(&dt.timestamp_millis()) {
            vals[i] = *v;
            restored += 1;
        }
    }

    let new_s = Series::new(column.as_str().into(), &vals);
    df.replace(&column, new_s)
        .map_err(|e| format!("replace column {}: {}", column, e))?;
    write_cleaned_to_slot(&mut app_data, &dataset_key, df)?;
    logger::add_log(
        &mut app_data.logs,
        LogLevel::Info,
        format!("Voie B SAITS : {} point(s) restauré(s) sur '{}'", restored, column),
    );
    crate::utils::session_persist::save(&app_data);

    Ok(serde_json::json!({ "restored": restored, "column": column }))
}
