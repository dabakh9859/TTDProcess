use std::collections::HashMap;
use tauri::State;

use crate::core::data_cleaning;
use crate::core::types::{DetectionState, LogLevel};
use crate::ml::engine;
use crate::state::AppState;
use crate::utils::logger;

/// Flag points whose prediction residual is farther than `threshold` σ from the
/// mean residual. Used by the "LSTM" (MLP-based) anomaly detector.
fn residual_outliers(
    actuals: &[f64],
    predictions: &[f64],
    threshold: f64,
    offset: usize,
    total_len: usize,
) -> Vec<bool> {
    let mut mask = vec![false; total_len];
    if actuals.is_empty() || actuals.len() != predictions.len() {
        return mask;
    }
    let residuals: Vec<f64> = actuals
        .iter()
        .zip(predictions.iter())
        .map(|(a, p)| a - p)
        .collect();
    let n = residuals.len() as f64;
    let mean = residuals.iter().sum::<f64>() / n;
    let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-12 {
        return mask;
    }
    for (i, &r) in residuals.iter().enumerate() {
        if ((r - mean) / std).abs() > threshold {
            let idx = i + offset;
            if idx < total_len {
                mask[idx] = true;
            }
        }
    }
    mask
}

#[tauri::command]
pub async fn detect_outliers(
    state: State<'_, AppState>,
    dataset: Option<String>,
    columns: Vec<String>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    // Clone the DataFrame out of the lock
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());
    let df = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let df_ref = match dataset_key.as_str() {
            "raw" => app.raw_data.as_ref(),
            "cleaned" => app.cleaned_data.as_ref(),
            "sap_flow" => app.results.sap_flow.as_ref(),
            other => return Err(format!("Unknown dataset '{}'", other)),
        };
        df_ref
            .ok_or_else(|| format!("Dataset '{}' is not loaded", dataset_key))?
            .clone()
    };

    let threshold = params
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.5);
    let window = params
        .get("window")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as usize;
    let contamination = params
        .get("contamination")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.05);

    let columns_clone = columns.clone();
    let method_clone = method.clone();

    // Heavy work on blocking thread
    let outlier_indices: HashMap<String, Vec<usize>> = tokio::task::spawn_blocking(move || {
        let method_lower = method_clone.to_lowercase();
        let mut indices: HashMap<String, Vec<usize>> = HashMap::new();

        for col_name in &columns_clone {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let cast = match col.cast(&polars::prelude::DataType::Float64) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let values = match cast.f64() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let data: Vec<f64> = values.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();

            let mask = match method_lower.as_str() {
                "iqr" => data_cleaning::detect_outliers_iqr(&data, threshold),
                "zscore" => data_cleaning::detect_outliers_zscore(&data, threshold),
                "mad" => data_cleaning::detect_outliers_mad(&data, threshold),
                "isolationforest" | "isolation_forest" => {
                    data_cleaning::detect_outliers_isolation_forest(&data, contamination)
                }
                "rollingzscore" | "rolling_zscore" => {
                    data_cleaning::detect_outliers_rolling_zscore(&data, window, threshold)
                }
                "lstm" | "mlp" => {
                    // Sliding-window MLP trained on the whole column, residuals
                    // (actual - predicted) are z-scored; points beyond `threshold`
                    // σ are flagged as anomalies. Uses "Rapide" preset for speed.
                    let finite_count = data.iter().filter(|v| v.is_finite()).count();
                    if finite_count < 40 {
                        // Not enough usable points — skip this column silently
                        continue;
                    }
                    // Replace NaN/Inf with column mean so training is stable
                    let finite_mean: f64 = data
                        .iter()
                        .filter(|v| v.is_finite())
                        .sum::<f64>()
                        / finite_count as f64;
                    let clean: Vec<f64> = data
                        .iter()
                        .map(|v| if v.is_finite() { *v } else { finite_mean })
                        .collect();
                    let ts_dummy: Vec<String> = Vec::new();
                    let cache = match engine::train_mlp(&clean, "Rapide", 0.2, &ts_dummy) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    // ModelCache covers indices [window_size..data.len()], so the
                    // first `window_size` points can never be flagged.
                    let offset = clean.len().saturating_sub(cache.actuals.len());
                    residual_outliers(
                        &cache.actuals,
                        &cache.predictions,
                        threshold,
                        offset,
                        data.len(),
                    )
                }
                _ => continue,
            };

            let idxs: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &is_o)| if is_o { Some(i) } else { None })
                .collect();
            indices.insert(col_name.clone(), idxs);
        }
        indices
    })
    .await
    .map_err(|e| e.to_string())?;

    let total_outliers: usize = outlier_indices.values().map(|v| v.len()).sum();
    let col_summary: HashMap<String, usize> = outlier_indices
        .iter()
        .map(|(k, v)| (k.clone(), v.len()))
        .collect();

    // Store back in state (brief lock)
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        app.detection_results = Some(DetectionState {
            method: method.clone(),
            dataset: dataset_key.clone(),
            outlier_indices,
            validated: false,
        });
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            format!(
                "Detected {} outliers across {} columns using {}",
                total_outliers,
                columns.len(),
                method
            ),
        );
    }

    Ok(serde_json::json!({
        "method": method,
        "total_outliers": total_outliers,
        "columns": col_summary,
    }))
}

#[tauri::command]
pub async fn fill_gaps(
    state: State<'_, AppState>,
    columns: Vec<String>,
    method: String,
    window_size: Option<usize>,
) -> Result<serde_json::Value, String> {
    // Clone data out of lock — use the same dataset that was detected on
    let (df, detection) = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let det = app
            .detection_results
            .as_ref()
            .ok_or_else(|| "No detection results. Run detect_outliers first.".to_string())?
            .clone();
        let df_ref = match det.dataset.as_str() {
            "raw" => app.raw_data.as_ref(),
            "cleaned" => app.cleaned_data.as_ref(),
            "sap_flow" => app.results.sap_flow.as_ref(),
            other => return Err(format!("Unknown dataset '{}' in detection state", other)),
        };
        let d = df_ref
            .ok_or_else(|| format!("Dataset '{}' is not loaded", det.dataset))?
            .clone();
        (d, det)
    };

    let columns_clone = columns.clone();
    let method_clone = method.clone();
    let win = window_size.unwrap_or(6);

    // Heavy work on blocking thread
    let new_df = tokio::task::spawn_blocking(move || -> Result<polars::prelude::DataFrame, String> {
        let mut new_df = df.clone();
        let method_lower = method_clone.to_lowercase();

        for col_name in &columns_clone {
            let col = df.column(col_name).map_err(|e| e.to_string())?;
            let cast = col.cast(&polars::prelude::DataType::Float64).map_err(|e| e.to_string())?;
            let values = cast.f64().map_err(|e| e.to_string())?;
            let data: Vec<f64> = values.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();

            let outlier_idx = detection.outlier_indices.get(col_name).cloned().unwrap_or_default();
            let mask: Vec<bool> = (0..data.len())
                .map(|i| outlier_idx.contains(&i) || data[i].is_nan())
                .collect();

            let filled = match method_lower.as_str() {
                "linear" => data_cleaning::fill_gaps_linear(&data, &mask),
                "movingaverage" | "moving_average" => {
                    data_cleaning::fill_gaps_moving_average(&data, &mask, win)
                        .map_err(|e| e.to_string())?
                }
                "mlmodel" | "ml_model" | "mlp" | "lstm" => {
                    // 1) Seed the series with a linear fill so the MLP sees no NaN.
                    // 2) Train a small sliding-window MLP on the seeded series.
                    // 3) Replace each masked position with the MLP prediction
                    //    (falling back to the seeded value for points earlier than
                    //    the window, which the model can't predict).
                    let seeded = data_cleaning::fill_gaps_linear(&data, &mask);
                    let finite_count = seeded.iter().filter(|v| v.is_finite()).count();
                    if finite_count < 40 {
                        seeded
                    } else {
                        let ts_dummy: Vec<String> = Vec::new();
                        match engine::train_mlp(&seeded, "Rapide", 0.2, &ts_dummy) {
                            Ok(cache) => {
                                let offset = seeded.len().saturating_sub(cache.predictions.len());
                                let mut out = seeded.clone();
                                for (i, &is_masked) in mask.iter().enumerate() {
                                    if !is_masked { continue; }
                                    if i >= offset {
                                        let p = cache.predictions[i - offset];
                                        if p.is_finite() {
                                            out[i] = p;
                                        }
                                    }
                                    // else: keep the linear seed value
                                }
                                out
                            }
                            Err(_) => seeded, // training failure → linear seed
                        }
                    }
                }
                _ => data_cleaning::fill_gaps_linear(&data, &mask),
            };

            use polars::prelude::NamedFrom;
            let new_series = polars::prelude::Series::new(col_name.as_str().into(), filled);
            let _ = new_df.replace(col_name, new_series);
        }
        Ok(new_df)
    })
    .await
    .map_err(|e| e.to_string())??;

    let filled_count: usize = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        if let Some(det) = &app.detection_results {
            columns
                .iter()
                .filter_map(|c| det.outlier_indices.get(c))
                .map(|v| v.len())
                .sum()
        } else {
            0
        }
    };

    // Store cleaned data back
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        app.cleaned_data = Some(new_df);
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!("Filled {} values using {} in {} columns", filled_count, method, columns.len()),
        );
    }

    Ok(serde_json::json!({
        "method": method,
        "filled_count": filled_count,
        "columns": columns,
    }))
}

#[tauri::command]
pub fn validate_detection(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;

    let detection = app
        .detection_results
        .as_mut()
        .ok_or_else(|| "No detection results to validate".to_string())?;

    detection.validated = true;

    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        "Detection results validated".to_string(),
    );

    Ok(serde_json::json!({ "validated": true }))
}
