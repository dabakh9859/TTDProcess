use tauri::State;
use uuid::Uuid;

use crate::core::types::{LogLevel, ModelType, TrainedModelInfo};
use crate::ml::engine;
use crate::state::AppState;
use crate::utils::logger;

// =============================================================================
// train_model
// =============================================================================

#[tauri::command]
pub async fn train_model(
    state: State<'_, AppState>,
    config: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("LinearRegression")
        .to_string();

    let preset = config
        .get("preset")
        .and_then(|v| v.as_str())
        .unwrap_or("Rapide")
        .to_string();

    let target_column = config
        .get("target_column")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let test_split = config
        .get("test_split")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2)
        .clamp(0.05, 0.5);

    if target_column.is_empty() {
        return Err("target_column is required".to_string());
    }

    // Extract target column values + timestamps from a brief lock
    let (values, timestamps) = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let df = data
            .cleaned_data
            .as_ref()
            .or(data.raw_data.as_ref())
            .ok_or_else(|| "No data loaded. Import data first.".to_string())?;

        let col = df.column(&target_column).map_err(|e| e.to_string())?;
        let vals: Vec<f64> = col
            .f64()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        if vals.len() < 20 {
            return Err("Not enough data points to train (minimum 20 required)".to_string());
        }

        let ts: Vec<String> = if let Ok(ts_col) = df.column("TIMESTAMP") {
            ts_col
                .cast(&polars::prelude::DataType::String)
                .unwrap_or_else(|_| ts_col.clone())
                .str()
                .map(|ca| ca.into_iter().map(|v| v.unwrap_or("").to_string()).collect())
                .unwrap_or_else(|_| vec![String::new(); vals.len()])
        } else {
            vec![String::new(); vals.len()]
        };
        (vals, ts)
    };

    // Heavy training on blocking thread
    let model_type_clone = model_type.clone();
    let preset_clone = preset.clone();
    let start = std::time::Instant::now();
    let cache = tokio::task::spawn_blocking(move || {
        engine::train(&model_type_clone, &values, &preset_clone, test_split, &timestamps)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    // Re-acquire lock and store results
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;

    let model_id = Uuid::new_v4().to_string();

    let model_type_enum = match model_type.as_str() {
        "RandomForest" => ModelType::RandomForest,
        "LSTM" => ModelType::LSTM,
        "GradientBoosting" => ModelType::GradientBoosting,
        "Prophet" => ModelType::Prophet,
        _ => ModelType::LinearRegression,
    };

    let info = TrainedModelInfo {
        model_id: model_id.clone(),
        model_type: model_type_enum,
        target_column: target_column.clone(),
        metrics: cache.metrics.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    data.trained_models.insert(model_id.clone(), info);
    data.model_cache.insert(model_id.clone(), cache.clone());

    logger::add_log(
        &mut data.logs,
        LogLevel::Success,
        format!(
            "Model {} ({}) trained in {}ms – RMSE: {:.4}, R²: {:.4}",
            model_id, model_type, elapsed_ms, cache.metrics.rmse, cache.metrics.r2
        ),
    );

    Ok(serde_json::json!({
        "model_id": model_id,
        "model_type": model_type,
        "target_column": target_column,
        "training_time_ms": elapsed_ms,
        "metrics": {
            "mse":  cache.metrics.mse,
            "rmse": cache.metrics.rmse,
            "mae":  cache.metrics.mae,
            "r2":   cache.metrics.r2,
            "mape": cache.metrics.mape,
        },
    }))
}

// =============================================================================
// predict
// =============================================================================

#[tauri::command]
pub fn predict(
    state: State<'_, AppState>,
    model_id: String,
    start_idx: usize,
    end_idx: usize,
) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;

    let cache = data
        .model_cache
        .get(&model_id)
        .ok_or_else(|| format!("Model '{}' not found", model_id))?;

    let n = cache.predictions.len();
    let start = start_idx.min(n);
    let end = end_idx.min(n);

    let preds = &cache.predictions[start..end];
    let actuals = &cache.actuals[start..end];
    let ts = &cache.timestamps[start.min(cache.timestamps.len())..end.min(cache.timestamps.len())];

    Ok(serde_json::json!({
        "model_id": model_id,
        "predictions": preds,
        "actual": actuals,
        "timestamps": ts,
        "test_start": cache.test_start,
    }))
}

// =============================================================================
// list_models
// =============================================================================

#[tauri::command]
pub fn list_models(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;

    let models: Vec<serde_json::Value> = data
        .trained_models
        .values()
        .map(|info| {
            serde_json::json!({
                "model_id": info.model_id,
                "model_type": format!("{:?}", info.model_type),
                "target_column": info.target_column,
                "created_at": info.created_at,
                "metrics": {
                    "mse":  info.metrics.mse,
                    "rmse": info.metrics.rmse,
                    "mae":  info.metrics.mae,
                    "r2":   info.metrics.r2,
                    "mape": info.metrics.mape,
                },
            })
        })
        .collect();

    Ok(serde_json::json!({ "models": models }))
}

// =============================================================================
// save_model / load_model (future work — minimal stub)
// =============================================================================

#[tauri::command]
pub fn save_model(
    state: State<'_, AppState>,
    model_id: String,
    path: String,
) -> Result<String, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;

    let cache = data
        .model_cache
        .get(&model_id)
        .ok_or_else(|| format!("Model '{}' not found", model_id))?;

    let info = data
        .trained_models
        .get(&model_id)
        .ok_or_else(|| format!("Model info '{}' not found", model_id))?;

    let payload = serde_json::json!({ "info": info, "cache": cache });
    let json = serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())?;

    Ok(path)
}

#[tauri::command]
pub fn load_model(
    state: State<'_, AppState>,
    path: String,
) -> Result<String, String> {
    let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let payload: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;

    let info: TrainedModelInfo = serde_json::from_value(
        payload["info"].clone(),
    )
    .map_err(|e| e.to_string())?;

    let cache: crate::ml::engine::ModelCache = serde_json::from_value(
        payload["cache"].clone(),
    )
    .map_err(|e| e.to_string())?;

    let model_id = info.model_id.clone();

    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    data.trained_models.insert(model_id.clone(), info);
    data.model_cache.insert(model_id.clone(), cache);

    Ok(model_id)
}
