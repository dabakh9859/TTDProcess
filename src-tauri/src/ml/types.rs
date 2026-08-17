use serde::{Deserialize, Serialize};

/// Configuration for ML model training.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainRequest {
    pub model_type: String,
    pub preset: String,
    pub input_columns: Vec<String>,
    pub target_column: String,
    pub train_start_idx: usize,
    pub train_end_idx: usize,
    pub test_split: f64,
    pub custom_params: Option<serde_json::Value>,
}

/// Result returned after training.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainResult {
    pub model_id: String,
    pub model_type: String,
    pub target_column: String,
    pub metrics: MLMetrics,
    pub training_time_ms: u64,
}

/// Metrics from model training / evaluation.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub mape: f64,
}

/// Prediction request.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictRequest {
    pub model_id: String,
    pub start_idx: usize,
    pub end_idx: usize,
}

/// Prediction result.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictResult {
    pub model_id: String,
    pub predictions: Vec<f64>,
    pub actual: Vec<f64>,
}
