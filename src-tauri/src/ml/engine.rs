//! ML engine: Linear Regression, Random Forest, MLP (LSTM-like), Prophet-like
//!
//! All models use a **sliding-window** approach:
//!   features  = [x_{t-W}, ..., x_{t-1}]  (window_size = W)
//!   target    = x_t
//!
//! After training the full prediction vector is cached so the frontend can
//! draw real-vs-predicted charts without re-running inference.

use ndarray::{Array1, Array2};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::types::TrainingMetrics;

// =============================================================================
// Types stored per trained model
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCache {
    /// Predictions on the full input vector (including train + test)
    pub predictions: Vec<f64>,
    /// Corresponding actuals
    pub actuals: Vec<f64>,
    /// ISO-8601 timestamp strings (empty if not available)
    pub timestamps: Vec<String>,
    /// Metrics computed on the test split
    pub metrics: TrainingMetrics,
    /// Index at which the test split starts (first test sample)
    pub test_start: usize,
}

// =============================================================================
// Preset → hyperparameters
// =============================================================================

pub struct Hyperparams {
    pub window_size: usize,
    pub n_estimators: usize,  // RF / GB trees
    pub epochs: usize,        // MLP
    pub learning_rate: f64,   // MLP
    pub hidden_size: usize,   // MLP hidden layer
}

pub fn preset_params(preset: &str) -> Hyperparams {
    match preset {
        "Precis" => Hyperparams {
            window_size: 12,
            n_estimators: 500,
            epochs: 500,
            learning_rate: 0.001,
            hidden_size: 128,
        },
        "Standard" => Hyperparams {
            window_size: 8,
            n_estimators: 200,
            epochs: 200,
            learning_rate: 0.005,
            hidden_size: 64,
        },
        _ => Hyperparams {
            // Rapide (default)
            window_size: 6,
            n_estimators: 50,
            epochs: 50,
            learning_rate: 0.01,
            hidden_size: 32,
        },
    }
}

// =============================================================================
// Dataset building
// =============================================================================

/// Build (X, y) from a flat time series with a sliding window.
fn build_xy(data: &[f64], window_size: usize) -> (Array2<f64>, Array1<f64>) {
    let n = data.len();
    assert!(n > window_size, "Data too short for window size {}", window_size);
    let n_samples = n - window_size;
    let mut x = Array2::<f64>::zeros((n_samples, window_size));
    let mut y = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        for j in 0..window_size {
            x[[i, j]] = data[i + j];
        }
        y[i] = data[i + window_size];
    }
    (x, y)
}

/// Min-max normalize data and return (normalized, min, max).
fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-12 { 1.0 } else { max - min };
    let normalized = data.iter().map(|v| (v - min) / range).collect();
    (normalized, min, max)
}

fn denormalize(v: f64, min: f64, max: f64) -> f64 {
    v * (max - min) + min
}

// =============================================================================
// Metrics
// =============================================================================

fn compute_metrics(y_true: &[f64], y_pred: &[f64]) -> TrainingMetrics {
    let n = y_true.len() as f64;
    if n == 0.0 {
        return TrainingMetrics { mse: 0.0, rmse: 0.0, mae: 0.0, r2: 0.0, mape: 0.0 };
    }

    let mse = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n;

    let rmse = mse.sqrt();

    let mae = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n;

    let mean_y = y_true.iter().sum::<f64>() / n;
    let ss_res: f64 = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean_y).powi(2)).sum();
    let r2 = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 1.0 };

    let non_zero: Vec<(f64, f64)> = y_true
        .iter()
        .zip(y_pred)
        .filter(|(t, _)| t.abs() > 1e-12)
        .map(|(&t, &p)| (t, p))
        .collect();
    let mape = if non_zero.is_empty() {
        0.0
    } else {
        non_zero
            .iter()
            .map(|(t, p)| ((t - p) / t).abs() * 100.0)
            .sum::<f64>()
            / non_zero.len() as f64
    };

    TrainingMetrics { mse, rmse, mae, r2, mape }
}

// =============================================================================
// Linear Regression (via linfa-linear)
// =============================================================================

pub fn train_linear_regression(
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    use linfa::prelude::*;
    use linfa_linear::LinearRegression;

    let hp = preset_params(preset);
    let (norm, min_v, max_v) = normalize(data);
    let (x, y) = build_xy(&norm, hp.window_size);

    let n = x.nrows();
    let n_train = ((n as f64) * (1.0 - test_split)).ceil() as usize;
    let n_train = n_train.max(1).min(n - 1);

    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..n_train]).to_owned();

    let dataset = Dataset::new(x_train, y_train);
    let model = LinearRegression::new()
        .fit(&dataset)
        .context("Linear regression fit failed")?;

    // Manual prediction: y = X @ w + intercept
    let weights = model.params(); // Array1<f64>
    let intercept = model.intercept();
    let predictions: Vec<f64> = (0..n)
        .map(|i| {
            let pred = weights
                .iter()
                .enumerate()
                .map(|(j, w)| w * x[[i, j]])
                .sum::<f64>()
                + intercept;
            denormalize(pred, min_v, max_v)
        })
        .collect();
    let actuals: Vec<f64> = y.iter().map(|v| denormalize(*v, min_v, max_v)).collect();

    let test_preds = &predictions[n_train..];
    let test_actuals = &actuals[n_train..];
    let metrics = compute_metrics(test_actuals, test_preds);

    let ts_offset = hp.window_size.min(timestamps.len());
    let timestamps_out = timestamps[ts_offset..].to_vec();

    Ok(ModelCache {
        predictions,
        actuals,
        timestamps: timestamps_out,
        metrics,
        test_start: n_train,
    })
}

// =============================================================================
// Random Forest (manual ensemble of regression stumps via ndarray)
//
// Strategy: bootstrap N sub-samples, fit a simple regression stump on each
// (split on median of random feature), then average predictions.
// This gives a genuine variance-reducing ensemble without external crate issues.
// =============================================================================

/// A single-split regression stump.
struct Stump {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl Stump {
    fn fit(x: &Array2<f64>, y: &[f64]) -> Option<Self> {
        use rand::Rng;
        let n = x.nrows();
        let n_features = x.ncols();
        if n < 2 || n_features == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut best: Option<(f64, usize, f64)> = None; // (mse_gain, feat, threshold)

        // Try a random subset of features (sqrt(n_features) features)
        let n_try = ((n_features as f64).sqrt().ceil() as usize).max(1);
        let feats: Vec<usize> = {
            let mut f: Vec<usize> = (0..n_features).collect();
            for i in (1..n_features).rev() {
                let j = rng.gen_range(0..=i);
                f.swap(i, j);
            }
            f[..n_try].to_vec()
        };

        let y_mean = y.iter().sum::<f64>() / n as f64;
        let total_mse: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;

        for feat in feats {
            let feat_vals: Vec<f64> = (0..n).map(|i| x[[i, feat]]).collect();
            let threshold = {
                let mut sorted = feat_vals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };

            let left: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|(i, _)| feat_vals[*i] <= threshold)
                .map(|(_, v)| *v)
                .collect();
            let right: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|(i, _)| feat_vals[*i] > threshold)
                .map(|(_, v)| *v)
                .collect();

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let lm = left.iter().sum::<f64>() / left.len() as f64;
            let rm = right.iter().sum::<f64>() / right.len() as f64;
            let mse_split = (left.iter().map(|v| (v - lm).powi(2)).sum::<f64>()
                + right.iter().map(|v| (v - rm).powi(2)).sum::<f64>())
                / n as f64;

            let gain = total_mse - mse_split;
            if best.as_ref().map(|(g, _, _)| gain > *g).unwrap_or(true) {
                best = Some((gain, feat, threshold));
            }
        }

        best.map(|(_, feat, threshold)| {
            let feat_vals: Vec<f64> = (0..n).map(|i| x[[i, feat]]).collect();
            let left: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|(i, _)| feat_vals[*i] <= threshold)
                .map(|(_, v)| *v)
                .collect();
            let right: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|(i, _)| feat_vals[*i] > threshold)
                .map(|(_, v)| *v)
                .collect();
            let left_value = left.iter().sum::<f64>() / left.len().max(1) as f64;
            let right_value = right.iter().sum::<f64>() / right.len().max(1) as f64;
            Stump { feature: feat, threshold, left_value, right_value }
        })
    }

    fn predict_one(&self, row: &[f64]) -> f64 {
        if row[self.feature] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }
}

pub fn train_random_forest(
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    use rand::Rng;

    let hp = preset_params(preset);
    let (norm, min_v, max_v) = normalize(data);
    let (x, y) = build_xy(&norm, hp.window_size);

    let n = x.nrows();
    let n_train = ((n as f64) * (1.0 - test_split)).ceil() as usize;
    let n_train = n_train.max(1).min(n - 1);

    let n_trees = hp.n_estimators.min(200); // cap for performance
    let mut stumps: Vec<Stump> = Vec::new();
    let mut rng = rand::thread_rng();

    let y_train: Vec<f64> = (0..n_train).map(|i| y[i]).collect();

    for _ in 0..n_trees {
        // Bootstrap sample
        let boot_idx: Vec<usize> = (0..n_train).map(|_| rng.gen_range(0..n_train)).collect();
        let x_boot = Array2::from_shape_fn((n_train, hp.window_size), |(i, j)| {
            x[[boot_idx[i], j]]
        });
        let y_boot: Vec<f64> = boot_idx.iter().map(|&i| y_train[i]).collect();

        if let Some(stump) = Stump::fit(&x_boot, &y_boot) {
            stumps.push(stump);
        }
    }

    if stumps.is_empty() {
        // Fallback: constant prediction = mean
        let mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
        let predictions = vec![denormalize(mean, min_v, max_v); n];
        let actuals: Vec<f64> = y.iter().map(|v| denormalize(*v, min_v, max_v)).collect();
        let metrics = compute_metrics(&actuals[n_train..], &predictions[n_train..]);
        return Ok(ModelCache {
            predictions,
            actuals,
            timestamps: timestamps[hp.window_size.min(timestamps.len())..].to_vec(),
            metrics,
            test_start: n_train,
        });
    }

    // Aggregate predictions (ensemble average)
    let predictions: Vec<f64> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..hp.window_size).map(|j| x[[i, j]]).collect();
            let avg = stumps.iter().map(|s| s.predict_one(&row)).sum::<f64>()
                / stumps.len() as f64;
            denormalize(avg, min_v, max_v)
        })
        .collect();
    let actuals: Vec<f64> = y.iter().map(|v| denormalize(*v, min_v, max_v)).collect();

    let metrics = compute_metrics(&actuals[n_train..], &predictions[n_train..]);

    let ts_offset = hp.window_size.min(timestamps.len());
    let timestamps_out = timestamps[ts_offset..].to_vec();

    Ok(ModelCache {
        predictions,
        actuals,
        timestamps: timestamps_out,
        metrics,
        test_start: n_train,
    })
}

// =============================================================================
// MLP (used as LSTM / Gradient Boosting backends)
//
// Architecture: input(W) → hidden(H, ReLU) → output(1, linear)
// Training:     stochastic gradient descent, mini-batch size 32
// =============================================================================

fn relu(x: f64) -> f64 {
    x.max(0.0)
}
fn relu_d(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn train_mlp(
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    use rand::Rng;

    let hp = preset_params(preset);
    let (norm, min_v, max_v) = normalize(data);
    let (x, y) = build_xy(&norm, hp.window_size);

    let n = x.nrows();
    let n_train = ((n as f64) * (1.0 - test_split)).ceil() as usize;
    let n_train = n_train.max(1).min(n - 1);

    let input_size = hp.window_size;
    let hidden_size = hp.hidden_size;
    let lr = hp.learning_rate;
    let epochs = hp.epochs;
    let batch_size = 32_usize;

    let mut rng = rand::thread_rng();

    // Xavier initialization
    let scale1 = (2.0_f64 / input_size as f64).sqrt();
    let scale2 = (2.0_f64 / hidden_size as f64).sqrt();

    let mut w1: Vec<Vec<f64>> = (0..hidden_size)
        .map(|_| (0..input_size).map(|_| rng.gen::<f64>() * 2.0 * scale1 - scale1).collect())
        .collect();
    let mut b1: Vec<f64> = vec![0.0; hidden_size];
    let mut w2: Vec<f64> = (0..hidden_size)
        .map(|_| rng.gen::<f64>() * 2.0 * scale2 - scale2)
        .collect();
    let mut b2: f64 = 0.0;

    // Forward pass for a single sample
    let forward = |x_row: &[f64],
                   w1: &Vec<Vec<f64>>,
                   b1: &Vec<f64>,
                   w2: &Vec<f64>,
                   b2: f64|
     -> (Vec<f64>, Vec<f64>, f64) {
        let z1: Vec<f64> = (0..hidden_size)
            .map(|h| {
                b1[h] + w1[h].iter().zip(x_row).map(|(w, x)| w * x).sum::<f64>()
            })
            .collect();
        let a1: Vec<f64> = z1.iter().map(|&z| relu(z)).collect();
        let out = b2 + w2.iter().zip(&a1).map(|(w, a)| w * a).sum::<f64>();
        (z1, a1, out)
    };

    // Training loop
    let x_train: Vec<Vec<f64>> = (0..n_train)
        .map(|i| (0..input_size).map(|j| x[[i, j]]).collect())
        .collect();
    let y_train: Vec<f64> = (0..n_train).map(|i| y[i]).collect();

    for _epoch in 0..epochs {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_train).collect();
        for i in (1..n_train).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for batch_start in (0..n_train).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_train);
            let batch_size_actual = batch_end - batch_start;

            let mut dw1 = vec![vec![0.0_f64; input_size]; hidden_size];
            let mut db1 = vec![0.0_f64; hidden_size];
            let mut dw2 = vec![0.0_f64; hidden_size];
            let mut db2 = 0.0_f64;

            for &idx in &indices[batch_start..batch_end] {
                let x_row = &x_train[idx];
                let y_true = y_train[idx];
                let (z1, a1, out) = forward(x_row, &w1, &b1, &w2, b2);

                let err = out - y_true; // MSE gradient
                let d_out = 2.0 * err / batch_size_actual as f64;

                db2 += d_out;
                for h in 0..hidden_size {
                    dw2[h] += d_out * a1[h];
                    let d_h = d_out * w2[h] * relu_d(z1[h]);
                    db1[h] += d_h;
                    for j in 0..input_size {
                        dw1[h][j] += d_h * x_row[j];
                    }
                }
            }

            // Apply gradients
            b2 -= lr * db2;
            for h in 0..hidden_size {
                w2[h] -= lr * dw2[h];
                b1[h] -= lr * db1[h];
                for j in 0..input_size {
                    w1[h][j] -= lr * dw1[h][j];
                }
            }
        }
    }

    // Predict on full dataset
    let predictions: Vec<f64> = (0..n)
        .map(|i| {
            let x_row: Vec<f64> = (0..input_size).map(|j| x[[i, j]]).collect();
            let (_, _, out) = forward(&x_row, &w1, &b1, &w2, b2);
            denormalize(out.clamp(0.0, 1.0), min_v, max_v)
        })
        .collect();
    let actuals: Vec<f64> = y.iter().map(|v| denormalize(*v, min_v, max_v)).collect();

    let test_preds = &predictions[n_train..];
    let test_actuals = &actuals[n_train..];
    let metrics = compute_metrics(test_actuals, test_preds);

    let ts_offset = hp.window_size;
    let timestamps_out = if timestamps.len() > ts_offset {
        timestamps[ts_offset..].to_vec()
    } else {
        vec![String::new(); predictions.len()]
    };

    Ok(ModelCache {
        predictions,
        actuals,
        timestamps: timestamps_out,
        metrics,
        test_start: n_train,
    })
}

// =============================================================================
// Prophet-like (trend + weekly seasonality via Fourier terms)
// =============================================================================

pub fn train_prophet(
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    use linfa::prelude::*;
    use linfa_linear::LinearRegression;

    let hp = preset_params(preset);
    // Number of Fourier pairs for seasonality
    let n_fourier = match preset {
        "Precis" => 8,
        "Standard" => 5,
        _ => 3,
    };

    let n = data.len();
    let period = 48.0_f64; // 48 half-hour steps ≈ 1 day

    // Build feature matrix: [t/n, sin(2πk*t/P), cos(2πk*t/P), ...]
    let n_features = 1 + 2 * n_fourier;
    let mut x_mat = Array2::<f64>::zeros((n, n_features));
    for t in 0..n {
        x_mat[[t, 0]] = t as f64 / n as f64; // linear trend
        for k in 1..=n_fourier {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / period;
            x_mat[[t, 2 * k - 1]] = angle.sin();
            x_mat[[t, 2 * k]] = angle.cos();
        }
    }

    let (norm, min_v, max_v) = normalize(data);
    let y_arr = Array1::from_vec(norm);

    let n_train = ((n as f64) * (1.0 - test_split)).ceil() as usize;
    let n_train = n_train.max(1).min(n - 1);

    let x_train = x_mat.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train = y_arr.slice(ndarray::s![..n_train]).to_owned();

    let dataset = Dataset::new(x_train, y_train);
    let model = LinearRegression::new()
        .fit(&dataset)
        .context("Prophet (linear) fit failed")?;

    // Manual prediction using model weights
    let weights = model.params();
    let intercept = model.intercept();
    let predictions: Vec<f64> = (0..n)
        .map(|t| {
            let pred = weights
                .iter()
                .enumerate()
                .map(|(j, w)| w * x_mat[[t, j]])
                .sum::<f64>()
                + intercept;
            denormalize(pred, min_v, max_v)
        })
        .collect();
    let actuals: Vec<f64> = data.to_vec();

    let test_preds = &predictions[n_train..];
    let test_actuals = &actuals[n_train..];
    let metrics = compute_metrics(test_actuals, test_preds);

    let _ = hp; // suppress unused warning

    Ok(ModelCache {
        predictions,
        actuals,
        timestamps: timestamps.to_vec(),
        metrics,
        test_start: n_train,
    })
}

// =============================================================================
// Dispatch
// =============================================================================

pub fn train(
    model_type: &str,
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    match model_type {
        "LinearRegression" => train_linear_regression(data, preset, test_split, timestamps),
        "RandomForest"     => train_random_forest(data, preset, test_split, timestamps),
        "LSTM"             => train_mlp(data, preset, test_split, timestamps),
        "GradientBoosting" => train_gradient_boosting(data, preset, test_split, timestamps),
        "Prophet"          => train_prophet(data, preset, test_split, timestamps),
        other              => anyhow::bail!("Unknown model type: {}", other),
    }
}

// =============================================================================
// Gradient Boosting (additive ensemble of regression stumps)
// =============================================================================

pub fn train_gradient_boosting(
    data: &[f64],
    preset: &str,
    test_split: f64,
    timestamps: &[String],
) -> Result<ModelCache> {
    use rand::Rng;

    let hp = preset_params(preset);
    let lr_gb = 0.1_f64; // GB learning rate (shrinkage)
    let (norm, min_v, max_v) = normalize(data);
    let (x, y) = build_xy(&norm, hp.window_size);

    let n = x.nrows();
    let n_train = ((n as f64) * (1.0 - test_split)).ceil() as usize;
    let n_train = n_train.max(1).min(n - 1);

    let y_train: Vec<f64> = (0..n_train).map(|i| y[i]).collect();
    let y_mean = y_train.iter().sum::<f64>() / y_train.len() as f64;

    // F_0 = mean of training targets
    let mut f_train = vec![y_mean; n_train];
    let mut stumps_with_lr: Vec<(Stump, f64)> = Vec::new();

    let n_trees = hp.n_estimators.min(200);
    let mut rng = rand::thread_rng();

    for _ in 0..n_trees {
        // Pseudo-residuals
        let residuals: Vec<f64> = y_train.iter().zip(&f_train).map(|(y, f)| y - f).collect();

        // Fit stump on residuals (subsample 80%)
        let n_sub = (n_train as f64 * 0.8).ceil() as usize;
        let sub_idx: Vec<usize> = {
            let mut idx: Vec<usize> = (0..n_train).collect();
            for i in (1..n_train).rev() {
                let j = rng.gen_range(0..=i);
                idx.swap(i, j);
            }
            idx[..n_sub].to_vec()
        };
        let x_sub = Array2::from_shape_fn((n_sub, hp.window_size), |(i, j)| {
            x[[sub_idx[i], j]]
        });
        let res_sub: Vec<f64> = sub_idx.iter().map(|&i| residuals[i]).collect();

        if let Some(stump) = Stump::fit(&x_sub, &res_sub) {
            // Update F on training set
            for i in 0..n_train {
                let row: Vec<f64> = (0..hp.window_size).map(|j| x[[i, j]]).collect();
                f_train[i] += lr_gb * stump.predict_one(&row);
            }
            stumps_with_lr.push((stump, lr_gb));
        }
    }

    // Predict on full dataset
    let predictions: Vec<f64> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..hp.window_size).map(|j| x[[i, j]]).collect();
            let pred = y_mean
                + stumps_with_lr
                    .iter()
                    .map(|(s, lr)| lr * s.predict_one(&row))
                    .sum::<f64>();
            denormalize(pred, min_v, max_v)
        })
        .collect();
    let actuals: Vec<f64> = y.iter().map(|v| denormalize(*v, min_v, max_v)).collect();

    let metrics = compute_metrics(&actuals[n_train..], &predictions[n_train..]);
    let ts_offset = hp.window_size.min(timestamps.len());

    Ok(ModelCache {
        predictions,
        actuals,
        timestamps: timestamps[ts_offset..].to_vec(),
        metrics,
        test_start: n_train,
    })
}
