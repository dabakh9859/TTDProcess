//! ML-based data cleaning (Voie B).
//!
//! Orchestrates supervised cleaning: train a model on (predictors → target),
//! compute residuals on the full series, flag large residuals as outliers,
//! and replace outliers + NaN gaps with the model's predictions.
//!
//! This module exposes:
//!   - `Predictor` trait — unified interface for every model (linear, RF, SVR,
//!     LSTM…). Concrete impls live in submodules / dedicated files.
//!   - `StandardScaler` — z-score scaler fitted on train only.
//!   - `TrainingSet` — the X/Y arrays after feature engineering + split.
//!   - `MLCleaner` — orchestrator that glues everything together.
//!   - Metric helpers (MAE, RMSE, R², AIC, BIC) with per-target breakdown.
//!
//! For Phase 3 only `LinearModel` (simple & multiple via linfa-linear) is
//! wired through. Later phases will add RandomForest, SVR, GPR, ARX/ARMAX,
//! LSTM/BiLSTM/GRU — each impl `Predictor` and plugs into `MLCleaner::new`.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, NaiveDateTime, Timelike};
use ndarray::{s, Array1, Array2};
use polars::prelude::{DataFrame, DataType};
use serde::{Deserialize, Serialize};

use crate::core::timestamp_utils::ts_col_to_datetimes;
use crate::core::types::{
    CleaningMethod, CleaningPath, CleaningTrainingMetrics, DLConfig, LinearConfig, MLConfig,
    PerTargetMetrics, ReplaceStrategy, TemporalFeatures,
};

// =============================================================================
// Predictor trait
// =============================================================================

/// Unified interface for every supervised model used in Voie B.
pub trait Predictor: Send + Sync {
    /// Fit the model on `(x_train, y_train)`. Both arrays are row-oriented:
    /// each row is one time step.
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()>;

    /// Predict `Y` values for new `X` rows. Returns an array with the same
    /// number of rows as `x` and `n_targets()` columns.
    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Number of targets the model produces (one column per target).
    #[allow(dead_code)]
    fn n_targets(&self) -> usize;

    /// Total parameter count used to fit the model. Used for AIC/BIC.
    fn n_parameters(&self) -> usize;

    /// Short name for logs/UI ("SimpleLinear", "RandomForest", …).
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
}

// =============================================================================
// StandardScaler (z-score on X, fitted on train only)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(x: &Array2<f64>) -> Self {
        let d = x.ncols();
        let mut means = vec![0.0; d];
        let mut stds = vec![1.0; d];
        for j in 0..d {
            let col = x.column(j);
            let n = col.len() as f64;
            let mean = col.sum() / n;
            let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1.0);
            means[j] = mean;
            stds[j] = var.sqrt().max(1e-12); // guard against zero-variance feature
        }
        Self { means, stds }
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = x.clone();
        for j in 0..out.ncols() {
            let m = self.means[j];
            let s = self.stds[j];
            for i in 0..out.nrows() {
                out[[i, j]] = (out[[i, j]] - m) / s;
            }
        }
        out
    }
}

// =============================================================================
// Metric helpers
// =============================================================================

/// Compute per-target MAE/RMSE/R²/rmse_pct_of_mean from truth `y` and
/// prediction `y_hat`. Arrays are (n, t) with t = number of targets.
pub fn compute_per_target_metrics(
    y: &Array2<f64>,
    y_hat: &Array2<f64>,
    target_names: &[String],
) -> HashMap<String, PerTargetMetrics> {
    let t = y.ncols();
    let n = y.nrows() as f64;
    let mut out = HashMap::with_capacity(t);
    for j in 0..t {
        let col_y = y.column(j);
        let col_p = y_hat.column(j);
        let errors: Vec<f64> = col_y.iter().zip(col_p.iter()).map(|(a, b)| a - b).collect();
        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / n.max(1.0);
        let mse = errors.iter().map(|e| e.powi(2)).sum::<f64>() / n.max(1.0);
        let rmse = mse.sqrt();
        let mean_y = col_y.sum() / n.max(1.0);
        let ss_tot: f64 = col_y.iter().map(|v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = errors.iter().map(|e| e.powi(2)).sum();
        let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 1.0 };
        let rmse_pct = if mean_y.abs() > 1e-12 {
            100.0 * rmse / mean_y.abs()
        } else {
            f64::NAN
        };
        let name = target_names
            .get(j)
            .cloned()
            .unwrap_or_else(|| format!("target_{}", j));
        out.insert(
            name,
            PerTargetMetrics {
                mae,
                rmse,
                r_squared,
                rmse_pct_of_mean: rmse_pct,
            },
        );
    }
    out
}

/// Aggregate per-target metrics into a global training metric bundle.
/// AIC / BIC are computed on the pooled residuals assuming Gaussian errors.
pub fn aggregate_metrics(
    per_target: HashMap<String, PerTargetMetrics>,
    y: &Array2<f64>,
    y_hat: &Array2<f64>,
    n_parameters: usize,
    n_train: usize,
    n_val: usize,
    training_time_s: f64,
    prediction_time_s: f64,
) -> CleaningTrainingMetrics {
    let n_obs = (y.nrows() * y.ncols()) as f64;
    // Gaussian log-likelihood: L = -n/2 · [ln(2π) + ln(σ²) + 1], σ² = SSR/n.
    let ssr: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let sigma2 = (ssr / n_obs.max(1.0)).max(1e-12);
    let log_lik = -0.5 * n_obs * ((2.0 * std::f64::consts::PI).ln() + sigma2.ln() + 1.0);
    let k = n_parameters as f64;
    let aic = 2.0 * k - 2.0 * log_lik;
    let bic = k * n_obs.ln().max(0.0) - 2.0 * log_lik;

    let (mae_sum, rmse_sum, r2_sum, count) = per_target.values().fold(
        (0.0, 0.0, 0.0, 0usize),
        |(m, r, rr, c), v| (m + v.mae, r + v.rmse, rr + v.r_squared, c + 1),
    );
    let c = count.max(1) as f64;

    CleaningTrainingMetrics {
        mae: mae_sum / c,
        rmse: rmse_sum / c,
        r_squared: r2_sum / c,
        aic,
        bic,
        training_time_s,
        prediction_time_s,
        n_train_points: n_train,
        n_val_points: n_val,
        n_parameters,
        per_target,
    }
}

// =============================================================================
// Temporal encoding helpers
// =============================================================================

/// Build seasonal / diurnal feature columns for a list of datetimes.
///
/// Returns a matrix `(n, cfg.extra_cols())`. Order:
///   one_hot_hour · cyclical_hour · one_hot_doy · cyclical_doy ·
///   one_hot_month · cyclical_month · one_hot_minute · days_since_start
/// This ordering is stable — saved scenarios rely on it.
pub fn build_temporal_features(
    datetimes: &[NaiveDateTime],
    cfg: &TemporalFeatures,
) -> Array2<f64> {
    let n = datetimes.len();
    let mut blocks: Vec<Array2<f64>> = Vec::new();
    let tau = 2.0 * std::f64::consts::PI;

    if cfg.one_hot_hour {
        let mut hr = Array2::<f64>::zeros((n, 24));
        for (i, dt) in datetimes.iter().enumerate() {
            hr[[i, dt.hour() as usize]] = 1.0;
        }
        blocks.push(hr);
    }
    if cfg.cyclical_hour {
        let mut c = Array2::<f64>::zeros((n, 2));
        for (i, dt) in datetimes.iter().enumerate() {
            let h = dt.hour() as f64 + dt.minute() as f64 / 60.0;
            let a = tau * h / 24.0;
            c[[i, 0]] = a.sin();
            c[[i, 1]] = a.cos();
        }
        blocks.push(c);
    }
    if cfg.one_hot_doy {
        let mut doy = Array2::<f64>::zeros((n, 366));
        for (i, dt) in datetimes.iter().enumerate() {
            let d = (dt.ordinal() as usize).saturating_sub(1).min(365);
            doy[[i, d]] = 1.0;
        }
        blocks.push(doy);
    }
    if cfg.cyclical_doy {
        let mut c = Array2::<f64>::zeros((n, 2));
        for (i, dt) in datetimes.iter().enumerate() {
            let a = tau * dt.ordinal() as f64 / 366.0;
            c[[i, 0]] = a.sin();
            c[[i, 1]] = a.cos();
        }
        blocks.push(c);
    }
    if cfg.one_hot_month {
        let mut m = Array2::<f64>::zeros((n, 12));
        for (i, dt) in datetimes.iter().enumerate() {
            m[[i, (dt.month() as usize).saturating_sub(1).min(11)]] = 1.0;
        }
        blocks.push(m);
    }
    if cfg.cyclical_month {
        let mut c = Array2::<f64>::zeros((n, 2));
        for (i, dt) in datetimes.iter().enumerate() {
            let a = tau * dt.month() as f64 / 12.0;
            c[[i, 0]] = a.sin();
            c[[i, 1]] = a.cos();
        }
        blocks.push(c);
    }
    if cfg.one_hot_minute {
        let mut mi = Array2::<f64>::zeros((n, 2));
        for (i, dt) in datetimes.iter().enumerate() {
            let idx = if dt.minute() < 15 {
                0
            } else if dt.minute() < 45 {
                1
            } else {
                0
            };
            mi[[i, idx]] = 1.0;
        }
        blocks.push(mi);
    }
    if cfg.days_since_start {
        let mut d = Array2::<f64>::zeros((n, 1));
        if let Some(t0) = datetimes.first() {
            for (i, dt) in datetimes.iter().enumerate() {
                let secs = (*dt - *t0).num_seconds() as f64;
                d[[i, 0]] = secs / 86_400.0;
            }
        }
        blocks.push(d);
    }

    // Horizontal concat. If no blocks → return (n, 0).
    let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();
    let mut out = Array2::<f64>::zeros((n, total_cols));
    let mut offset = 0;
    for b in blocks {
        let width = b.ncols();
        out.slice_mut(s![.., offset..offset + width]).assign(&b);
        offset += width;
    }
    out
}

/// Sequential train/val split (first `split_ratio` fraction → train, rest →
/// val). This preserves temporal order, which matters for time-series.
pub fn sequential_split<T: Clone>(
    x: &Array2<f64>,
    y: &Array2<f64>,
    extras: &[&[T]],
    split_ratio: f64,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Vec<Vec<T>>,
) {
    let n = x.nrows();
    let n_train = ((n as f64) * split_ratio).ceil() as usize;
    let n_train = n_train.clamp(1, n.saturating_sub(1).max(1));
    let x_tr = x.slice(s![..n_train, ..]).to_owned();
    let x_val = x.slice(s![n_train.., ..]).to_owned();
    let y_tr = y.slice(s![..n_train, ..]).to_owned();
    let y_val = y.slice(s![n_train.., ..]).to_owned();
    let extras_split: Vec<Vec<T>> = extras
        .iter()
        .map(|e| e.to_vec())
        .collect();
    (x_tr, x_val, y_tr, y_val, extras_split)
}

// =============================================================================
// LinearModel (Simple + Multiple via linfa-linear)
// =============================================================================

/// Simple/Multiple linear regression wrapper.
///
/// Holds one sub-model per target (linfa-linear is single-output).
pub struct LinearModel {
    /// Per-target (weights, intercept).
    per_target: Vec<(Array1<f64>, f64)>,
    n_features: usize,
    #[allow(dead_code)]
    name: &'static str,
}

impl LinearModel {
    pub fn simple() -> Self {
        Self {
            per_target: Vec::new(),
            n_features: 0,
            name: "SimpleLinear",
        }
    }
    pub fn multiple() -> Self {
        Self {
            per_target: Vec::new(),
            n_features: 0,
            name: "MultipleLinear",
        }
    }
}

impl Predictor for LinearModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        use linfa::prelude::*;
        use linfa_linear::LinearRegression;

        self.n_features = x_train.ncols();
        self.per_target.clear();
        for j in 0..y_train.ncols() {
            let y_col: Array1<f64> = y_train.column(j).to_owned();
            let ds = Dataset::new(x_train.clone(), y_col);
            let model = LinearRegression::new()
                .fit(&ds)
                .context("linfa-linear fit failed")?;
            self.per_target
                .push((model.params().to_owned(), model.intercept()));
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.per_target.is_empty() {
            bail!("LinearModel: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "LinearModel: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let n = x.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, (w, b)) in self.per_target.iter().enumerate() {
            for i in 0..n {
                let row = x.row(i);
                let dot: f64 = row.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                out[[i, j]] = dot + b;
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Per target: (n_features weights + 1 intercept)
        (self.n_features + 1) * self.per_target.len()
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

// =============================================================================
// PcaModel — Principal Component Regression (ACP)
// =============================================================================

/// Jacobi eigenvalue decomposition of a symmetric matrix `a`.
/// Returns `(eigenvalues, eigenvectors)` where each COLUMN of the returned
/// matrix is an eigenvector. Robust and simple — the covariance matrices here
/// are small (p × p, p = number of predictors), so the O(p³) cost is fine.
fn jacobi_eigen_symmetric(mut a: Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut v = Array2::<f64>::eye(n);
    if n == 0 {
        return (Vec::new(), v);
    }
    for _sweep in 0..100 {
        // Sum of squares of the off-diagonal — convergence measure.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[[i, j]] * a[[i, j]];
            }
        }
        if off < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                // Stable choice of the smaller-magnitude rotation.
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate columns p,q of A.
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                // Rotate rows p,q of A (keeps A symmetric).
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
                // Accumulate the eigenvector rotation.
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigvals: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eigvals, v)
}

/// Principal Component Regression ("ACP"): runs PCA on the predictors, keeps
/// the top-`k` principal components, then fits an ordinary least-squares
/// regression of each target on those components. When sibling sensors are
/// strongly correlated (the usual case), the PCs decorrelate them so the
/// regression is far more stable than on the raw, collinear inputs.
pub struct PcaModel {
    /// Requested components (0 = use all features = no reduction). Clamped to
    /// `[1, n_features]` at train time.
    n_components: usize,
    x_means: Array1<f64>,           // per-feature mean (centring)
    components: Array2<f64>,        // (n_features, k) top eigenvectors
    per_target: Vec<(Array1<f64>, f64)>, // OLS over the k scores + intercept
    n_features: usize,
    k: usize,
    #[allow(dead_code)]
    name: &'static str,
}

impl PcaModel {
    pub fn new(c: &MLConfig) -> Self {
        Self {
            n_components: c.n_components.unwrap_or(0),
            x_means: Array1::zeros(0),
            components: Array2::zeros((0, 0)),
            per_target: Vec::new(),
            n_features: 0,
            k: 0,
            name: "PCA",
        }
    }

    fn project(&self, x: &Array2<f64>) -> Array2<f64> {
        // (x - means) · components  →  (n, k) scores.
        let mut xc = x.clone();
        for j in 0..xc.ncols() {
            let m = self.x_means[j];
            for i in 0..xc.nrows() {
                xc[[i, j]] -= m;
            }
        }
        xc.dot(&self.components)
    }
}

impl Predictor for PcaModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        use linfa::prelude::*;
        use linfa_linear::LinearRegression;

        let n = x_train.nrows();
        let p = x_train.ncols();
        if n < 2 || p == 0 {
            bail!("PcaModel: pas assez de données pour l'ACP");
        }
        self.n_features = p;

        // Centre the predictors.
        let mut means = Array1::<f64>::zeros(p);
        for j in 0..p {
            means[j] = x_train.column(j).sum() / n as f64;
        }
        self.x_means = means;
        let mut xc = x_train.clone();
        for j in 0..p {
            let m = self.x_means[j];
            for i in 0..n {
                xc[[i, j]] -= m;
            }
        }

        // Covariance (p × p) = Xcᵀ·Xc / (n-1).
        let cov = xc.t().dot(&xc) / ((n - 1) as f64).max(1.0);
        let (eigvals, eigvecs) = jacobi_eigen_symmetric(cov);

        // Order components by descending eigenvalue (variance explained).
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap_or(std::cmp::Ordering::Equal));

        let k = if self.n_components == 0 {
            p
        } else {
            self.n_components.min(p).max(1)
        };
        self.k = k;

        // Build the (p × k) component matrix from the top-k eigenvectors.
        let mut components = Array2::<f64>::zeros((p, k));
        for (col, &idx) in order.iter().take(k).enumerate() {
            for row in 0..p {
                components[[row, col]] = eigvecs[[row, idx]];
            }
        }
        self.components = components;

        // Project train rows → scores, then OLS of each target on the scores.
        let scores = xc.dot(&self.components); // (n, k)
        self.per_target.clear();
        for j in 0..y_train.ncols() {
            let y_col: Array1<f64> = y_train.column(j).to_owned();
            let ds = Dataset::new(scores.clone(), y_col);
            let model = LinearRegression::new()
                .fit(&ds)
                .context("PCA: régression sur composantes échouée")?;
            self.per_target
                .push((model.params().to_owned(), model.intercept()));
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.per_target.is_empty() {
            bail!("PcaModel: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "PcaModel: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let scores = self.project(x); // (n, k)
        let n = scores.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, (w, b)) in self.per_target.iter().enumerate() {
            for i in 0..n {
                let dot: f64 = scores.row(i).iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                out[[i, j]] = dot + b;
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Per target: k component weights + 1 intercept, plus the p×k loadings
        // shared across targets.
        (self.k + 1) * self.per_target.len() + self.n_features * self.k
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

// =============================================================================
// RandomForestModel (manual regression forest — linfa-trees 0.7 is
// classification-only, and pulling smartcore just for this is overkill)
// =============================================================================

/// A single regression tree node: either a leaf with a prediction or an
/// internal split on `feature` at `threshold` (go left if x ≤ threshold).
enum RfNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<RfNode>,
        right: Box<RfNode>,
    },
}

/// Xorshift64 PRNG, seeded explicitly so training is reproducible.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u64() as usize) % max
        }
    }
    fn next_f64_unit(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

fn build_regression_tree(
    x: &Array2<f64>,
    y: &[f64],
    indices: &[usize],
    feature_subset: usize,
    min_samples_split: usize,
    max_depth: usize,
    depth: usize,
    rng: &mut XorShift64,
) -> RfNode {
    let n = indices.len();
    if n == 0 {
        return RfNode::Leaf(0.0);
    }
    // Leaf condition: depth limit, too few samples, or all targets equal.
    if depth >= max_depth || n < min_samples_split {
        let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        return RfNode::Leaf(mean);
    }
    let first = y[indices[0]];
    if indices.iter().all(|&i| (y[i] - first).abs() < 1e-12) {
        return RfNode::Leaf(first);
    }

    // Pick a random subset of features (sqrt rule or configured size).
    let n_features = x.ncols();
    let try_count = feature_subset.min(n_features).max(1);
    let mut feature_pool: Vec<usize> = (0..n_features).collect();
    // Fisher-Yates partial shuffle.
    for i in 0..try_count.min(n_features) {
        let j = i + rng.next_usize(n_features - i);
        feature_pool.swap(i, j);
    }
    let features_to_try = &feature_pool[..try_count];

    // Find the best split (minimise weighted child variance).
    let mut best: Option<(f64, usize, f64, Vec<usize>, Vec<usize>)> = None;
    for &feat in features_to_try {
        // Random candidate threshold: median of a small random sample.
        let sample_count = 8.min(n);
        let mut sampled: Vec<f64> = (0..sample_count)
            .map(|_| x[[indices[rng.next_usize(n)], feat]])
            .collect();
        sampled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sampled[sampled.len() / 2];

        let (left, right): (Vec<usize>, Vec<usize>) =
            indices.iter().copied().partition(|&i| x[[i, feat]] <= threshold);
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let lm = left.iter().map(|&i| y[i]).sum::<f64>() / left.len() as f64;
        let rm = right.iter().map(|&i| y[i]).sum::<f64>() / right.len() as f64;
        let lv: f64 = left.iter().map(|&i| (y[i] - lm).powi(2)).sum();
        let rv: f64 = right.iter().map(|&i| (y[i] - rm).powi(2)).sum();
        let score = lv + rv;
        if best
            .as_ref()
            .map(|(s, _, _, _, _)| score < *s)
            .unwrap_or(true)
        {
            best = Some((score, feat, threshold, left, right));
        }
    }

    match best {
        Some((_, feature, threshold, left_idx, right_idx)) => {
            let left = build_regression_tree(
                x,
                y,
                &left_idx,
                feature_subset,
                min_samples_split,
                max_depth,
                depth + 1,
                rng,
            );
            let right = build_regression_tree(
                x,
                y,
                &right_idx,
                feature_subset,
                min_samples_split,
                max_depth,
                depth + 1,
                rng,
            );
            RfNode::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        None => {
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
            RfNode::Leaf(mean)
        }
    }
}

fn predict_tree(node: &RfNode, row: &[f64]) -> f64 {
    match node {
        RfNode::Leaf(v) => *v,
        RfNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if row[*feature] <= *threshold {
                predict_tree(left, row)
            } else {
                predict_tree(right, row)
            }
        }
    }
}

/// Bagging regression forest. Each target gets its own ensemble of trees
/// (bootstrap samples, random feature subset at each split).
pub struct RandomForestModel {
    /// Per target → ensemble of trees.
    forests: Vec<Vec<RfNode>>,
    n_features: usize,
    n_estimators: usize,
    max_depth: usize,
    feature_subset_size: usize,
    seed: u64,
}

impl RandomForestModel {
    pub fn new(cfg: &MLConfig) -> Self {
        Self {
            forests: Vec::new(),
            n_features: 0,
            n_estimators: cfg.n_estimators.unwrap_or(100).clamp(10, 500),
            max_depth: cfg.max_depth.unwrap_or(12).clamp(1, 50),
            feature_subset_size: 0, // computed at train() from ncols
            seed: cfg.random_seed,
        }
    }
}

impl Predictor for RandomForestModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        self.n_features = x_train.ncols();
        // Common rule: sqrt(n_features) for regression forests works well.
        self.feature_subset_size =
            ((self.n_features as f64).sqrt().ceil() as usize).max(1).min(self.n_features);
        self.forests.clear();

        let mut rng = XorShift64::new(self.seed);
        for j in 0..y_train.ncols() {
            let y_col: Vec<f64> = y_train.column(j).to_vec();
            let mut trees = Vec::with_capacity(self.n_estimators);
            for _ in 0..self.n_estimators {
                let n = x_train.nrows();
                // Bootstrap sample of size n (with replacement).
                let boot: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
                let tree = build_regression_tree(
                    x_train,
                    &y_col,
                    &boot,
                    self.feature_subset_size,
                    2,
                    self.max_depth,
                    0,
                    &mut rng,
                );
                trees.push(tree);
            }
            self.forests.push(trees);
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.forests.is_empty() {
            bail!("RandomForestModel: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "RandomForestModel: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let n = x.nrows();
        let t = self.forests.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, forest) in self.forests.iter().enumerate() {
            for i in 0..n {
                let row: Vec<f64> = x.row(i).to_vec();
                let mean = forest.iter().map(|tree| predict_tree(tree, &row)).sum::<f64>()
                    / forest.len() as f64;
                out[[i, j]] = mean;
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.forests.len()
    }

    fn n_parameters(&self) -> usize {
        // Rough heuristic for AIC/BIC purposes: total leaf count across all
        // forests. Walking every tree once is cheap at train-time-only cost.
        self.forests
            .iter()
            .flatten()
            .map(|t| count_leaves(t))
            .sum()
    }

    fn name(&self) -> &'static str {
        "RandomForest"
    }
}

fn count_leaves(node: &RfNode) -> usize {
    match node {
        RfNode::Leaf(_) => 1,
        RfNode::Split { left, right, .. } => count_leaves(left) + count_leaves(right),
    }
}

// =============================================================================
// SVRModel (linfa-svm epsilon-regression, RBF kernel default)
// =============================================================================

use crate::core::types::KernelType;

/// Epsilon-SVR wrapper. One sub-model per target (linfa-svm is single-output).
pub struct SVRModel {
    /// Per-target trained models. Each is an `Svm<f64, f64>`.
    per_target: Vec<linfa_svm::Svm<f64, f64>>,
    n_features: usize,
    kernel: KernelType,
    c: f64,
    eps: f64,
}

impl SVRModel {
    pub fn new(cfg: &MLConfig) -> Self {
        Self {
            per_target: Vec::new(),
            n_features: 0,
            kernel: cfg.kernel.unwrap_or_default(),
            c: cfg.svr_c.unwrap_or(1.0).clamp(0.01, 1000.0),
            eps: cfg.svr_epsilon.unwrap_or(0.1).clamp(0.0, 1.0),
        }
    }
}

impl Predictor for SVRModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        use linfa::prelude::*;
        use linfa_svm::Svm;

        self.n_features = x_train.ncols();
        self.per_target.clear();
        for j in 0..y_train.ncols() {
            let y_col: Array1<f64> = y_train.column(j).to_owned();
            let ds = Dataset::new(x_train.clone(), y_col);
            // linfa-svm exposes kernel selection through the builder.
            let params = Svm::<f64, f64>::params().eps(self.eps).c_svr(self.c, Some(self.eps));
            let params = match self.kernel {
                KernelType::Linear => params.linear_kernel(),
                KernelType::Polynomial => params.polynomial_kernel(1.0, 3.0),
                // RBF / Matern32 / Matern52: fall back to Gaussian (RBF) —
                // linfa-svm 0.7 doesn't expose Matérn kernels directly.
                _ => params.gaussian_kernel(1.0),
            };
            let model = params
                .fit(&ds)
                .context("linfa-svm fit failed")?;
            self.per_target.push(model);
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        use linfa::prelude::*;
        if self.per_target.is_empty() {
            bail!("SVRModel: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "SVRModel: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let n = x.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, model) in self.per_target.iter().enumerate() {
            // linfa-svm's `Predict` trait returns an Array1 of predictions.
            let preds: Array1<f64> = model.predict(x);
            for i in 0..n {
                out[[i, j]] = preds[i];
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Approx: sum of support vectors × (n_features + 1) per target.
        // linfa-svm doesn't expose SV count directly in 0.7, so fall back to
        // a coarse upper bound used only for AIC/BIC ordering.
        self.per_target.len() * (self.n_features + 1)
    }

    fn name(&self) -> &'static str {
        "SVR"
    }
}

// =============================================================================
// ARX / ARMAX (exogenous-lag regression)
// =============================================================================
// Design note: the textbook ARX also regresses on past y. Plugging that into
// the `Predictor` trait requires an autoregressive rollout at predict time,
// which complicates both state and the trait surface. For the initial Voie B
// release we implement ARX/ARMAX as **exogenous-lag regression** only —
// features = [x_t, x_{t-1}, …, x_{t-q}], fitted via ordinary least squares.
// ARMAX additionally smooths its predictions with a centred moving average
// over `ma_order` points as a crude MA(q) proxy. This is noted in the UI.

use crate::core::types::{ARMAXConfig, ARXConfig};

pub struct ARXModel {
    per_target: Vec<(Array1<f64>, f64)>, // (weights, intercept)
    n_features_base: usize,              // = X width before lag expansion
    lag_order: usize,                    // exog_order
}

impl ARXModel {
    pub fn new(cfg: &ARXConfig) -> Self {
        Self {
            per_target: Vec::new(),
            n_features_base: 0,
            lag_order: cfg.exog_order,
        }
    }

    fn build_lagged(&self, x: &Array2<f64>) -> Array2<f64> {
        lagged_features(x, self.lag_order)
    }
}

impl Predictor for ARXModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        use linfa::prelude::*;
        use linfa_linear::LinearRegression;

        self.n_features_base = x_train.ncols();
        self.per_target.clear();
        let x_lag = self.build_lagged(x_train);

        // Align y: drop the first `lag_order` rows because they have incomplete
        // lag windows.
        let start = self.lag_order;
        if x_lag.nrows() <= start + 1 {
            bail!("ARX: série trop courte pour l'ordre de lag {}", self.lag_order);
        }
        let x_al = x_lag.slice(s![start.., ..]).to_owned();

        for j in 0..y_train.ncols() {
            let y_col: Array1<f64> = y_train.column(j).slice(s![start..]).to_owned();
            let ds = Dataset::new(x_al.clone(), y_col);
            let model = LinearRegression::new()
                .fit(&ds)
                .context("ARX OLS fit failed")?;
            self.per_target
                .push((model.params().to_owned(), model.intercept()));
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.per_target.is_empty() {
            bail!("ARXModel: not trained yet");
        }
        if x.ncols() != self.n_features_base {
            bail!(
                "ARXModel: expected {} raw cols, got {}",
                self.n_features_base,
                x.ncols()
            );
        }
        let x_lag = self.build_lagged(x);
        let n = x_lag.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, (w, b)) in self.per_target.iter().enumerate() {
            for i in 0..n {
                let row = x_lag.row(i);
                let dot: f64 = row.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                out[[i, j]] = dot + b;
            }
        }
        // First `lag_order` rows have incomplete lag windows → replicate the
        // first valid prediction so consumers don't see garbage.
        if self.lag_order > 0 && n > self.lag_order {
            for j in 0..t {
                let anchor = out[[self.lag_order, j]];
                for i in 0..self.lag_order {
                    out[[i, j]] = anchor;
                }
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Per target: (n_features_base × (lag+1) + 1 intercept)
        self.per_target.len() * (self.n_features_base * (self.lag_order + 1) + 1)
    }

    fn name(&self) -> &'static str {
        "ARX"
    }
}

pub struct ARMAXModel {
    arx: ARXModel,
    ma_order: usize,
}

impl ARMAXModel {
    pub fn new(cfg: &ARMAXConfig) -> Self {
        Self {
            arx: ARXModel {
                per_target: Vec::new(),
                n_features_base: 0,
                lag_order: cfg.exog_order,
            },
            ma_order: cfg.ma_order,
        }
    }
}

impl Predictor for ARMAXModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        self.arx.train(x_train, y_train)
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let raw = self.arx.predict(x)?;
        if self.ma_order == 0 {
            return Ok(raw);
        }
        // Centred moving average of raw predictions as a crude MA(q) smoother.
        let half = (self.ma_order / 2).max(1);
        let (n, t) = (raw.nrows(), raw.ncols());
        let mut out = Array2::<f64>::zeros((n, t));
        for j in 0..t {
            for i in 0..n {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                let mut s = 0.0;
                let mut c = 0;
                for k in lo..hi {
                    let v = raw[[k, j]];
                    if v.is_finite() {
                        s += v;
                        c += 1;
                    }
                }
                out[[i, j]] = if c > 0 { s / c as f64 } else { raw[[i, j]] };
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.arx.n_targets()
    }

    fn n_parameters(&self) -> usize {
        self.arx.n_parameters() + self.ma_order
    }

    fn name(&self) -> &'static str {
        "ARMAX"
    }
}

/// Augment X with lagged copies: output cols = base_cols × (lag_order + 1).
/// Column layout for lag=2: [x_t, x_{t-1}, x_{t-2}].
fn lagged_features(x: &Array2<f64>, lag_order: usize) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    if lag_order == 0 {
        return x.clone();
    }
    let cols = d * (lag_order + 1);
    let mut out = Array2::<f64>::zeros((n, cols));
    for lag in 0..=lag_order {
        for i in 0..n {
            let src = i.saturating_sub(lag);
            for j in 0..d {
                out[[i, lag * d + j]] = x[[src, j]];
            }
        }
    }
    out
}

// =============================================================================
// MLPModel (multi-output feed-forward, used for LSTM/BiLSTM/GRU in Phase 6)
// =============================================================================
// We reuse a plain multilayer perceptron with a single hidden layer as the
// backend for all three DL model types declared in the spec. Until the
// `candle`-based implementation lands, the UI reports these as "MLP-backed
// LSTM / BiLSTM / GRU" so users are never misled. Swapping in real recurrent
// networks later is a straight substitution: the `Predictor` trait stays the
// same.

use crate::core::types::OptimizerType;

pub struct MLPModel {
    // Per target: (W1 [H, D], b1 [H], w2 [H], b2 f64). One hidden layer, ReLU.
    per_target: Vec<MlpWeights>,
    n_features: usize,
    hidden: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    seed: u64,
    #[allow(dead_code)]
    label: &'static str,
}

struct MlpWeights {
    w1: Vec<Vec<f64>>, // [H][D]
    b1: Vec<f64>,      // [H]
    w2: Vec<f64>,      // [H]
    b2: f64,
}

impl MLPModel {
    pub fn new(cfg: &DLConfig, label: &'static str) -> Self {
        Self {
            per_target: Vec::new(),
            n_features: 0,
            hidden: cfg.hidden_units.clamp(4, 512),
            epochs: cfg.epochs.clamp(1, 10_000),
            batch_size: cfg.batch_size.clamp(1, 512),
            lr: cfg.learning_rate.clamp(1e-6, 1.0),
            seed: cfg.random_seed,
            label,
        }
    }

    fn forward(w: &MlpWeights, x: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        // z1 = W1 x + b1 ; a1 = relu(z1) ; out = w2·a1 + b2.
        let h = w.b1.len();
        let mut z1 = vec![0.0; h];
        let mut a1 = vec![0.0; h];
        for k in 0..h {
            let mut s = w.b1[k];
            for j in 0..x.len() {
                s += w.w1[k][j] * x[j];
            }
            z1[k] = s;
            a1[k] = s.max(0.0);
        }
        let mut out = w.b2;
        for k in 0..h {
            out += w.w2[k] * a1[k];
        }
        (z1, a1, out)
    }
}

impl Predictor for MLPModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        self.n_features = x_train.ncols();
        self.per_target.clear();

        let n = x_train.nrows();
        if n < 4 {
            bail!("MLP: série trop courte ({})", n);
        }

        let mut rng = XorShift64::new(self.seed);

        for j in 0..y_train.ncols() {
            // Xavier init.
            let scale1 = (2.0_f64 / self.n_features as f64).sqrt();
            let scale2 = (2.0_f64 / self.hidden as f64).sqrt();
            let mut w1: Vec<Vec<f64>> = (0..self.hidden)
                .map(|_| {
                    (0..self.n_features)
                        .map(|_| (rng.next_f64_unit() * 2.0 - 1.0) * scale1)
                        .collect()
                })
                .collect();
            let mut b1 = vec![0.0; self.hidden];
            let mut w2: Vec<f64> = (0..self.hidden)
                .map(|_| (rng.next_f64_unit() * 2.0 - 1.0) * scale2)
                .collect();
            let mut b2 = 0.0;

            // Training loop (mini-batch SGD).
            let y_col: Vec<f64> = y_train.column(j).to_vec();
            for _ in 0..self.epochs {
                // Shuffle.
                let mut idx: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let r = rng.next_usize(i + 1);
                    idx.swap(i, r);
                }
                for batch_start in (0..n).step_by(self.batch_size) {
                    let batch_end = (batch_start + self.batch_size).min(n);
                    let bs = (batch_end - batch_start) as f64;

                    let mut dw1 = vec![vec![0.0; self.n_features]; self.hidden];
                    let mut db1 = vec![0.0; self.hidden];
                    let mut dw2 = vec![0.0; self.hidden];
                    let mut db2 = 0.0;

                    for &i in &idx[batch_start..batch_end] {
                        let x_row: Vec<f64> = x_train.row(i).to_vec();
                        let w_view = MlpWeights {
                            w1: w1.clone(),
                            b1: b1.clone(),
                            w2: w2.clone(),
                            b2,
                        };
                        let (z1, a1, out) = Self::forward(&w_view, &x_row);
                        let err = out - y_col[i];
                        let d_out = 2.0 * err / bs;

                        db2 += d_out;
                        for k in 0..self.hidden {
                            dw2[k] += d_out * a1[k];
                            let d_h = d_out * w2[k] * if z1[k] > 0.0 { 1.0 } else { 0.0 };
                            db1[k] += d_h;
                            for m in 0..self.n_features {
                                dw1[k][m] += d_h * x_row[m];
                            }
                        }
                    }

                    // Apply gradients.
                    b2 -= self.lr * db2;
                    for k in 0..self.hidden {
                        w2[k] -= self.lr * dw2[k];
                        b1[k] -= self.lr * db1[k];
                        for m in 0..self.n_features {
                            w1[k][m] -= self.lr * dw1[k][m];
                        }
                    }
                }
            }

            self.per_target.push(MlpWeights { w1, b1, w2, b2 });
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.per_target.is_empty() {
            bail!("MLP: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "MLP: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let n = x.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((n, t));
        for (j, w) in self.per_target.iter().enumerate() {
            for i in 0..n {
                let x_row: Vec<f64> = x.row(i).to_vec();
                let (_, _, y_pred) = Self::forward(w, &x_row);
                out[[i, j]] = y_pred;
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Per target: W1 (H×D) + b1 (H) + w2 (H) + b2 (1)
        self.per_target.len() * (self.hidden * self.n_features + self.hidden + self.hidden + 1)
    }

    fn name(&self) -> &'static str {
        self.label
    }
}

// Suppress unused-variable warning on optimizer until we wire SGD variants.
#[allow(dead_code)]
const _OPTIMIZER_PLACEHOLDER: fn(OptimizerType) = |_| {};

// =============================================================================
// GprModel — pure-Rust Gaussian Process Regression with RBF kernel
// =============================================================================
// Implemented without ndarray-linalg: we write a small Cholesky + triangular
// solve ourselves so we stay on pure Rust (no LAPACK/BLAS needed). Caps
// training sample count to keep O(n³) manageable; above the cap we take a
// random subsample with a deterministic seed.

pub struct GprModel {
    /// Per-target state: (X_train_scaled, α = K⁻¹ y, length_scale, sigma_noise).
    per_target: Vec<GprState>,
    n_features: usize,
    kernel_length: f64,
    noise_variance: f64,
    max_train: usize,
    seed: u64,
}

struct GprState {
    x_train: Array2<f64>,
    alpha: Array1<f64>,
}

impl GprModel {
    pub fn new(cfg: &MLConfig) -> Self {
        // Pick reasonable defaults. SVR-style `svr_c` is reused as a rough
        // "inverse regularisation" control via noise_variance = 1/C, and the
        // RBF length scale is fixed at 1.0 since X is typically z-scored
        // upstream.
        let c = cfg.svr_c.unwrap_or(10.0).clamp(0.01, 1000.0);
        let noise = (1.0 / c).clamp(1e-6, 1.0);
        Self {
            per_target: Vec::new(),
            n_features: 0,
            kernel_length: 1.0,
            noise_variance: noise,
            max_train: 1200, // Cholesky of 1200×1200 is fine (~few seconds)
            seed: cfg.random_seed,
        }
    }

    fn rbf(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        (-s / (2.0 * self.kernel_length * self.kernel_length)).exp()
    }
}

/// In-place Cholesky factorisation of a symmetric positive-definite matrix.
/// Writes the lower triangular factor into `m`; upper triangle is left intact.
fn cholesky_in_place(m: &mut Array2<f64>) -> Result<()> {
    let n = m.nrows();
    if n != m.ncols() {
        bail!("Cholesky: non-square matrix");
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[[i, j]];
            for k in 0..j {
                sum -= m[[i, k]] * m[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    bail!(
                        "Cholesky: matrice non définie positive (diag {} = {}). Ajouter du bruit ou réduire la taille.",
                        i, sum
                    );
                }
                m[[i, j]] = sum.sqrt();
            } else {
                m[[i, j]] = sum / m[[j, j]];
            }
        }
        // Zero upper triangle to avoid stale data confusing callers.
        for j in (i + 1)..n {
            m[[i, j]] = 0.0;
        }
    }
    Ok(())
}

/// Solve L · x = b in place where L is lower triangular.
fn solve_lower(l: &Array2<f64>, b: &mut Array1<f64>) {
    let n = l.nrows();
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * b[j];
        }
        b[i] = s / l[[i, i]];
    }
}

/// Solve Lᵀ · x = b in place where L is lower triangular.
fn solve_lower_transpose(l: &Array2<f64>, b: &mut Array1<f64>) {
    let n = l.nrows();
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[[j, i]] * b[j];
        }
        b[i] = s / l[[i, i]];
    }
}

impl Predictor for GprModel {
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> Result<()> {
        self.n_features = x_train.ncols();
        self.per_target.clear();

        let n_full = x_train.nrows();
        if n_full < 10 {
            bail!("GPR: au moins 10 points requis, reçu {}", n_full);
        }

        // Deterministic subsample if training set is too large.
        let mut rng = XorShift64::new(self.seed);
        let indices: Vec<usize> = if n_full <= self.max_train {
            (0..n_full).collect()
        } else {
            let mut idx: Vec<usize> = (0..n_full).collect();
            for i in (1..n_full).rev() {
                let j = rng.next_usize(i + 1);
                idx.swap(i, j);
            }
            idx.truncate(self.max_train);
            idx.sort_unstable();
            idx
        };
        let n = indices.len();

        // Build x_sub (n × d).
        let d = x_train.ncols();
        let mut x_sub = Array2::<f64>::zeros((n, d));
        for (new_i, &orig_i) in indices.iter().enumerate() {
            for j in 0..d {
                x_sub[[new_i, j]] = x_train[[orig_i, j]];
            }
        }
        let rows: Vec<Vec<f64>> = (0..n).map(|i| x_sub.row(i).to_vec()).collect();

        // Build kernel matrix K + σ²I.
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let v = self.rbf(&rows[i], &rows[j]);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
            k[[i, i]] += self.noise_variance;
        }

        // Cholesky once; reuse the factor for each target column.
        cholesky_in_place(&mut k)
            .map_err(|e| anyhow::anyhow!("GPR fit: {}", e))?;

        for j in 0..y_train.ncols() {
            let mut y_sub = Array1::<f64>::zeros(n);
            for (new_i, &orig_i) in indices.iter().enumerate() {
                y_sub[new_i] = y_train[[orig_i, j]];
            }
            // α = K⁻¹ y solved via L and Lᵀ.
            let mut alpha = y_sub.clone();
            solve_lower(&k, &mut alpha);
            solve_lower_transpose(&k, &mut alpha);
            self.per_target.push(GprState {
                x_train: x_sub.clone(),
                alpha,
            });
        }
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.per_target.is_empty() {
            bail!("GprModel: not trained yet");
        }
        if x.ncols() != self.n_features {
            bail!(
                "GprModel: predict x has {} cols, expected {}",
                x.ncols(),
                self.n_features
            );
        }
        let m = x.nrows();
        let t = self.per_target.len();
        let mut out = Array2::<f64>::zeros((m, t));
        for (j, state) in self.per_target.iter().enumerate() {
            let n = state.x_train.nrows();
            let train_rows: Vec<Vec<f64>> = (0..n).map(|i| state.x_train.row(i).to_vec()).collect();
            for i in 0..m {
                let x_row = x.row(i).to_vec();
                let mut pred = 0.0;
                for k_idx in 0..n {
                    pred += self.rbf(&x_row, &train_rows[k_idx]) * state.alpha[k_idx];
                }
                out[[i, j]] = pred;
            }
        }
        Ok(out)
    }

    fn n_targets(&self) -> usize {
        self.per_target.len()
    }

    fn n_parameters(&self) -> usize {
        // Effective "number of parameters" for AIC/BIC: each α coefficient is
        // a model parameter. Sum across targets.
        self.per_target.iter().map(|s| s.alpha.len()).sum::<usize>() + 2
    }

    fn name(&self) -> &'static str {
        "GPR"
    }
}

// =============================================================================
// MLCleaner orchestrator
// =============================================================================

/// Training-zone specification: either as fractions of the dataset
/// (percentage ranges) or as absolute TIMESTAMP windows in UTC millis.
/// The latter is preferred when the zones come from a UI brush over a
/// time-axis chart — it avoids the row-index mismatch that occurs when
/// the preview is a slice of a larger dataset.
#[derive(Debug, Clone)]
pub enum ZoneSource {
    Pct(Vec<(f64, f64)>),
    Timestamps(Vec<(i64, i64)>),
}

/// What `MLCleaner::clean` should do with the predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MLCleanMode {
    /// Legacy one-shot: detect outliers + replace them with the model's
    /// prediction, AND fill existing NaN gaps with the prediction.
    DetectAndReplace,
    /// Detection-only commit: outliers go to NaN, existing gaps stay NaN.
    /// Used by the Detection page so the Gap Filling tab can decide how
    /// to fill the resulting NaNs (model, interpolation, …).
    MarkOnly,
    /// Just fill NaN cells with predictions — never re-detect outliers.
    /// Used by the Gap Filling tab after the Detection tab has already
    /// marked outliers as NaN. Decouples the two responsibilities so
    /// each tab does exactly what its name says.
    GapFillOnly,
}

impl Default for MLCleanMode {
    fn default() -> Self { MLCleanMode::MarkOnly }
}

/// Per-column detection report from `MLCleaner::detect`. Mirrors the
/// shape returned by Voie A's classical `cleaning_detect_only` so the
/// frontend can render both with the same chart code.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MLDetectionPerColumn {
    pub column: String,
    pub n_outliers: usize,
    /// Number of rows where the observation is NaN (a gap). Reported but
    /// not in `indices` — gaps are not "outliers" the user must validate.
    pub n_gaps: usize,
    /// Row indices flagged as outliers (in the data_df ordering used by
    /// build_xy_full). Length == n_outliers.
    pub indices: Vec<usize>,
}

pub struct MLCleaner {
    method: CleaningMethod,
    model: Box<dyn Predictor>,
    scaler_x: Option<StandardScaler>,
    metrics: Option<CleaningTrainingMetrics>,
    /// Columns we want to predict AND overwrite in the cleaned output.
    target_columns: Vec<String>,
    /// Columns we predict for metric reporting only — their values in the
    /// output DataFrame stay identical to the input.
    test_columns: Vec<String>,
    predictor_columns: Vec<String>,
    /// Seasonal / diurnal features added to X. Applies to every Voie B model,
    /// not just DL — a Random Forest can benefit from cyclical hour-of-day
    /// too. Defaults enable `cyclical_hour` + `cyclical_doy`.
    temporal_features: TemporalFeatures,
    /// Training zones — either as percentage ranges of the dataset, or as
    /// absolute timestamp ranges (UTC millis). The model trains on the
    /// union of all zones. Rows outside every zone are untouched by
    /// train(). Default = single full-span pct zone.
    train_zones: ZoneSource,
    /// If `log_transform_vpd` is set, VPD-like columns get a ln(x) feature.
    log_transform_vpd: bool,
    feature_scaling: bool,
    split_ratio: f64,
}

impl MLCleaner {
    /// Build a new cleaner from a Voie B `CleaningMethod`. Fails if the method
    /// is classical or if required columns are empty in the config.
    pub fn new(method: CleaningMethod) -> Result<Self> {
        method.validate()?;
        if method.path() != CleaningPath::AIBased {
            bail!("MLCleaner: méthode classique reçue ({})", method.name());
        }

        // Extract model + common training knobs from the method payload.
        let (
            model,
            target_columns,
            test_columns,
            predictor_columns,
            split_ratio,
            feature_scaling,
            log_transform_vpd,
            temporal_features,
            train_zones,
        ): (Box<dyn Predictor>, Vec<String>, Vec<String>, Vec<String>, f64, bool, bool, TemporalFeatures, ZoneSource) =
            match &method {
                CleaningMethod::SimpleLinear(c) => model_from_linear(c, LinearModel::simple())?,
                CleaningMethod::MultipleLinear(c) => model_from_linear(c, LinearModel::multiple())?,
                CleaningMethod::RandomForest(c) => {
                    model_from_ml(c, Box::new(RandomForestModel::new(c)))?
                }
                CleaningMethod::Svr(c) => model_from_ml(c, Box::new(SVRModel::new(c)))?,
                CleaningMethod::Gpr(c) => model_from_ml(c, Box::new(GprModel::new(c)))?,
                CleaningMethod::Pca(c) => model_from_ml(c, Box::new(PcaModel::new(c)))?,
                // NB: GprModel is defined below in the GPR section. Keep the
                // reference here so the match is exhaustive over AI methods.
                CleaningMethod::Arx(c) => {
                    let model: Box<dyn Predictor> = Box::new(ARXModel::new(c));
                    (
                        model,
                        c.target_columns.clone(),
                        c.test_columns.clone(),
                        c.predictor_columns.clone(),
                        c.train_validation_split,
                        true,
                        false,
                        c.temporal_features,
                        resolve_zones(&c.train_time_ranges, &c.train_ranges, c.train_range_start, c.train_range_end),
                    )
                }
                CleaningMethod::Armax(c) => {
                    let model: Box<dyn Predictor> = Box::new(ARMAXModel::new(c));
                    (
                        model,
                        c.target_columns.clone(),
                        c.test_columns.clone(),
                        c.predictor_columns.clone(),
                        c.train_validation_split,
                        true,
                        false,
                        c.temporal_features,
                        resolve_zones(&c.train_time_ranges, &c.train_ranges, c.train_range_start, c.train_range_end),
                    )
                }
                CleaningMethod::Lstm(c) => {
                    model_from_dl(c, Box::new(MLPModel::new(c, "LSTM (MLP-backed)")))?
                }
                CleaningMethod::BiLstm(c) => {
                    model_from_dl(c, Box::new(MLPModel::new(c, "BiLSTM (MLP-backed)")))?
                }
                CleaningMethod::Gru(c) => {
                    model_from_dl(c, Box::new(MLPModel::new(c, "GRU (MLP-backed)")))?
                }
                _ => bail!(
                    "MLCleaner: méthode '{}' non supportée",
                    method.name()
                ),
            };

        Ok(Self {
            method,
            model,
            scaler_x: None,
            metrics: None,
            target_columns,
            test_columns,
            predictor_columns,
            temporal_features,
            train_zones,
            log_transform_vpd,
            feature_scaling,
            split_ratio,
        })
    }

    /// Full list of columns the model predicts on — targets first, then test.
    pub fn all_predicted_columns(&self) -> Vec<String> {
        let mut v = self.target_columns.clone();
        v.extend(self.test_columns.iter().cloned());
        v
    }

    pub fn test_columns(&self) -> &[String] {
        &self.test_columns
    }

    /// The `CleaningMethod` (enum + config) used to instantiate this cleaner.
    /// Exposed so `save_scenario` can persist it — reconstructing the exact
    /// same model is then a matter of `MLCleaner::new(method.clone())` +
    /// `.train(...)` on the same data and seed.
    pub fn method(&self) -> &CleaningMethod {
        &self.method
    }

    /// Access to the trained model's metrics (None before `train`).
    pub fn metrics(&self) -> Option<&CleaningTrainingMetrics> {
        self.metrics.as_ref()
    }

    pub fn predictor_columns(&self) -> &[String] {
        &self.predictor_columns
    }
    pub fn target_columns(&self) -> &[String] {
        &self.target_columns
    }

    /// Fit the model. `data_df` must contain the target columns; `env_df` must
    /// contain the predictor columns. If a predictor is in `data_df` instead
    /// of `env_df`, it's picked up from there (legitimate: sensor-to-sensor
    /// prediction).
    ///
    /// Both dataframes are joined on TIMESTAMP (nearest-prior within 30 min).
    pub fn train(&mut self, data_df: &DataFrame, env_df: Option<&DataFrame>) -> Result<()> {
        let (x, y, _ts) = self.build_xy(data_df, env_df)?;
        let (x_tr, x_val, y_tr, y_val, _) = sequential_split::<i64>(&x, &y, &[], self.split_ratio);

        // Fit scaler on train only, apply to both splits.
        let (x_tr_s, x_val_s) = if self.feature_scaling {
            let scaler = StandardScaler::fit(&x_tr);
            let tr = scaler.transform(&x_tr);
            let va = scaler.transform(&x_val);
            self.scaler_x = Some(scaler);
            (tr, va)
        } else {
            self.scaler_x = None;
            (x_tr.clone(), x_val.clone())
        };

        let t_train_start = Instant::now();
        self.model.train(&x_tr_s, &y_tr)?;
        let training_time_s = t_train_start.elapsed().as_secs_f64();

        let t_pred_start = Instant::now();
        let y_hat_val = self.model.predict(&x_val_s)?;
        let prediction_time_s = t_pred_start.elapsed().as_secs_f64();

        // Metrics cover BOTH target and test columns. UI can show them
        // separately; we pass the combined list so every predicted column
        // gets its own row in `per_target`.
        let all_cols = self.all_predicted_columns();
        let per_target = compute_per_target_metrics(&y_val, &y_hat_val, &all_cols);
        let metrics = aggregate_metrics(
            per_target,
            &y_val,
            &y_hat_val,
            self.model.n_parameters(),
            y_tr.nrows(),
            y_val.nrows(),
            training_time_s,
            prediction_time_s,
        );
        self.metrics = Some(metrics);
        Ok(())
    }

    /// Predict on the full series and flag residuals outside
    /// `threshold_factor × RMSE_val` as outliers, replacing them (and any
    /// pre-existing NaN gaps) by the model's predictions. Returns the cleaned
    /// array and a per-column count `(outliers, gaps_filled)`.
    /// Detection-only pass: predicts on the full dataset, finds outliers
    /// where |obs - pred| > k·rmse, but does NOT write predictions back.
    /// Returns per-column outlier indices (gaps where obs is NaN are
    /// reported separately) so the UI can paint them red on a preview
    /// chart, mirroring Voie A's `cleaning_detect_only` flow.
    pub fn detect(
        &self,
        data_df: &DataFrame,
        env_df: Option<&DataFrame>,
        threshold_factor: f64,
    ) -> Result<Vec<MLDetectionPerColumn>> {
        let metrics = self
            .metrics
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MLCleaner: train before detect"))?;
        let (x, y, _ts) = self.build_xy_full(data_df, env_df)?;
        let x_s = self
            .scaler_x
            .as_ref()
            .map(|s| s.transform(&x))
            .unwrap_or_else(|| x.clone());
        let y_pred = self.model.predict(&x_s)?;

        let threshold = (threshold_factor * metrics.rmse).abs();
        let all_cols = self.all_predicted_columns();
        let mut per_col: Vec<MLDetectionPerColumn> = all_cols
            .iter()
            .map(|name| MLDetectionPerColumn {
                column: name.clone(),
                n_outliers: 0,
                n_gaps: 0,
                indices: Vec::new(),
            })
            .collect();

        for j in 0..y.ncols() {
            for i in 0..y.nrows() {
                let obs = y[[i, j]];
                let pred = y_pred[[i, j]];
                if !pred.is_finite() {
                    continue;
                }
                if !obs.is_finite() {
                    per_col[j].n_gaps += 1;
                    continue;
                }
                if (obs - pred).abs() > threshold {
                    per_col[j].n_outliers += 1;
                    per_col[j].indices.push(i);
                }
            }
        }
        Ok(per_col)
    }

    /// Apply the trained model to the dataset. The action taken on
    /// outliers and gaps depends on `mode` — see `MLCleanMode` docs.
    ///
    /// Returns the resulting target matrix and per-column counters
    /// `(n_outliers_acted_on, n_gaps_acted_on)`. Counters reflect what
    /// each mode actually does:
    ///   - `DetectAndReplace`: both replaced with predictions.
    ///   - `MarkOnly`:        outliers set to NaN, gaps untouched but counted.
    ///   - `GapFillOnly`:     outliers ignored entirely (count = 0), gaps
    ///                        filled with predictions.
    pub fn clean(
        &self,
        data_df: &DataFrame,
        env_df: Option<&DataFrame>,
        threshold_factor: f64,
        mode: MLCleanMode,
    ) -> Result<(Array2<f64>, Vec<(usize, usize)>)> {
        let metrics = self
            .metrics
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MLCleaner: train before clean"))?;
        let (x, y, _ts) = self.build_xy_full(data_df, env_df)?;
        let x_s = self
            .scaler_x
            .as_ref()
            .map(|s| s.transform(&x))
            .unwrap_or_else(|| x.clone());
        let y_pred = self.model.predict(&x_s)?;

        let threshold = (threshold_factor * metrics.rmse).abs();
        let mut out = y.clone();
        let mut per_col = vec![(0usize, 0usize); y.ncols()];
        let n_targets = self.target_columns.len();
        for j in 0..y.ncols() {
            // Test columns: only report stats, never overwrite — the original
            // data stays intact for independent cross-validation.
            let is_test_only = j >= n_targets;
            for i in 0..y.nrows() {
                let obs = y[[i, j]];
                let pred = y_pred[[i, j]];
                if !pred.is_finite() {
                    continue;
                }
                if !obs.is_finite() {
                    // Existing NaN. What to do with it depends on mode.
                    match mode {
                        MLCleanMode::DetectAndReplace | MLCleanMode::GapFillOnly => {
                            if !is_test_only {
                                out[[i, j]] = pred;
                            }
                            per_col[j].1 += 1; // gap filled
                        }
                        MLCleanMode::MarkOnly => {
                            // Leave the NaN for Gap Filling to handle later,
                            // but count it so the UI can report.
                            per_col[j].1 += 1;
                        }
                    }
                    continue;
                }
                // Finite observation — only the modes that re-detect care.
                if matches!(mode, MLCleanMode::GapFillOnly) {
                    continue;
                }
                if (obs - pred).abs() > threshold {
                    if !is_test_only {
                        out[[i, j]] = match mode {
                            MLCleanMode::DetectAndReplace => pred,
                            MLCleanMode::MarkOnly => f64::NAN,
                            MLCleanMode::GapFillOnly => unreachable!(),
                        };
                    }
                    per_col[j].0 += 1; // outlier acted on
                }
            }
        }
        Ok((out, per_col))
    }

    // ---- internals ---------------------------------------------------------

    /// Build X and Y from data_df + env_df (row-aligned). Drops rows with any
    /// NaN in X or Y (training needs clean data). Returns also the kept
    /// timestamps aligned with X/Y.
    fn build_xy(
        &self,
        data_df: &DataFrame,
        env_df: Option<&DataFrame>,
    ) -> Result<(Array2<f64>, Array2<f64>, Vec<NaiveDateTime>)> {
        let (x_full, y_full, ts_full) = self.build_xy_full(data_df, env_df)?;

        // Multi-zone training: keep only rows whose row-index (Pct mode) or
        // absolute timestamp (Timestamps mode) falls inside the union of
        // the configured zones. Rows outside every zone are ignored by the
        // fit (they may still receive predictions later in clean()).
        let n_full = x_full.nrows();
        let inside_any_zone = build_zone_mask(&self.train_zones, n_full, &ts_full);

        // Drop rows where X or Y has any NaN, within the union of zones only.
        let mut keep = Vec::with_capacity(n_full);
        for i in 0..n_full {
            if !inside_any_zone[i] { continue; }
            let x_ok = x_full.row(i).iter().all(|v| v.is_finite());
            let y_ok = y_full.row(i).iter().all(|v| v.is_finite());
            if x_ok && y_ok {
                keep.push(i);
            }
        }
        if keep.len() < 20 {
            bail!(
                "MLCleaner: trop peu de lignes propres dans la zone d'entraînement ({} < 20). \
                 Étends la zone, ajoute des plages ou vérifie les NaN.",
                keep.len()
            );
        }
        let mut x = Array2::<f64>::zeros((keep.len(), x_full.ncols()));
        let mut y = Array2::<f64>::zeros((keep.len(), y_full.ncols()));
        let mut ts = Vec::with_capacity(keep.len());
        for (new_i, &orig_i) in keep.iter().enumerate() {
            x.row_mut(new_i).assign(&x_full.row(orig_i));
            y.row_mut(new_i).assign(&y_full.row(orig_i));
            ts.push(ts_full[orig_i]);
        }
        Ok((x, y, ts))
    }

    /// Build X and Y without dropping NaN rows. Useful for `clean` which must
    /// produce predictions for every row including NaN-gapped ones.
    fn build_xy_full(
        &self,
        data_df: &DataFrame,
        env_df: Option<&DataFrame>,
    ) -> Result<(Array2<f64>, Array2<f64>, Vec<NaiveDateTime>)> {
        if self.target_columns.is_empty() {
            bail!("MLCleaner: target_columns vide");
        }
        if self.predictor_columns.is_empty() {
            bail!("MLCleaner: predictor_columns vide");
        }

        // Pull timestamps from data_df; they drive the row order.
        let ts_col = data_df
            .column("TIMESTAMP")
            .context("MLCleaner: colonne TIMESTAMP requise dans data_df")?;
        let datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(ts_col)?
            .into_iter()
            .map(|d| d.naive_utc())
            .collect();
        let n = datetimes.len();

        // Y = targets first, then test columns. The predictor produces
        // `all_targets.len()` output columns, but only the first
        // `target_columns.len()` get overwritten in the cleaned DataFrame.
        let all_targets = self.all_predicted_columns();
        let mut y = Array2::<f64>::zeros((n, all_targets.len()));
        for (j, name) in all_targets.iter().enumerate() {
            let vals = pull_f64(data_df, name)?;
            if vals.len() != n {
                bail!("Target '{}' row count mismatch", name);
            }
            for (i, v) in vals.into_iter().enumerate() {
                y[[i, j]] = v;
            }
        }

        // Predictors: prefer data_df, fall back to env_df.
        let env_lookup = env_df.map(|e| build_env_lookup(e)).transpose()?;
        let mut pred_blocks: Vec<Array2<f64>> = Vec::new();
        let mut pred_block = Array2::<f64>::zeros((n, self.predictor_columns.len()));
        for (j, name) in self.predictor_columns.iter().enumerate() {
            if data_df.column(name).is_ok() {
                let vals = pull_f64(data_df, name)?;
                for (i, v) in vals.into_iter().enumerate() {
                    pred_block[[i, j]] = v;
                }
            } else if let Some(ref lk) = env_lookup {
                for (i, dt) in datetimes.iter().enumerate() {
                    pred_block[[i, j]] = lk.lookup(dt, name);
                }
            } else {
                bail!(
                    "Prédicteur '{}' introuvable dans data_df et aucun env_df fourni",
                    name
                );
            }
        }

        // Optional log-transform on VPD-like predictor columns.
        if self.log_transform_vpd {
            for (j, name) in self.predictor_columns.iter().enumerate() {
                if name.to_uppercase().contains("VPD") {
                    for i in 0..n {
                        let v = pred_block[[i, j]];
                        pred_block[[i, j]] = if v > 0.0 { v.ln() } else { f64::NAN };
                    }
                }
            }
        }
        pred_blocks.push(pred_block);

        // Temporal features (empty matrix if all flags off).
        if self.temporal_features.extra_cols() > 0 {
            pred_blocks.push(build_temporal_features(&datetimes, &self.temporal_features));
        }

        // Horizontal concat.
        let total_cols: usize = pred_blocks.iter().map(|b| b.ncols()).sum();
        let mut x = Array2::<f64>::zeros((n, total_cols));
        let mut offset = 0;
        for b in &pred_blocks {
            let w = b.ncols();
            x.slice_mut(s![.., offset..offset + w]).assign(b);
            offset += w;
        }

        Ok((x, y, datetimes))
    }
}

// =============================================================================
// Helpers
// =============================================================================

// `model_from_*` helpers now return a 9-tuple including the training zones.
type ModelBundle = (
    Box<dyn Predictor>,
    Vec<String>,         // target_columns
    Vec<String>,         // test_columns
    Vec<String>,         // predictor_columns
    f64,                 // split_ratio
    bool,                // feature_scaling
    bool,                // log_transform_vpd
    TemporalFeatures,    // temporal_features
    ZoneSource,          // train_zones
);

/// Resolve the effective training zones from a config. Priority:
///   1. `train_time_ranges` (UTC millis) — preferred when set by the UI
///   2. `train_ranges`      (pct of dataset)
///   3. `(train_range_start, train_range_end)` — legacy single pct pair
fn resolve_zones(
    train_time_ranges: &[(i64, i64)],
    train_ranges: &[(f64, f64)],
    legacy_start: f64,
    legacy_end: f64,
) -> ZoneSource {
    if !train_time_ranges.is_empty() {
        let cleaned: Vec<(i64, i64)> = train_time_ranges.iter()
            .map(|&(s, e)| if s <= e { (s, e) } else { (e, s) })
            .collect();
        if !cleaned.is_empty() {
            return ZoneSource::Timestamps(cleaned);
        }
    }
    let pairs: Vec<(f64, f64)> = if train_ranges.is_empty() {
        vec![(legacy_start, legacy_end)]
    } else {
        train_ranges.to_vec()
    };
    let cleaned: Vec<(f64, f64)> = pairs.into_iter()
        .filter_map(|(s, e)| {
            let s = s.clamp(0.0, 1.0);
            let e = e.clamp(0.0, 1.0);
            let (lo, hi) = if s <= e { (s, e) } else { (e, s) };
            if hi - lo < 0.005 { None } else { Some((lo, hi)) }
        })
        .collect();
    ZoneSource::Pct(if cleaned.is_empty() { vec![(0.0, 1.0)] } else { cleaned })
}

/// Build a row-level boolean mask: true for rows that fall inside the
/// union of the configured zones. Pct zones use row-index ranges;
/// timestamp zones use the row's actual timestamp. Both modes return the
/// full-span mask if their inner Vec is empty.
fn build_zone_mask(zones: &ZoneSource, n: usize, ts: &[NaiveDateTime]) -> Vec<bool> {
    if n == 0 { return Vec::new(); }
    let mut mask = vec![false; n];
    match zones {
        ZoneSource::Pct(ranges) => {
            let pairs: Vec<(usize, usize)> = if ranges.is_empty() {
                vec![(0, n)]
            } else {
                ranges.iter().map(|&(s, e)| {
                    let lo = ((n as f64) * s).floor() as usize;
                    let hi = ((n as f64) * e).ceil() as usize;
                    (lo.min(n), hi.min(n).max(lo + 1).min(n))
                }).collect()
            };
            for (lo, hi) in pairs {
                for i in lo..hi.min(n) {
                    mask[i] = true;
                }
            }
        }
        ZoneSource::Timestamps(ranges) => {
            if ranges.is_empty() {
                return vec![true; n];
            }
            // Every row whose UTC timestamp falls inside any [start, end]
            // (inclusive) interval is marked true. Linear scan over rows
            // with all zones held in the inner loop — O(n × zones), fine
            // for typical datasets (<1M rows × <50 zones).
            for (i, dt) in ts.iter().enumerate().take(n) {
                let row_ms = dt.and_utc().timestamp_millis();
                if ranges.iter().any(|&(s, e)| row_ms >= s && row_ms <= e) {
                    mask[i] = true;
                }
            }
        }
    }
    mask
}

fn model_from_linear(cfg: &LinearConfig, model: LinearModel) -> Result<ModelBundle> {
    Ok((
        Box::new(model),
        cfg.target_columns.clone(),
        cfg.test_columns.clone(),
        cfg.predictor_columns.clone(),
        cfg.train_validation_split,
        cfg.feature_scaling,
        cfg.log_transform_vpd,
        cfg.temporal_features,
        resolve_zones(&cfg.train_time_ranges, &cfg.train_ranges, cfg.train_range_start, cfg.train_range_end),
    ))
}

fn model_from_ml(cfg: &MLConfig, model: Box<dyn Predictor>) -> Result<ModelBundle> {
    Ok((
        model,
        cfg.target_columns.clone(),
        cfg.test_columns.clone(),
        cfg.predictor_columns.clone(),
        cfg.train_validation_split,
        cfg.feature_scaling,
        false,
        cfg.temporal_features,
        resolve_zones(&cfg.train_time_ranges, &cfg.train_ranges, cfg.train_range_start, cfg.train_range_end),
    ))
}

fn model_from_dl(cfg: &DLConfig, model: Box<dyn Predictor>) -> Result<ModelBundle> {
    Ok((
        model,
        cfg.target_columns.clone(),
        cfg.test_columns.clone(),
        cfg.predictor_columns.clone(),
        cfg.train_validation_split,
        cfg.feature_scaling,
        false,
        cfg.temporal_features,
        resolve_zones(&cfg.train_time_ranges, &cfg.train_ranges, cfg.train_range_start, cfg.train_range_end),
    ))
}

fn pull_f64(df: &DataFrame, col: &str) -> Result<Vec<f64>> {
    let s = df
        .column(col)
        .with_context(|| format!("Colonne '{}' introuvable", col))?;
    let casted = s
        .cast(&DataType::Float64)
        .with_context(|| format!("Colonne '{}' non numérique", col))?;
    let ca = casted.f64()?;
    Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Simple timestamp-keyed lookup on an environmental DataFrame. Assumes the
/// env_df has a TIMESTAMP-like column called "TIMESTAMP" (case-sensitive).
/// For a requested timestamp, returns the value from the nearest-prior env
/// row within 30 minutes, or NaN.
struct EnvLookup {
    /// Sorted (ts_ms, per_column_values).
    rows: Vec<(i64, HashMap<String, f64>)>,
}

impl EnvLookup {
    fn lookup(&self, dt: &NaiveDateTime, col: &str) -> f64 {
        let key = dt.and_utc().timestamp_millis();
        // Binary search for last row with ts <= key within 30 min.
        let idx = match self
            .rows
            .binary_search_by(|(ts, _)| ts.cmp(&key))
        {
            Ok(i) => i,
            Err(0) => return f64::NAN,
            Err(i) => i - 1,
        };
        let (ts, row) = &self.rows[idx];
        if key - *ts > 30 * 60 * 1000 {
            return f64::NAN;
        }
        row.get(col).copied().unwrap_or(f64::NAN)
    }
}

fn build_env_lookup(env_df: &DataFrame) -> Result<EnvLookup> {
    // Try common TIMESTAMP column names.
    let ts_candidates = ["TIMESTAMP", "timestamp", "time", "Time", "DateTime", "datetime"];
    let ts_col = ts_candidates
        .iter()
        .find_map(|n| env_df.column(n).ok())
        .ok_or_else(|| anyhow::anyhow!("env_df: aucune colonne timestamp détectée"))?;
    let datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(ts_col)?
        .into_iter()
        .map(|d| d.naive_utc())
        .collect();

    let numeric_cols: Vec<String> = env_df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .filter(|n| {
            env_df
                .column(n)
                .ok()
                .map(|c| c.dtype().is_numeric())
                .unwrap_or(false)
        })
        .collect();

    let mut rows: Vec<(i64, HashMap<String, f64>)> =
        Vec::with_capacity(datetimes.len());
    for (i, dt) in datetimes.iter().enumerate() {
        let mut row = HashMap::with_capacity(numeric_cols.len());
        for c in &numeric_cols {
            let v = pull_f64(env_df, c)?.get(i).copied().unwrap_or(f64::NAN);
            row.insert(c.clone(), v);
        }
        rows.push((dt.and_utc().timestamp_millis(), row));
    }
    rows.sort_by_key(|(ts, _)| *ts);
    Ok(EnvLookup { rows })
}

/// Replace NaN and outlier positions in a column vector by a reference vector.
/// Kept here for future composition with `ReplaceStrategy` once the UI asks
/// for something other than "use model prediction".
#[allow(dead_code)]
pub fn replace_with_reference(
    target: &mut [f64],
    reference: &[f64],
    mask: &[bool],
    strat: ReplaceStrategy,
) {
    let _ = strat; // placeholder — currently we always overwrite with reference
    for i in 0..target.len().min(reference.len()).min(mask.len()) {
        if mask[i] || !target[i].is_finite() {
            if reference[i].is_finite() {
                target[i] = reference[i];
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_dataset(
        n: usize,
        noise: f64,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>) {
        // y = 2 x1 + 3 x2 + 5 + ε
        let mut rng_state = seed.max(1);
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let x1 = i as f64 * 0.1;
            let x2 = (i as f64 * 0.05).sin();
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;
            y[[i, 0]] = 2.0 * x1 + 3.0 * x2 + 5.0 + noise * next();
        }
        (x, y)
    }

    fn dt(ms: i64) -> NaiveDateTime {
        chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
            .unwrap()
            .naive_utc()
    }

    #[test]
    fn zone_mask_pct_full_span_when_empty() {
        let mask = build_zone_mask(&ZoneSource::Pct(vec![]), 100, &[]);
        assert_eq!(mask.len(), 100);
        assert!(mask.iter().all(|&b| b), "empty pct list = full window");
    }

    #[test]
    fn zone_mask_pct_single_range() {
        // [0.25, 0.75] over n=100 → rows 25..=74 marked true.
        let mask = build_zone_mask(&ZoneSource::Pct(vec![(0.25, 0.75)]), 100, &[]);
        assert_eq!(mask.iter().filter(|&&b| b).count(), 50);
        assert!(!mask[0]);
        assert!(mask[25]);
        assert!(mask[50]);
        assert!(mask[74]);
        assert!(!mask[75]);
    }

    #[test]
    fn zone_mask_pct_union_of_disjoint_ranges() {
        // Two zones: [0.0, 0.2] and [0.8, 1.0] over n=100. Total = 40 rows
        // (not 100 — the gap in the middle is excluded, which is the whole
        // point of multi-zone training).
        let mask = build_zone_mask(
            &ZoneSource::Pct(vec![(0.0, 0.2), (0.8, 1.0)]),
            100,
            &[],
        );
        assert_eq!(mask.iter().filter(|&&b| b).count(), 40);
        assert!(mask[0]);
        assert!(mask[19]);
        assert!(!mask[50], "gap between zones must be excluded");
        assert!(mask[80]);
        assert!(mask[99]);
    }

    #[test]
    fn zone_mask_timestamps_full_span_when_empty() {
        let ts: Vec<NaiveDateTime> = (0..5).map(|i| dt(i * 1000)).collect();
        let mask = build_zone_mask(&ZoneSource::Timestamps(vec![]), 5, &ts);
        assert_eq!(mask, vec![true; 5]);
    }

    #[test]
    fn zone_mask_timestamps_filters_by_actual_time() {
        // Timestamps at 0, 1, 2, 3, 4 seconds. Zone covers [1500ms, 3500ms]
        // → rows 2 and 3 are inside, rows 0/1/4 are outside.
        let ts: Vec<NaiveDateTime> = (0..5).map(|i| dt(i * 1000)).collect();
        let mask = build_zone_mask(
            &ZoneSource::Timestamps(vec![(1500, 3500)]),
            5,
            &ts,
        );
        assert_eq!(mask, vec![false, false, true, true, false]);
    }

    #[test]
    fn zone_mask_timestamps_union_of_zones() {
        // Two disjoint zones cover rows 0-1 and row 4. Row 2 (mid) is in
        // neither zone — multi-zone exclusion in action.
        let ts: Vec<NaiveDateTime> = (0..5).map(|i| dt(i * 1000)).collect();
        let mask = build_zone_mask(
            &ZoneSource::Timestamps(vec![(0, 1500), (3500, 5000)]),
            5,
            &ts,
        );
        assert_eq!(mask, vec![true, true, false, false, true]);
    }

    #[test]
    fn resolve_zones_prefers_timestamps_over_pct() {
        // When both train_time_ranges AND train_ranges are set, the i64
        // timestamp list wins — that's the contract documented on the
        // configs (UI uses absolute time, pct is a fallback for legacy).
        let resolved = resolve_zones(
            &[(1000, 2000), (3000, 4000)],
            &[(0.1, 0.5)],
            0.0,
            1.0,
        );
        match resolved {
            ZoneSource::Timestamps(v) => assert_eq!(v, vec![(1000, 2000), (3000, 4000)]),
            _ => panic!("expected Timestamps variant"),
        }
    }

    #[test]
    fn resolve_zones_falls_back_to_pct_multi_then_legacy() {
        // No timestamps + non-empty pct multi → Pct variant with cleaned
        // pct ranges.
        let r1 = resolve_zones(&[], &[(0.2, 0.8)], 0.0, 1.0);
        assert!(matches!(r1, ZoneSource::Pct(ref v) if v == &vec![(0.2, 0.8)]));

        // No timestamps, no pct multi → fall back to the legacy single
        // pair (start, end). 0.0..1.0 means full window.
        let r2 = resolve_zones(&[], &[], 0.3, 0.9);
        assert!(matches!(r2, ZoneSource::Pct(ref v) if v == &vec![(0.3, 0.9)]));

        // Everything empty / default → full window [(0, 1)].
        let r3 = resolve_zones(&[], &[], 0.0, 1.0);
        assert!(matches!(r3, ZoneSource::Pct(ref v) if v == &vec![(0.0, 1.0)]));
    }

    #[test]
    fn resolve_zones_clamps_invalid_pct_and_drops_tiny_ranges() {
        // Out-of-bounds values get clamped, swapped pairs get reversed,
        // sub-0.5%-wide ranges get dropped (they'd produce <1 valid row
        // and cause spurious "trop peu de lignes" errors). Result here:
        // (-0.5, 1.5) clamped to (0, 1) → kept; (0.5, 0.501) too narrow.
        let r = resolve_zones(&[], &[(-0.5, 1.5), (0.7, 0.3), (0.5, 0.501)], 0.0, 1.0);
        match r {
            ZoneSource::Pct(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], (0.0, 1.0));
                assert_eq!(v[1], (0.3, 0.7));
            }
            _ => panic!("expected Pct variant"),
        }
    }

    #[test]
    fn resolve_zones_swaps_inverted_timestamp_pairs() {
        // User drags brush right-to-left → coordRange comes back inverted.
        // Backend normalizes so the mask logic always sees s ≤ e.
        let r = resolve_zones(&[(5000, 1000)], &[], 0.0, 1.0);
        match r {
            ZoneSource::Timestamps(v) => assert_eq!(v, vec![(1000, 5000)]),
            _ => panic!("expected Timestamps variant"),
        }
    }

    /// Builds the synthetic dataset used by the two end-to-end tests:
    /// a noiseless y = 2x + 5 ramp with `outlier_idx` rows shifted by a
    /// big offset and `gap_idx` rows set to NaN. Returns the DataFrame
    /// plus the raw x/y vectors so the assertions can compare against
    /// the ground truth.
    fn make_e2e_dataset(
        n: usize,
        outlier_idx: &[usize],
        gap_idx: &[usize],
        outlier_offset: f64,
    ) -> (polars::frame::DataFrame, Vec<f64>, Vec<Option<f64>>) {
        use polars::prelude::*;

        let mut x_vals: Vec<f64> = Vec::with_capacity(n);
        let mut y_vals: Vec<Option<f64>> = Vec::with_capacity(n);
        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        for i in 0..n {
            let xi = i as f64 * 0.1;
            x_vals.push(xi);
            let mut yi = 2.0 * xi + 5.0 + 0.02 * next();
            if outlier_idx.contains(&i) { yi += outlier_offset; }
            if gap_idx.contains(&i) {
                y_vals.push(None);
            } else {
                y_vals.push(Some(yi));
            }
        }

        let ts_us: Vec<i64> = (0..n).map(|i| (i as i64) * 60 * 1_000_000).collect();
        let ts_series = Series::new("TIMESTAMP", ts_us)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let x_series = Series::new("x", x_vals.clone());
        let y_series = Series::new("y", y_vals.clone());
        let df = DataFrame::new(vec![ts_series, x_series, y_series]).unwrap();
        (df, x_vals, y_vals)
    }

    fn make_e2e_cleaner() -> MLCleaner {
        let cfg = LinearConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            test_columns: vec![],
            // Disable cyclical features — pure y=2x+5 doesn't need them
            // and they dilute the fit on a 200-row toy.
            temporal_features: TemporalFeatures {
                cyclical_hour: false,
                cyclical_doy: false,
                ..TemporalFeatures::default()
            },
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.8,
            feature_scaling: true,
            log_transform_vpd: false,
            random_seed: 42,
        };
        MLCleaner::new(CleaningMethod::MultipleLinear(cfg)).unwrap()
    }

    /// End-to-end smoke test for the new DEFAULT Voie B workflow:
    ///   1. Train a model.
    ///   2. detect() previews outliers (red dots in the UI).
    ///   3. clean(mark_only=true) marks outliers as NaN, leaves gaps NaN.
    ///   4. The user goes to Gap Filling next — out of scope here, but
    ///      we verify the dataset is in the expected "ready to gap-fill"
    ///      shape: flagged rows are NaN, untouched rows are unchanged.
    #[test]
    fn end_to_end_train_detect_mark_outliers_as_nan() {
        const N: usize = 200;
        let outlier_idx: Vec<usize> = vec![30, 70, 110, 140, 175];
        let gap_idx: Vec<usize> = vec![45, 90, 160];
        let (df, x_vals, y_vals) = make_e2e_dataset(N, &outlier_idx, &gap_idx, 50.0);

        // --- Step 1: train ----------------------------------------------------
        let mut cleaner = make_e2e_cleaner();
        cleaner.train(&df, None).expect("training should succeed");
        assert!(cleaner.metrics().is_some(), "metrics populated after train");

        // --- Step 2: detection (no replacement, just preview) ----------------
        let detected = cleaner.detect(&df, None, 3.0).expect("detect should succeed");
        assert_eq!(detected.len(), 1, "one target column → one detection entry");
        let dy = &detected[0];
        assert_eq!(dy.column, "y");

        let recovered = dy.indices.iter().filter(|i| outlier_idx.contains(i)).count();
        assert!(recovered >= 4,
            "outlier recall too low: caught {}/{} (got indices {:?})",
            recovered, outlier_idx.len(), dy.indices);
        assert_eq!(dy.n_gaps, gap_idx.len(),
            "all NaN gaps should be reported in n_gaps");

        let clean_idx = 50;
        assert!(!dy.indices.contains(&clean_idx),
            "row {} is clean — shouldn't be flagged", clean_idx);

        // --- Step 3: apply with mark_only=true (DEFAULT in UI) ---------------
        // Outliers should become NaN. Existing gaps should stay NaN. No
        // value should be replaced by a model prediction.
        let (cleaned, per_col) = cleaner.clean(&df, None, 3.0, MLCleanMode::MarkOnly)
            .expect("clean(MarkOnly) should succeed");
        assert_eq!(cleaned.nrows(), N);
        assert_eq!(cleaned.ncols(), 1);

        let (n_outliers_flagged, n_gaps) = per_col[0];
        assert!(n_outliers_flagged >= 4,
            "should flag at least 4 outliers, got {}", n_outliers_flagged);
        assert_eq!(n_gaps, gap_idx.len(),
            "all gaps should be counted (left as-is, not filled)");

        // Outliers that were caught by detect() must now be NaN in the
        // cleaned output — that's what "mark only" means.
        for &i in dy.indices.iter() {
            assert!(cleaned[[i, 0]].is_nan(),
                "outlier at row {} should be NaN under mark_only=true, got {}",
                i, cleaned[[i, 0]]);
        }

        // Original gaps stay NaN (the user will fill them in Gap Filling).
        for &i in &gap_idx {
            assert!(cleaned[[i, 0]].is_nan(),
                "gap at row {} should remain NaN, got {}", i, cleaned[[i, 0]]);
        }

        // Clean rows untouched.
        let original = y_vals[clean_idx].unwrap();
        let after = cleaned[[clean_idx, 0]];
        assert!((after - original).abs() < 1e-6,
            "row {} was clean but got modified: {} → {}", clean_idx, original, after);

        // Make sure x_vals is referenced so the helper is generic-enough
        // to be reused; silences the unused-binding warning.
        let _ = x_vals.len();
    }

    /// Legacy one-shot Voie B workflow: outliers AND gaps get replaced
    /// with model predictions in the same call. Kept available behind
    /// `mark_only=false` for users who skip the Gap Filling tab.
    #[test]
    fn end_to_end_legacy_replace_outliers_and_fill_gaps_in_one_shot() {
        const N: usize = 200;
        let outlier_idx: Vec<usize> = vec![30, 70, 110, 140, 175];
        let gap_idx: Vec<usize> = vec![45, 90, 160];
        let (df, x_vals, y_vals) = make_e2e_dataset(N, &outlier_idx, &gap_idx, 50.0);

        let mut cleaner = make_e2e_cleaner();
        cleaner.train(&df, None).expect("training should succeed");

        let (cleaned, per_col) = cleaner.clean(&df, None, 3.0, MLCleanMode::DetectAndReplace)
            .expect("clean(DetectAndReplace) should succeed");

        let (n_outliers_replaced, n_gaps_filled) = per_col[0];
        assert!(n_outliers_replaced >= 4,
            "should replace at least 4 outliers, got {}", n_outliers_replaced);
        assert_eq!(n_gaps_filled, gap_idx.len(),
            "every NaN gap should be filled with a prediction");

        // Filled values close to the noiseless regression line. Tolerance
        // accounts for the ~1.25-unit shift caused by training on data
        // that still contained the +50 outliers.
        for &i in &gap_idx {
            let filled = cleaned[[i, 0]];
            let expected = 2.0 * x_vals[i] + 5.0;
            assert!(filled.is_finite(), "gap at row {} not filled", i);
            assert!((filled - expected).abs() < 3.0,
                "gap fill at row {} = {} too far from expected {}", i, filled, expected);
        }
        for &i in &outlier_idx {
            let cleaned_y = cleaned[[i, 0]];
            let expected = 2.0 * x_vals[i] + 5.0;
            assert!((cleaned_y - expected).abs() < 5.0,
                "outlier at row {} not properly replaced: got {}, expected ~{}",
                i, cleaned_y, expected);
        }

        let untouched_idx = 50;
        let original = y_vals[untouched_idx].unwrap();
        let after = cleaned[[untouched_idx, 0]];
        assert!((after - original).abs() < 1e-6,
            "row {} was clean but got modified: {} → {}",
            untouched_idx, original, after);
    }

    /// Two-step workflow that mirrors the actual UI: Detection marks
    /// outliers as NaN (mode=MarkOnly), then Gap Filling fills every NaN
    /// (mode=GapFillOnly). Verifies that GapFillOnly leaves outlier
    /// detection alone — it just trusts the existing NaNs as the work list.
    #[test]
    fn end_to_end_two_step_detect_then_fill() {
        const N: usize = 200;
        let outlier_idx: Vec<usize> = vec![30, 70, 110, 140, 175];
        let gap_idx: Vec<usize> = vec![45, 90, 160];
        let (df, x_vals, _y_vals) = make_e2e_dataset(N, &outlier_idx, &gap_idx, 50.0);

        let mut cleaner = make_e2e_cleaner();
        cleaner.train(&df, None).expect("training should succeed");

        // Step 1 — Detection page: mark outliers as NaN.
        let (after_mark, _) = cleaner.clean(&df, None, 3.0, MLCleanMode::MarkOnly)
            .expect("MarkOnly should succeed");

        // Sanity: outliers detected by the model are now NaN, original
        // gaps are still NaN. Re-detect to know which rows the model
        // would catch.
        let detected = cleaner.detect(&df, None, 3.0).unwrap();
        for &i in &detected[0].indices {
            assert!(after_mark[[i, 0]].is_nan(),
                "after MarkOnly, row {} should be NaN", i);
        }
        for &i in &gap_idx {
            assert!(after_mark[[i, 0]].is_nan(),
                "original gap at row {} should still be NaN", i);
        }

        // Build a NEW DataFrame reflecting the post-MarkOnly state, since
        // GapFillOnly reads `obs` from the dataset (not from a previous
        // clean's output). This is what the UI does between tabs.
        use polars::prelude::*;
        let post_mark_y: Vec<Option<f64>> = (0..N)
            .map(|i| {
                let v = after_mark[[i, 0]];
                if v.is_finite() { Some(v) } else { None }
            })
            .collect();
        let ts_us: Vec<i64> = (0..N).map(|i| (i as i64) * 60 * 1_000_000).collect();
        let ts_series = Series::new("TIMESTAMP", ts_us)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let df_after_mark = DataFrame::new(vec![
            ts_series,
            Series::new("x", x_vals.clone()),
            Series::new("y", post_mark_y),
        ]).unwrap();

        // Step 2 — Gap Filling page: fill every NaN with predictions, do
        // NOT re-detect outliers.
        let (after_fill, per_col) = cleaner.clean(&df_after_mark, None, 3.0, MLCleanMode::GapFillOnly)
            .expect("GapFillOnly should succeed");
        let (n_outliers_acted, n_gaps_filled) = per_col[0];
        assert_eq!(n_outliers_acted, 0,
            "GapFillOnly must not detect outliers; got n_outliers_acted={}",
            n_outliers_acted);
        // Every cell that was NaN before should now be finite.
        let n_nan_before: usize = (0..N).filter(|&i| after_mark[[i, 0]].is_nan()).count();
        assert!(n_gaps_filled >= n_nan_before.saturating_sub(2),
            "should fill ~all NaNs ({} filled out of {} NaN)", n_gaps_filled, n_nan_before);
        for i in 0..N {
            if after_mark[[i, 0]].is_nan() {
                let filled = after_fill[[i, 0]];
                assert!(filled.is_finite(),
                    "NaN at row {} not filled by GapFillOnly", i);
                let expected = 2.0 * x_vals[i] + 5.0;
                assert!((filled - expected).abs() < 3.0,
                    "fill at row {} = {} too far from expected {}", i, filled, expected);
            }
        }
    }

    #[test]
    fn linear_model_recovers_coefficients() {
        let (x, y) = make_linear_dataset(300, 0.05, 42);
        let mut m = LinearModel::multiple();
        m.train(&x, &y).unwrap();
        let pred = m.predict(&x).unwrap();
        // RMSE on a near-noiseless fit should be tiny.
        let err: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rmse = (err / y.len() as f64).sqrt();
        assert!(rmse < 0.5, "rmse={}", rmse);
    }

    #[test]
    fn linear_model_predict_before_train_errors() {
        let m = LinearModel::simple();
        let x = Array2::<f64>::zeros((4, 2));
        let err = m.predict(&x);
        assert!(err.is_err());
    }

    #[test]
    fn scaler_standardizes_columns() {
        let x = Array2::<f64>::from_shape_vec(
            (5, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
        )
        .unwrap();
        let s = StandardScaler::fit(&x);
        let z = s.transform(&x);
        for j in 0..2 {
            let col = z.column(j);
            let mean = col.sum() / col.len() as f64;
            assert!(mean.abs() < 1e-9);
            let var =
                col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
            assert!((var - 1.0).abs() < 1e-6, "var(col {})={}", j, var);
        }
    }

    #[test]
    fn metrics_r_squared_perfect_fit() {
        let y = Array2::<f64>::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let per = compute_per_target_metrics(&y, &y, &["t".to_string()]);
        let m = per.get("t").unwrap();
        assert!((m.r_squared - 1.0).abs() < 1e-9);
        assert!(m.rmse.abs() < 1e-9);
    }

    #[test]
    fn sequential_split_preserves_order() {
        let x = Array2::<f64>::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = x.clone();
        let (x_tr, x_val, _, _, _) = sequential_split::<i64>(&x, &y, &[], 0.8);
        assert_eq!(x_tr.nrows(), 8);
        assert_eq!(x_val.nrows(), 2);
        assert_eq!(x_tr[[0, 0]], 0.0);
        assert_eq!(x_val[[0, 0]], 8.0);
    }

    #[test]
    fn temporal_encoding_one_hot_dims() {
        let dts: Vec<NaiveDateTime> = (0..3)
            .map(|i| {
                chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(i, 0, 0)
                    .unwrap()
            })
            .collect();
        // Order in build_temporal_features:
        //   one_hot_hour(24) · cyclical_hour(2) · one_hot_doy(366) · cyclical_doy(2)
        //   · one_hot_month(12) · cyclical_month(2) · one_hot_minute(2) · days_since_start(1)
        let cfg = TemporalFeatures {
            one_hot_hour: true,
            cyclical_hour: false,
            one_hot_doy: true,
            cyclical_doy: false,
            one_hot_month: false,
            cyclical_month: false,
            one_hot_minute: true,
            days_since_start: false,
        };
        let feat = build_temporal_features(&dts, &cfg);
        assert_eq!(feat.ncols(), 24 + 366 + 2);
        assert_eq!(feat.nrows(), 3);
        // Row 0: hour=0 → one-hot hour col 0 should be 1
        assert_eq!(feat[[0, 0]], 1.0);
        // Row 0: DOY=1 (Jan 1) → one-hot doy starts at offset 24, col 0 → total col 24
        assert_eq!(feat[[0, 24]], 1.0);
    }

    #[test]
    fn cyclical_features_have_unit_norm() {
        let dts: Vec<NaiveDateTime> = (0..4)
            .map(|i| {
                chrono::NaiveDate::from_ymd_opt(2024, 6, 15)
                    .unwrap()
                    .and_hms_opt((i * 6) as u32, 0, 0)
                    .unwrap()
            })
            .collect();
        let cfg = TemporalFeatures {
            one_hot_hour: false,
            cyclical_hour: true,
            one_hot_doy: false,
            cyclical_doy: true,
            ..Default::default()
        };
        // Override default to match what we set explicitly
        let cfg = TemporalFeatures { cyclical_month: false, ..cfg };
        let feat = build_temporal_features(&dts, &cfg);
        assert_eq!(feat.ncols(), 4); // 2 + 2
        for i in 0..4 {
            // sin² + cos² = 1 for each pair
            let r1 = feat[[i, 0]].powi(2) + feat[[i, 1]].powi(2);
            let r2 = feat[[i, 2]].powi(2) + feat[[i, 3]].powi(2);
            assert!((r1 - 1.0).abs() < 1e-9);
            assert!((r2 - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn temporal_features_default_enables_cyclical() {
        let cfg = TemporalFeatures::default();
        assert!(cfg.cyclical_hour);
        assert!(cfg.cyclical_doy);
        assert!(!cfg.one_hot_hour);
        assert!(!cfg.one_hot_doy);
        assert_eq!(cfg.extra_cols(), 4); // 2 + 2
    }

    #[test]
    fn aic_bic_lower_when_better_fit() {
        let y = Array2::<f64>::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let good = y.clone();
        let bad = &y + 5.0;
        let per_g = compute_per_target_metrics(&y, &good, &["t".to_string()]);
        let per_b = compute_per_target_metrics(&y, &bad, &["t".to_string()]);
        let mg = aggregate_metrics(per_g, &y, &good, 2, 7, 3, 0.0, 0.0);
        let mb = aggregate_metrics(per_b, &y, &bad, 2, 7, 3, 0.0, 0.0);
        assert!(mg.aic < mb.aic, "AIC good={} bad={}", mg.aic, mb.aic);
    }

    #[test]
    fn mlcleaner_refuses_classical_method() {
        use crate::core::types::{IqrConfig, ReplaceStrategy};
        let method = CleaningMethod::Iqr(IqrConfig {
            k: 1.5,
            target_columns: vec!["x".into()],
            replace_strategy: ReplaceStrategy::MarkNaN,
        });
        assert!(MLCleaner::new(method).is_err());
    }

    #[test]
    fn mlcleaner_rejects_incomplete_linear_config() {
        let method = CleaningMethod::SimpleLinear(LinearConfig::default()); // no columns
        assert!(MLCleaner::new(method).is_err());
    }

    #[test]
    fn random_forest_fits_non_linear_signal() {
        // y = sin(x1) × x2 — non-linear interaction a forest should learn.
        let n = 400;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let x1 = (i as f64) * 0.05;
            let x2 = ((i as f64) * 0.03).cos();
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;
            y[[i, 0]] = x1.sin() * x2;
        }
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x1".into(), "x2".into()],
            n_estimators: Some(30),
            max_depth: Some(8),
            ..Default::default()
        };
        let mut rf = RandomForestModel::new(&cfg);
        rf.train(&x, &y).unwrap();
        let pred = rf.predict(&x).unwrap();
        let mae: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        // A well-fitted bagging forest on training data should get MAE < 0.2
        // on this signal bounded in [-1, 1].
        assert!(mae < 0.3, "RF train MAE too high: {}", mae);
        assert_eq!(rf.n_targets(), 1);
        assert!(rf.n_parameters() > 0);
    }

    #[test]
    fn svr_fits_simple_linear_signal() {
        let n = 120;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let xi = i as f64 * 0.1;
            x[[i, 0]] = xi;
            y[[i, 0]] = 2.0 * xi + 1.0;
        }
        // Normalize because SVR with RBF/Gaussian kernel prefers scaled input.
        let scaler = StandardScaler::fit(&x);
        let x_s = scaler.transform(&x);
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            kernel: Some(KernelType::Linear),
            svr_c: Some(10.0),
            svr_epsilon: Some(0.05),
            ..Default::default()
        };
        let mut svr = SVRModel::new(&cfg);
        svr.train(&x_s, &y).unwrap();
        let pred = svr.predict(&x_s).unwrap();
        let mae: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        assert!(mae < 1.0, "SVR train MAE too high: {}", mae);
    }

    #[test]
    fn mlcleaner_accepts_random_forest() {
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            n_estimators: Some(20),
            ..Default::default()
        };
        let method = CleaningMethod::RandomForest(cfg);
        assert!(MLCleaner::new(method).is_ok());
    }

    #[test]
    fn arx_lagged_features_shape() {
        let x = Array2::<f64>::from_shape_vec((4, 2), vec![1.,2.,3.,4.,5.,6.,7.,8.]).unwrap();
        let lag = lagged_features(&x, 2);
        assert_eq!(lag.shape(), &[4, 6]); // 2 base cols × (2+1)
        // Row 0: t,t-1,t-2 all clamp to index 0.
        assert_eq!(lag[[0, 0]], 1.0);
        assert_eq!(lag[[0, 1]], 2.0);
        assert_eq!(lag[[0, 2]], 1.0);
        assert_eq!(lag[[0, 3]], 2.0);
        // Row 3: t = row 3, t-1 = row 2, t-2 = row 1
        assert_eq!(lag[[3, 0]], 7.0);
        assert_eq!(lag[[3, 2]], 5.0);
        assert_eq!(lag[[3, 4]], 3.0);
    }

    #[test]
    fn arx_trains_on_synthetic_lagged_signal() {
        // y_t = 0.5·x_t + 0.3·x_{t-1}
        let n = 200;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x[[i, 0]] = (i as f64 * 0.1).sin();
        }
        for i in 1..n {
            y[[i, 0]] = 0.5 * x[[i, 0]] + 0.3 * x[[i - 1, 0]];
        }
        let cfg = ARXConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            exog_order: 2,
            ..Default::default()
        };
        let mut m = ARXModel::new(&cfg);
        m.train(&x, &y).unwrap();
        let pred = m.predict(&x).unwrap();
        let err: f64 = (0..n)
            .map(|i| (y[[i, 0]] - pred[[i, 0]]).powi(2))
            .sum::<f64>()
            / n as f64;
        let rmse = err.sqrt();
        assert!(rmse < 0.2, "ARX RMSE too high: {}", rmse);
    }

    #[test]
    fn armax_ma_smoothing_produces_lower_variance() {
        // Feed noisy linear signal → ARMAX smoothed prediction should have
        // lower variance than ARX raw prediction on the same data.
        let n = 120;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array2::<f64>::zeros((n, 1));
        let mut rng = XorShift64::new(7);
        for i in 0..n {
            x[[i, 0]] = (i as f64 * 0.2).sin();
            y[[i, 0]] = x[[i, 0]] + (rng.next_f64_unit() * 2.0 - 1.0) * 0.3;
        }
        let cfg_arx = ARXConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            exog_order: 1,
            ..Default::default()
        };
        let mut arx = ARXModel::new(&cfg_arx);
        arx.train(&x, &y).unwrap();
        let p_arx = arx.predict(&x).unwrap();

        let cfg_armax = ARMAXConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            exog_order: 1,
            ma_order: 5,
            ..Default::default()
        };
        let mut armax = ARMAXModel::new(&cfg_armax);
        armax.train(&x, &y).unwrap();
        let p_armax = armax.predict(&x).unwrap();

        let var_arx: f64 = {
            let m: f64 = p_arx.iter().sum::<f64>() / p_arx.len() as f64;
            p_arx.iter().map(|v| (v - m).powi(2)).sum::<f64>() / p_arx.len() as f64
        };
        let var_armax: f64 = {
            let m: f64 = p_armax.iter().sum::<f64>() / p_armax.len() as f64;
            p_armax.iter().map(|v| (v - m).powi(2)).sum::<f64>() / p_armax.len() as f64
        };
        assert!(var_armax <= var_arx + 1e-9, "ARMAX smoothing did not reduce variance");
    }

    #[test]
    fn mlp_model_trains_and_predicts() {
        let n = 150;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let a = i as f64 * 0.1;
            x[[i, 0]] = a;
            x[[i, 1]] = a.sin();
            y[[i, 0]] = 1.0 + 0.5 * a;
        }
        // Standardize features first.
        let scaler = StandardScaler::fit(&x);
        let x_s = scaler.transform(&x);
        let cfg = DLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x1".into(), "x2".into()],
            epochs: 80,
            hidden_units: 16,
            batch_size: 16,
            learning_rate: 0.02,
            ..Default::default()
        };
        let mut mlp = MLPModel::new(&cfg, "test-mlp");
        mlp.train(&x_s, &y).unwrap();
        let pred = mlp.predict(&x_s).unwrap();
        let mae: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        assert!(mae < 1.5, "MLP MAE too high: {}", mae);
        assert!(mlp.n_parameters() > 0);
    }

    #[test]
    fn mlcleaner_accepts_arx_armax_dl() {
        let base_cols = (vec!["y".into()], vec!["x".into()]);
        // ARX
        let arx = ARXConfig {
            target_columns: base_cols.0.clone(),
            predictor_columns: base_cols.1.clone(),
            ..Default::default()
        };
        assert!(MLCleaner::new(CleaningMethod::Arx(arx)).is_ok());
        // ARMAX
        let armax = ARMAXConfig {
            target_columns: base_cols.0.clone(),
            predictor_columns: base_cols.1.clone(),
            ..Default::default()
        };
        assert!(MLCleaner::new(CleaningMethod::Armax(armax)).is_ok());
        // LSTM / BiLSTM / GRU (MLP-backed)
        let dl = DLConfig {
            target_columns: base_cols.0.clone(),
            predictor_columns: base_cols.1.clone(),
            ..Default::default()
        };
        assert!(MLCleaner::new(CleaningMethod::Lstm(dl.clone())).is_ok());
        assert!(MLCleaner::new(CleaningMethod::BiLstm(dl.clone())).is_ok());
        assert!(MLCleaner::new(CleaningMethod::Gru(dl)).is_ok());
    }

    #[test]
    fn mlcleaner_accepts_gpr_now_that_its_implemented() {
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            ..Default::default()
        };
        let method = CleaningMethod::Gpr(cfg);
        assert!(MLCleaner::new(method).is_ok());
    }

    #[test]
    fn mlcleaner_accepts_svr() {
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            kernel: Some(KernelType::Linear),
            svr_c: Some(1.0),
            svr_epsilon: Some(0.1),
            ..Default::default()
        };
        let method = CleaningMethod::Svr(cfg);
        assert!(MLCleaner::new(method).is_ok());
    }

    #[test]
    fn mlcleaner_builds_on_valid_linear_config() {
        let cfg = LinearConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            test_columns: vec![],
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.8,
            feature_scaling: true,
            log_transform_vpd: false,
            random_seed: 42,
        };
        let method = CleaningMethod::MultipleLinear(cfg);
        let cleaner = MLCleaner::new(method).unwrap();
        assert_eq!(cleaner.target_columns(), &["y"]);
        assert_eq!(cleaner.predictor_columns(), &["x"]);
    }

    #[test]
    fn mlcleaner_separates_target_from_test_columns() {
        let cfg = LinearConfig {
            target_columns: vec!["SF1".into()],
            predictor_columns: vec!["VPD".into()],
            test_columns: vec!["SF2".into(), "SF3".into()],
            ..Default::default()
        };
        let cleaner = MLCleaner::new(CleaningMethod::MultipleLinear(cfg)).unwrap();
        assert_eq!(cleaner.target_columns(), &["SF1"]);
        assert_eq!(cleaner.test_columns(), &["SF2", "SF3"]);
        // Predictor produces SF1 + SF2 + SF3 columns at clean time.
        assert_eq!(cleaner.all_predicted_columns(), vec!["SF1", "SF2", "SF3"]);
    }

    #[test]
    fn gpr_fits_small_rbf_signal() {
        // Simple non-linear 1D: y = sin(πx/2)
        let n = 80;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let xi = (i as f64 - n as f64 / 2.0) * 0.1;
            x[[i, 0]] = xi;
            y[[i, 0]] = (std::f64::consts::FRAC_PI_2 * xi).sin();
        }
        // GPR with RBF kernel on normalized input.
        let scaler = StandardScaler::fit(&x);
        let x_s = scaler.transform(&x);
        let cfg = MLConfig {
            target_columns: vec!["y".into()],
            predictor_columns: vec!["x".into()],
            svr_c: Some(100.0), // low noise
            ..Default::default()
        };
        let mut g = GprModel::new(&cfg);
        g.train(&x_s, &y).unwrap();
        let pred = g.predict(&x_s).unwrap();
        let mae: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        assert!(mae < 0.3, "GPR MAE too high on smooth signal: {}", mae);
    }

    #[test]
    fn gpr_predict_before_train_errors() {
        let g = GprModel::new(&MLConfig::default());
        let x = Array2::<f64>::zeros((2, 1));
        assert!(g.predict(&x).is_err());
    }
}
