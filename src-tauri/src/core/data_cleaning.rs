use anyhow::Result;

/// Detect outliers using the IQR (Interquartile Range) method.
///
/// Returns a boolean mask where `true` indicates an outlier.
pub fn detect_outliers_iqr(data: &[f64], multiplier: f64) -> Vec<bool> {
    if data.is_empty() {
        return Vec::new();
    }

    // Sort only FINITE values — NaN/inf make the f64 comparator violate total
    // order, which makes Rust's sort panic. NaN cells are never flagged.
    let mut sorted: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < 4 {
        return vec![false; data.len()];
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[(3 * n) / 4];
    let iqr = q3 - q1;

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    data.iter()
        .map(|&val| val.is_finite() && (val < lower_bound || val > upper_bound))
        .collect()
}

/// Detect outliers using the Z-Score method.
///
/// Returns a boolean mask where `true` indicates an outlier.
pub fn detect_outliers_zscore(data: &[f64], threshold: f64) -> Vec<bool> {
    if data.is_empty() {
        return Vec::new();
    }

    let finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return vec![false; data.len()];
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let variance = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return vec![false; data.len()];
    }

    data.iter()
        .map(|&val| val.is_finite() && ((val - mean) / std_dev).abs() > threshold)
        .collect()
}

/// Detect outliers using the MAD (Median Absolute Deviation) method.
pub fn detect_outliers_mad(data: &[f64], threshold: f64) -> Vec<bool> {
    if data.is_empty() {
        return Vec::new();
    }

    // Finite-only: NaN/inf break the f64 sort comparator (total-order panic)
    // and poison the median. NaN cells are never flagged.
    let mut sorted: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![false; data.len()];
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut abs_devs: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if abs_devs.len() % 2 == 0 {
        (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
    } else {
        abs_devs[abs_devs.len() / 2]
    };

    // Scale factor for consistency with standard deviation (for normal distribution)
    let mad_scaled = mad * 1.4826;

    if mad_scaled == 0.0 {
        return vec![false; data.len()];
    }

    data.iter()
        .map(|&val| val.is_finite() && ((val - median) / mad_scaled).abs() > threshold)
        .collect()
}

/// Detect outliers using a rolling Z-Score method (sliding window).
pub fn detect_outliers_rolling_zscore(data: &[f64], window: usize, threshold: f64) -> Vec<bool> {
    let n = data.len();
    if n == 0 { return Vec::new(); }
    let half = window / 2;
    let mut mask = vec![false; n];
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window_data: Vec<f64> = data[start..end].iter().copied().filter(|v| v.is_finite()).collect();
        if window_data.len() < 3 { continue; }
        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
        let var = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window_data.len() as f64;
        let std = var.sqrt();
        if std > 0.0 {
            mask[i] = ((data[i] - mean) / std).abs() > threshold;
        }
    }
    mask
}

/// Detect outliers using a simple Isolation Forest approximation.
/// Scores each point by how often it lands in small partitions across random trees.
pub fn detect_outliers_isolation_forest(data: &[f64], contamination: f64) -> Vec<bool> {
    let n = data.len();
    if n == 0 { return Vec::new(); }
    let n_trees = 100usize;
    let subsample = (256usize).min(n);
    let mut scores = vec![0.0f64; n];
    // Deterministic PRNG (xorshift64)
    let mut seed: u64 = 0xdeadbeefcafe1234;
    let mut rng = || -> u64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    for _ in 0..n_trees {
        // Sample subsample indices
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..subsample {
            let j = i + (rng() as usize % (n - i));
            indices.swap(i, j);
        }
        let sample: Vec<f64> = indices[..subsample].iter().map(|&i| data[i]).collect();
        // Build path lengths for full data against this tree
        for (point_idx, &val) in data.iter().enumerate() {
            scores[point_idx] += isolation_path_length(&sample, val, 0, &mut rng);
        }
    }
    // Normalize scores
    let avg_score: f64 = scores.iter().sum::<f64>() / n as f64;
    if avg_score == 0.0 { return vec![false; n]; }
    let anomaly_scores: Vec<f64> = scores.iter().map(|s| s / avg_score).collect();
    // Mark top `contamination` fraction as outliers. Sort FINITE scores only —
    // NaN data points produce NaN scores that would panic the f64 sort.
    let mut sorted: Vec<f64> = anomaly_scores.iter().copied().filter(|s| s.is_finite()).collect();
    if sorted.is_empty() { return vec![false; n]; }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold_idx = ((1.0 - contamination) * sorted.len() as f64) as usize;
    let threshold = sorted[threshold_idx.min(sorted.len() - 1)];
    data.iter()
        .zip(anomaly_scores.iter())
        .map(|(&v, &s)| v.is_finite() && s.is_finite() && s >= threshold)
        .collect()
}

fn isolation_path_length(sample: &[f64], val: f64, depth: usize, rng: &mut impl FnMut() -> u64) -> f64 {
    let n = sample.len();
    if n <= 1 || depth >= 10 { return depth as f64; }
    let min = sample.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = sample.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 { return depth as f64; }
    let split = min + (rng() as f64 / u64::MAX as f64) * (max - min);
    let left: Vec<f64> = sample.iter().cloned().filter(|&x| x < split).collect();
    let right: Vec<f64> = sample.iter().cloned().filter(|&x| x >= split).collect();
    if val < split {
        isolation_path_length(&left, val, depth + 1, rng)
    } else {
        isolation_path_length(&right, val, depth + 1, rng)
    }
}

/// Fill gaps using linear interpolation.
///
/// `mask` is a boolean slice where `true` indicates a value that should be replaced
/// (e.g., an outlier). Those positions are treated as missing and interpolated.
pub fn fill_gaps_linear(data: &[f64], mask: &[bool]) -> Vec<f64> {
    let mut filled = data.to_vec();

    // Set masked values to NaN for interpolation
    for (i, &is_masked) in mask.iter().enumerate() {
        if is_masked {
            filled[i] = f64::NAN;
        }
    }

    // Interpolate NaN gaps
    for i in 0..filled.len() {
        if filled[i].is_nan() {
            let mut prev_idx: Option<usize> = None;
            let mut next_idx: Option<usize> = None;

            for j in (0..i).rev() {
                if !filled[j].is_nan() {
                    prev_idx = Some(j);
                    break;
                }
            }

            for j in (i + 1)..filled.len() {
                if !filled[j].is_nan() {
                    next_idx = Some(j);
                    break;
                }
            }

            match (prev_idx, next_idx) {
                (Some(p), Some(n)) => {
                    let ratio = (i - p) as f64 / (n - p) as f64;
                    filled[i] = filled[p] + ratio * (filled[n] - filled[p]);
                }
                (Some(p), None) => {
                    filled[i] = filled[p];
                }
                (None, Some(n)) => {
                    filled[i] = filled[n];
                }
                (None, None) => {
                    filled[i] = 0.0;
                }
            }
        }
    }

    filled
}

/// Fill gaps using a moving average window.
#[allow(dead_code)]
pub fn fill_gaps_moving_average(data: &[f64], mask: &[bool], window_size: usize) -> Result<Vec<f64>> {
    let mut filled = data.to_vec();

    for (i, &is_masked) in mask.iter().enumerate() {
        if is_masked {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(data.len());

            let window_values: Vec<f64> = (start..end)
                .filter(|&j| !mask[j])
                .map(|j| data[j])
                .collect();

            if !window_values.is_empty() {
                filled[i] = window_values.iter().sum::<f64>() / window_values.len() as f64;
            }
        }
    }

    Ok(filled)
}

// =============================================================================
// CLEANING v2 — high-level API dispatching on `CleaningMethod`
// =============================================================================
// This section adds orchestrated cleaning on top of the existing primitives.
// The old single-function callers (`detect_outliers_iqr`, etc.) remain
// untouched so legacy paths keep working. New callers (scenarios, UI wired
// to `CleaningMethod`) should go through `apply_cleaning_method`.

use std::collections::HashMap;

use chrono::{NaiveDate, NaiveDateTime};
use polars::prelude::{ChunkedArray, DataFrame, DataType, Float64Type, Series};

use crate::core::timestamp_utils::ts_col_to_datetimes;
use crate::core::types::{
    CleaningColumnReport, CleaningMethod, CleaningReport, IqrConfig, IsolationForestConfig,
    MadConfig, ReplaceStrategy, ReverseDetectionConfig, RollingWindowConfig, ZScoreConfig,
};

// =============================================================================
// Gap filling — column-wise classical fill (no detection involved)
// =============================================================================

/// Methods exposed to the UI for filling NaN holes in a column. Mirrors
/// `ReplaceStrategy` but framed around the user's mental model: "I have
/// holes — pick how to bridge them."
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "params")]
pub enum GapFillMethod {
    LinearInterpolate,
    ForwardFill,
    BackwardFill,
    /// Centered rolling mean. Window size in samples (>= 1).
    RollingMean { window: usize },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct GapFillReport {
    pub method: String,
    pub n_total: usize,
    pub total_gaps_before: usize,
    pub total_gaps_after: usize,
    pub total_filled: usize,
    pub per_column: HashMap<String, GapFillColumnReport>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GapFillColumnReport {
    pub n_gaps_before: usize,
    pub n_gaps_after: usize,
    pub n_filled: usize,
    pub longest_gap_before: usize,
}

/// Apply a classical gap-fill method to the listed columns of `df`. Existing
/// NaN holes are filled in place; non-NaN values are preserved. Returns the
/// updated DataFrame and a per-column report so the UI can show how many
/// gaps were bridged.
pub fn fill_gaps_classical(
    df: &DataFrame,
    columns: &[String],
    method: GapFillMethod,
) -> anyhow::Result<(DataFrame, GapFillReport)> {
    let cols = resolve_targets(df, columns)?;
    let mut out = df.clone();
    let mut report = GapFillReport::default();
    report.n_total = df.height();
    report.method = method_name(&method).to_string();

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let nan_mask: Vec<bool> = raw.iter().map(|v| !v.is_finite()).collect();
        // The fill primitives interpret `mask[i] || !data[i].is_finite()` as
        // "needs filling", so an empty mask is fine — they still fill NaNs.
        let zero_mask = vec![false; raw.len()];
        let filled = match method {
            GapFillMethod::LinearInterpolate => fill_gaps_linear(&raw, &zero_mask),
            GapFillMethod::ForwardFill => forward_fill(&raw, &zero_mask),
            GapFillMethod::BackwardFill => backward_fill(&raw, &zero_mask),
            GapFillMethod::RollingMean { window } => {
                rolling_mean_fill(&raw, &zero_mask, window.max(1))
            }
        };

        let n_gaps_before = nan_mask.iter().filter(|&&b| b).count();
        let n_gaps_after = filled.iter().filter(|v| !v.is_finite()).count();
        let n_filled = n_gaps_before.saturating_sub(n_gaps_after);
        let longest_gap_before = longest_run_true(&nan_mask);

        inject_column(&mut out, col_name, &filled)?;

        report.total_gaps_before += n_gaps_before;
        report.total_gaps_after += n_gaps_after;
        report.total_filled += n_filled;
        report.per_column.insert(
            col_name.clone(),
            GapFillColumnReport {
                n_gaps_before,
                n_gaps_after,
                n_filled,
                longest_gap_before,
            },
        );
    }

    Ok((out, report))
}

fn method_name(m: &GapFillMethod) -> &'static str {
    match m {
        GapFillMethod::LinearInterpolate => "LinearInterpolate",
        GapFillMethod::ForwardFill => "ForwardFill",
        GapFillMethod::BackwardFill => "BackwardFill",
        GapFillMethod::RollingMean { .. } => "RollingMean",
    }
}

fn longest_run_true(mask: &[bool]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &b in mask {
        if b {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

/// Per-column outlier indices for a method, without applying any replacement.
///
/// Used by the UI to show a "validate detection" preview before the user
/// commits to a destructive cleaning. Returns indices (not masks) keyed by
/// column name; an empty Vec means no outliers were flagged for that column.
pub fn detect_outliers_only(
    df: &DataFrame,
    method: &CleaningMethod,
) -> anyhow::Result<HashMap<String, Vec<usize>>> {
    method.validate()?;
    // Re-use apply_cleaning_method by forcing replace_strategy = MarkNaN, then
    // diff input vs output: a position becomes NaN in the cleaned dataframe
    // iff it was a finite outlier (or it was already NaN, which we filter out).
    let probe = with_marknan_strategy(method.clone());
    let (cleaned_df, _report) = apply_cleaning_method(df, &probe)?;

    let mut out = HashMap::new();
    let cols = collect_target_columns(method, df)?;
    for col in &cols {
        let raw = pull_f64(df, col)?;
        let clean = pull_f64(&cleaned_df, col)?;
        let mut indices = Vec::new();
        for i in 0..raw.len() {
            let raw_finite = raw[i].is_finite();
            let clean_nan = !clean[i].is_finite();
            // Outlier = was finite, became NaN.
            if raw_finite && clean_nan {
                indices.push(i);
            }
        }
        out.insert(col.clone(), indices);
    }
    Ok(out)
}

/// Returns the same method with replace_strategy switched to MarkNaN where
/// applicable, so the apply path produces NaN holes we can diff against.
fn with_marknan_strategy(method: CleaningMethod) -> CleaningMethod {
    match method {
        CleaningMethod::Iqr(mut c) => {
            c.replace_strategy = ReplaceStrategy::MarkNaN;
            CleaningMethod::Iqr(c)
        }
        CleaningMethod::ZScore(mut c) => {
            c.replace_strategy = ReplaceStrategy::MarkNaN;
            CleaningMethod::ZScore(c)
        }
        CleaningMethod::Mad(mut c) => {
            c.replace_strategy = ReplaceStrategy::MarkNaN;
            CleaningMethod::Mad(c)
        }
        // RollingWindow / ReverseDetection / IsolationForest already MarkNaN.
        // ClassicalPipeline: recurse. AbsoluteBounds: already MarkNaN.
        CleaningMethod::ClassicalPipeline(steps) => {
            let mapped = steps.into_iter().map(with_marknan_strategy).collect();
            CleaningMethod::ClassicalPipeline(mapped)
        }
        other => other,
    }
}

/// Resolve the columns a classical method writes to, so we know which to diff.
fn collect_target_columns(
    method: &CleaningMethod,
    df: &DataFrame,
) -> anyhow::Result<Vec<String>> {
    match method {
        CleaningMethod::Iqr(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::ZScore(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::Mad(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::RollingWindow(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::ReverseDetection(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::IsolationForest(c) => resolve_targets(df, &c.target_columns),
        CleaningMethod::AbsoluteBounds { target_columns, .. } => resolve_targets(df, target_columns),
        CleaningMethod::ClassicalPipeline(steps) => {
            let mut all: Vec<String> = Vec::new();
            for s in steps {
                for c in collect_target_columns(s, df)? {
                    if !all.contains(&c) {
                        all.push(c);
                    }
                }
            }
            Ok(all)
        }
        other => anyhow::bail!(
            "{} est une méthode Voie B — la prévisualisation de détection n'est pas applicable",
            other.name()
        ),
    }
}

/// Apply any `CleaningMethod` to a DataFrame.
///
/// Returns the cleaned DataFrame (same schema as input) and a `CleaningReport`
/// aggregating outlier counts and per-column before/after statistics.
/// AI-based methods are rejected — route them through `MLCleaner` instead.
pub fn apply_cleaning_method(
    df: &DataFrame,
    method: &CleaningMethod,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    method.validate()?;
    match method {
        CleaningMethod::ClassicalPipeline(steps) => run_pipeline(df, steps),
        CleaningMethod::AbsoluteBounds {
            min,
            max,
            target_columns,
        } => apply_absolute_bounds(df, target_columns, *min, *max),
        CleaningMethod::Iqr(c) => apply_iqr(df, c),
        CleaningMethod::ZScore(c) => apply_zscore(df, c),
        CleaningMethod::Mad(c) => apply_mad(df, c),
        CleaningMethod::RollingWindow(c) => apply_rolling_window(df, c),
        CleaningMethod::ReverseDetection(c) => apply_reverse_detection(df, c),
        CleaningMethod::IsolationForest(c) => apply_isolation_forest_multivariate(df, c),
        other => anyhow::bail!(
            "{} est une méthode Voie B — utilise MLCleaner (apply_cleaning_method ne route que les méthodes classiques)",
            other.name()
        ),
    }
}

fn run_pipeline(
    df: &DataFrame,
    steps: &[CleaningMethod],
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let mut current = df.clone();
    let mut total = CleaningReport::default();
    total.n_total = df.height();
    for step in steps {
        let (new_df, report) = apply_cleaning_method(&current, step)?;
        total.n_outliers_detected += report.n_outliers_detected;
        total.n_outliers_replaced += report.n_outliers_replaced;
        total.n_gaps_filled += report.n_gaps_filled;
        for (k, v) in report.per_column {
            // Keep the most recent per-column snapshot (each step overwrites).
            total.per_column.insert(k, v);
        }
        total.method_chain.extend(report.method_chain);
        current = new_df;
    }
    total.pct_data_modified = pct_modified(total.n_outliers_replaced, total.n_total);
    Ok((current, total))
}

// ----------------------------------------------------------------------------
// Absolute bounds
// ----------------------------------------------------------------------------

pub fn apply_absolute_bounds(
    df: &DataFrame,
    target_columns: &[String],
    min: f64,
    max: f64,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, target_columns)?;
    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report.method_chain.push(format!("AbsoluteBounds({min}, {max})"));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let mask: Vec<bool> = raw
            .iter()
            .map(|v| !v.is_finite() || *v < min || *v > max)
            .collect();
        let cleaned = replace_values(&raw, &mask, ReplaceStrategy::MarkNaN);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

// ----------------------------------------------------------------------------
// IQR / ZScore / MAD — each wraps an existing primitive + replacement policy
// ----------------------------------------------------------------------------

pub fn apply_iqr(
    df: &DataFrame,
    cfg: &IqrConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report.method_chain.push(format!("IQR(k={})", cfg.k));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let mask = detect_outliers_iqr(&raw, cfg.k);
        let cleaned = replace_values(&raw, &mask, cfg.replace_strategy);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

pub fn apply_zscore(
    df: &DataFrame,
    cfg: &ZScoreConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report
        .method_chain
        .push(format!("ZScore(t={})", cfg.threshold));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let mask = detect_outliers_zscore(&raw, cfg.threshold);
        let cleaned = replace_values(&raw, &mask, cfg.replace_strategy);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

pub fn apply_mad(
    df: &DataFrame,
    cfg: &MadConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report.method_chain.push(format!("MAD(t={})", cfg.threshold));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let mask = detect_outliers_mad(&raw, cfg.threshold);
        let cleaned = replace_values(&raw, &mask, cfg.replace_strategy);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

// ----------------------------------------------------------------------------
// Rolling window (timestamp-aware two-pointer with Welford's online variance)
// ----------------------------------------------------------------------------

pub fn apply_rolling_window(
    df: &DataFrame,
    cfg: &RollingWindowConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let ts_col = df
        .column("TIMESTAMP")
        .map_err(|_| anyhow::anyhow!("RollingWindow: colonne TIMESTAMP requise dans la DataFrame"))?;
    let datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(ts_col)?
        .into_iter()
        .map(|d| d.naive_utc())
        .collect();

    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report.method_chain.push(format!(
        "RollingWindow(w={}h, high={}, low={})",
        cfg.window_hours, cfg.high_var_threshold, cfg.low_var_threshold
    ));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let mask = rolling_window_variance_mask(
            &raw,
            &datetimes,
            cfg.window_hours,
            cfg.high_var_threshold,
            cfg.low_var_threshold,
        );
        // Rolling window always marks as NaN so subsequent steps can either
        // fill or drop — we don't impose a replacement here.
        let cleaned = replace_values(&raw, &mask, ReplaceStrategy::MarkNaN);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

/// Two-pointer timestamp-indexed rolling window. Flags points whose centred
/// window has σ > high_threshold (excessive noise) or σ < low_threshold
/// (stuck signal). Assumes `ts` is sorted ascending, which TTDprocess already
/// enforces at import time.
fn rolling_window_variance_mask(
    values: &[f64],
    ts: &[NaiveDateTime],
    window_hours: f64,
    high_threshold: f64,
    low_threshold: f64,
) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![false; n];
    if n < 3 || ts.len() != n {
        return mask;
    }
    let half_window_s = (window_hours * 3600.0 / 2.0) as i64;

    let mut left = 0usize;
    let mut right = 0usize;
    for i in 0..n {
        let t_c = ts[i].and_utc().timestamp();
        let t_lo = t_c - half_window_s;
        let t_hi = t_c + half_window_s;
        while right < n && ts[right].and_utc().timestamp() <= t_hi {
            right += 1;
        }
        while left < right && ts[left].and_utc().timestamp() < t_lo {
            left += 1;
        }
        let slice = &values[left..right];
        let std = welford_std(slice);
        if slice.iter().filter(|v| v.is_finite()).count() < 3 {
            continue;
        }
        if std > high_threshold || std < low_threshold {
            mask[i] = true;
        }
    }
    mask
}

fn welford_std(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for &x in values {
        if !x.is_finite() {
            continue;
        }
        count += 1;
        let delta = x - mean;
        mean += delta / count as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    if count < 2 {
        0.0
    } else {
        (m2 / (count - 1) as f64).sqrt()
    }
}

// ----------------------------------------------------------------------------
// Reverse detection (days where the daily max sits far below the nearest
// neighbours — typically jour/nuit inverted by a datalogger glitch)
// ----------------------------------------------------------------------------

pub fn apply_reverse_detection(
    df: &DataFrame,
    cfg: &ReverseDetectionConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let ts_col = df
        .column("TIMESTAMP")
        .map_err(|_| anyhow::anyhow!("ReverseDetection: colonne TIMESTAMP requise"))?;
    let datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(ts_col)?
        .into_iter()
        .map(|d| d.naive_utc())
        .collect();

    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = df.height();
    report.method_chain.push(format!(
        "ReverseDetection(nbrs={}, ratio={})",
        cfg.num_neighbors, cfg.ratio
    ));

    for col_name in &cols {
        let raw = pull_f64(df, col_name)?;
        let bad_days = detect_reversed_days(&raw, &datetimes, cfg.num_neighbors, cfg.ratio);
        let bad_set: std::collections::HashSet<NaiveDate> = bad_days.into_iter().collect();

        let mask: Vec<bool> = datetimes
            .iter()
            .map(|dt| bad_set.contains(&dt.date()))
            .collect();
        let cleaned = replace_values(&raw, &mask, ReplaceStrategy::MarkNaN);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(&raw, &cleaned, &mask));
    }

    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

/// Find days whose local daily maximum is < `ratio` × max of the
/// `num_neighbors` points on either side of it — a cheap proxy for "the
/// real maximum sits outside this day", which happens when the datalogger
/// swaps night/day or drops a chunk of records.
fn detect_reversed_days(
    values: &[f64],
    datetimes: &[NaiveDateTime],
    num_neighbors: usize,
    ratio: f64,
) -> Vec<NaiveDate> {
    let n = values.len();
    if n == 0 || datetimes.len() != n {
        return Vec::new();
    }
    // Group (day, max_value, argmax_idx).
    let mut day_max: HashMap<NaiveDate, (f64, usize)> = HashMap::new();
    for (i, dt) in datetimes.iter().enumerate() {
        let v = values[i];
        if !v.is_finite() {
            continue;
        }
        let d = dt.date();
        day_max
            .entry(d)
            .and_modify(|(m, idx)| {
                if v > *m {
                    *m = v;
                    *idx = i;
                }
            })
            .or_insert((v, i));
    }

    let mut reversed = Vec::new();
    for (day, (max_val, arg_idx)) in day_max {
        let lo = arg_idx.saturating_sub(num_neighbors);
        let hi = (arg_idx + num_neighbors + 1).min(n);
        let neighbour_max = values[lo..hi]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        if !neighbour_max.is_finite() {
            continue;
        }
        if max_val < ratio * neighbour_max {
            reversed.push(day);
        }
    }
    reversed
}

// ----------------------------------------------------------------------------
// Multivariate Isolation Forest
// ----------------------------------------------------------------------------

pub fn apply_isolation_forest_multivariate(
    df: &DataFrame,
    cfg: &IsolationForestConfig,
) -> anyhow::Result<(DataFrame, CleaningReport)> {
    let cols = resolve_targets(df, &cfg.target_columns)?;
    let n = df.height();
    if n == 0 {
        return Ok((df.clone(), CleaningReport::default()));
    }

    // Stack target columns into (n, d) matrix. NaN rows are marked unsampleable.
    let d = cols.len();
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut row_usable = vec![true; n];
    for col in &cols {
        let v = pull_f64(df, col)?;
        for (i, x) in v.iter().enumerate() {
            if !x.is_finite() {
                row_usable[i] = false;
            }
        }
        matrix.push(v);
    }

    // Flatten to row-major (n x d) for efficient tree traversal.
    let row = |i: usize| -> Vec<f64> { (0..d).map(|j| matrix[j][i]).collect() };

    // Score every row (finite rows only; NaN rows get max score → flagged).
    let scores = multivariate_isolation_scores(
        &matrix,
        &row_usable,
        cfg.n_estimators,
        cfg.max_samples.unwrap_or(256).min(n),
        cfg.random_seed,
    );

    // Flag top `contamination` fraction as outliers.
    let k = ((scores.len() as f64) * cfg.contamination).ceil() as usize;
    let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    // total_cmp = a TOTAL order over f64 (handles NaN without panicking); NaN
    // sorts as the largest, so NaN rows stay flagged as the comment intends.
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let flagged: std::collections::HashSet<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();

    let _ = row; // keep for future debugging hooks

    // Apply: any row flagged → mark NaN in all target columns.
    let mut out = df.clone();
    let mut report = CleaningReport::default();
    report.n_total = n;
    report.method_chain.push(format!(
        "IsolationForest(c={}, trees={}, seed={})",
        cfg.contamination, cfg.n_estimators, cfg.random_seed
    ));

    for (j, col_name) in cols.iter().enumerate() {
        let raw = &matrix[j];
        let mask: Vec<bool> = (0..n).map(|i| flagged.contains(&i) || !row_usable[i]).collect();
        let cleaned = replace_values(raw, &mask, ReplaceStrategy::MarkNaN);
        inject_column(&mut out, col_name, &cleaned)?;

        let n_out = mask.iter().filter(|&&b| b).count();
        report.n_outliers_detected += n_out;
        report.n_outliers_replaced += n_out;
        report
            .per_column
            .insert(col_name.clone(), build_report(raw, &cleaned, &mask));
    }
    report.pct_data_modified = pct_modified(report.n_outliers_replaced, report.n_total);
    Ok((out, report))
}

/// Build `n_trees` random partition trees on subsamples of size `sub`,
/// return the average path length for each row across trees. Longer paths
/// ⇒ deeper in the tree ⇒ harder to isolate ⇒ less anomalous (lower score).
/// Anomaly score = 2^(−E(h(x)) / c(sub)).
fn multivariate_isolation_scores(
    columns: &[Vec<f64>],
    usable: &[bool],
    n_trees: usize,
    sub: usize,
    seed: u64,
) -> Vec<f64> {
    let d = columns.len();
    let n = columns.first().map(|c| c.len()).unwrap_or(0);
    if n == 0 || d == 0 || sub < 2 {
        return vec![0.0; n];
    }
    let max_depth = (sub as f64).log2().ceil() as usize;
    let mut path_len_sum = vec![0.0_f64; n];
    let mut rng_state = seed.max(1);
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _ in 0..n_trees {
        // Sample `sub` usable indices (reservoir-style).
        let mut sample: Vec<usize> = Vec::with_capacity(sub);
        let usable_idx: Vec<usize> = (0..n).filter(|&i| usable[i]).collect();
        if usable_idx.len() < 2 {
            continue;
        }
        for i in 0..sub.min(usable_idx.len()) {
            let j = (next_u64() as usize) % (usable_idx.len() - i);
            sample.push(usable_idx[i + j]);
        }
        if sample.len() < 2 {
            continue;
        }
        // Build tree with explicit stack (depth-limited).
        struct Node {
            indices: Vec<usize>,
            depth: usize,
        }
        let mut stack = vec![Node {
            indices: sample.clone(),
            depth: 0,
        }];
        // Evaluate every row i against this tree (not just the sample).
        let mut row_depth = vec![max_depth as f64; n];
        while let Some(Node { indices, depth }) = stack.pop() {
            if indices.len() <= 1 || depth >= max_depth {
                for &idx in &indices {
                    row_depth[idx] = depth as f64 + c_factor(indices.len());
                }
                continue;
            }
            let feat = (next_u64() as usize) % d;
            let vals: Vec<f64> = indices.iter().map(|&i| columns[feat][i]).collect();
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (hi - lo).abs() < 1e-12 {
                for &idx in &indices {
                    row_depth[idx] = depth as f64 + c_factor(indices.len());
                }
                continue;
            }
            let split = lo + (next_u64() as f64 / u64::MAX as f64) * (hi - lo);
            let (left, right): (Vec<usize>, Vec<usize>) = indices
                .into_iter()
                .partition(|&i| columns[feat][i] < split);
            stack.push(Node {
                indices: left,
                depth: depth + 1,
            });
            stack.push(Node {
                indices: right,
                depth: depth + 1,
            });
        }
        for i in 0..n {
            path_len_sum[i] += row_depth[i];
        }
    }

    // Anomaly score per row.
    let c_sub = c_factor(sub);
    let scale = if c_sub > 0.0 { c_sub } else { 1.0 };
    (0..n)
        .map(|i| {
            if !usable[i] {
                1.0 // NaN rows → maximum anomaly
            } else {
                let avg_path = path_len_sum[i] / n_trees as f64;
                2f64.powf(-avg_path / scale)
            }
        })
        .collect()
}

fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    2.0 * (nf.ln() + 0.5772156649_f64) - 2.0 * (nf - 1.0) / nf
}

// ----------------------------------------------------------------------------
// Replacement dispatcher (ReplaceStrategy)
// ----------------------------------------------------------------------------

fn replace_values(data: &[f64], mask: &[bool], strat: ReplaceStrategy) -> Vec<f64> {
    match strat {
        ReplaceStrategy::MarkNaN => {
            data.iter()
                .zip(mask)
                .map(|(v, &m)| if m { f64::NAN } else { *v })
                .collect()
        }
        ReplaceStrategy::LinearInterpolate => fill_gaps_linear(data, mask),
        ReplaceStrategy::ForwardFill => forward_fill(data, mask),
        ReplaceStrategy::RollingMean(w) => rolling_mean_fill(data, mask, w.max(1)),
    }
}

fn forward_fill(data: &[f64], mask: &[bool]) -> Vec<f64> {
    let mut out = data.to_vec();
    let mut last: Option<f64> = None;
    for i in 0..out.len() {
        let is_bad = mask[i] || !out[i].is_finite();
        if !is_bad {
            last = Some(out[i]);
        } else if let Some(v) = last {
            out[i] = v;
        } else {
            out[i] = f64::NAN;
        }
    }
    out
}

/// Mirror of `forward_fill` running right-to-left. NaN/masked positions take
/// the value of the next finite point; positions after the last finite point
/// remain NaN.
pub fn backward_fill(data: &[f64], mask: &[bool]) -> Vec<f64> {
    let mut out = data.to_vec();
    let mut next: Option<f64> = None;
    for i in (0..out.len()).rev() {
        let is_bad = mask[i] || !out[i].is_finite();
        if !is_bad {
            next = Some(out[i]);
        } else if let Some(v) = next {
            out[i] = v;
        } else {
            out[i] = f64::NAN;
        }
    }
    out
}

fn rolling_mean_fill(data: &[f64], mask: &[bool], window: usize) -> Vec<f64> {
    let mut out = data.to_vec();
    let half = window / 2;
    for i in 0..out.len() {
        let is_bad = mask[i] || !out[i].is_finite();
        if !is_bad {
            continue;
        }
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(out.len());
        let vals: Vec<f64> = (lo..hi)
            .filter(|&j| !mask[j] && data[j].is_finite())
            .map(|j| data[j])
            .collect();
        out[i] = if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
    }
    out
}

// ----------------------------------------------------------------------------
// Report building
// ----------------------------------------------------------------------------

fn build_report(raw: &[f64], cleaned: &[f64], mask: &[bool]) -> CleaningColumnReport {
    let n = raw.len();
    let n_nan_raw = raw.iter().filter(|v| !v.is_finite()).count();
    let n_outliers = mask.iter().filter(|&&b| b).count();
    let n_replaced = n_outliers;
    let n_nan_final = cleaned.iter().filter(|v| !v.is_finite()).count();

    let finite = |it: &[f64]| -> Vec<f64> { it.iter().copied().filter(|v| v.is_finite()).collect() };
    let raw_f = finite(raw);
    let clean_f = finite(cleaned);
    let (mean_raw, std_raw, min_raw, max_raw) = stats(&raw_f);
    let (mean_cleaned, std_cleaned, min_cleaned, max_cleaned) = stats(&clean_f);

    CleaningColumnReport {
        n_raw: n,
        n_nan_raw,
        n_outliers,
        n_replaced,
        n_nan_final,
        mean_raw,
        mean_cleaned,
        std_raw,
        std_cleaned,
        min_raw,
        max_raw,
        min_cleaned,
        max_cleaned,
    }
}

fn stats(xs: &[f64]) -> (f64, f64, f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std, min, max)
}

fn pct_modified(replaced: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * replaced as f64 / total as f64
    }
}

// ----------------------------------------------------------------------------
// DataFrame helpers
// ----------------------------------------------------------------------------

fn resolve_targets(df: &DataFrame, requested: &[String]) -> anyhow::Result<Vec<String>> {
    if requested.is_empty() {
        // Default: all numeric columns except TIMESTAMP.
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        return Ok(names
            .into_iter()
            .filter(|n| n != "TIMESTAMP")
            .filter(|n| {
                df.column(n)
                    .ok()
                    .map(|c| c.dtype().is_numeric())
                    .unwrap_or(false)
            })
            .collect());
    }
    for c in requested {
        df.column(c)
            .map_err(|e| anyhow::anyhow!("Colonne '{}' introuvable: {}", c, e))?;
    }
    Ok(requested.to_vec())
}

fn pull_f64(df: &DataFrame, col: &str) -> anyhow::Result<Vec<f64>> {
    let s = df
        .column(col)
        .map_err(|e| anyhow::anyhow!("Colonne '{}' introuvable: {}", col, e))?;
    let casted = s
        .cast(&DataType::Float64)
        .map_err(|e| anyhow::anyhow!("Colonne '{}' non numérique: {}", col, e))?;
    let ca: &ChunkedArray<Float64Type> = casted.f64().unwrap();
    Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn inject_column(df: &mut DataFrame, col: &str, values: &[f64]) -> anyhow::Result<()> {
    use polars::prelude::NamedFrom;
    let opts: Vec<Option<f64>> = values
        .iter()
        .map(|v| if v.is_finite() { Some(*v) } else { None })
        .collect();
    let new_series = Series::new(col.into(), &opts);
    df.replace(col, new_series)
        .map_err(|e| anyhow::anyhow!("Impossible de remplacer '{}': {}", col, e))?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Tests (legacy + new)
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_detect_outliers_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers_iqr(&data, 1.5);
        assert!(outliers[5]);
        assert!(!outliers[0]);
    }

    #[test]
    fn test_detect_outliers_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers_zscore(&data, 2.0);
        assert!(outliers[5]);
    }

    #[test]
    fn test_fill_gaps_linear() {
        let data = vec![1.0, 0.0, 0.0, 4.0];
        let mask = vec![false, true, true, false];
        let filled = fill_gaps_linear(&data, &mask);
        assert!((filled[1] - 2.0).abs() < 0.01);
        assert!((filled[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn mad_uses_0_6745_scaling() {
        // With 11 points symmetric around 0 and one outlier, MAD-scaled
        // z should flag the outlier and not the bulk.
        let mut data = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.2, -0.2];
        data.push(50.0);
        let mask = detect_outliers_mad(&data, 3.5);
        assert!(mask[11], "50.0 should be flagged");
        assert!(!mask[0], "center should not be flagged");
    }

    #[test]
    fn forward_fill_preserves_leading_nan() {
        let data = vec![f64::NAN, f64::NAN, 1.0, f64::NAN, 2.0];
        let mask = vec![true, true, false, true, false];
        let out = forward_fill(&data, &mask);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
        assert_eq!(out[4], 2.0);
    }

    #[test]
    fn rolling_mean_fill_uses_neighbours() {
        let data = vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0];
        let mask = vec![false, false, false, true, false, false, false];
        // window=3, half=1, index 3 looks at [2,5) → finite valid = {3.0, 5.0},
        // masked index 3 skipped → mean = 4.0
        let out = rolling_mean_fill(&data, &mask, 3);
        assert!((out[3] - 4.0).abs() < 1e-9, "got {}", out[3]);
    }

    #[test]
    fn welford_matches_naive_std() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = welford_std(&data);
        // Sample std
        let mean = data.iter().sum::<f64>() / 5.0;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4.0;
        let naive = var.sqrt();
        assert!((w - naive).abs() < 1e-9);
    }

    #[test]
    fn rolling_window_flags_spike_burst() {
        // 24 points at 30-min intervals (12h).
        let mut ts = Vec::new();
        let start = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let mut values: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
        for i in 0..24 {
            ts.push(start + chrono::Duration::minutes(30 * i as i64));
        }
        // Inject a spike burst in the middle (high variance).
        values[10] = 100.0;
        values[11] = -100.0;
        values[12] = 100.0;
        let mask = rolling_window_variance_mask(&values, &ts, 1.0, 5.0, 0.001);
        assert!(mask[10] || mask[11] || mask[12]);
    }

    #[test]
    fn reverse_detection_flags_low_daily_max() {
        // 3 days, day 2 has a max that's tiny vs neighbours.
        let day1 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let day2 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let day3 = NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
        let mut ts = Vec::new();
        let mut values = Vec::new();
        for d in [day1, day2, day3] {
            for h in 0..24 {
                ts.push(d.and_hms_opt(h, 0, 0).unwrap());
                // Day 2 gets very low values.
                values.push(if d == day2 { 0.1 } else { 10.0 });
            }
        }
        let bad = detect_reversed_days(&values, &ts, 6, 0.75);
        assert!(bad.contains(&day2));
        assert!(!bad.contains(&day1));
    }

    #[test]
    fn isolation_forest_flags_clear_outlier() {
        // 100 normal points + 3 far outliers in 2D.
        let mut c1: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let mut c2: Vec<f64> = (0..100).map(|i| (i as f64).cos()).collect();
        c1.extend_from_slice(&[50.0, -50.0, 100.0]);
        c2.extend_from_slice(&[50.0, -50.0, 100.0]);
        let usable = vec![true; 103];
        let scores = multivariate_isolation_scores(&[c1, c2], &usable, 50, 64, 42);
        // Last 3 rows should score higher than the first 100 on average.
        let avg_normal = scores[..100].iter().sum::<f64>() / 100.0;
        let avg_outliers = scores[100..].iter().sum::<f64>() / 3.0;
        assert!(
            avg_outliers > avg_normal,
            "IF outlier avg {} should exceed normal avg {}",
            avg_outliers,
            avg_normal
        );
    }

    #[test]
    fn c_factor_monotonic() {
        assert!(c_factor(2) < c_factor(100));
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
    }

    #[test]
    fn fill_gaps_classical_linear_bridges_existing_nans() {
        use polars::prelude::{DataFrame, NamedFrom, Series};
        // 6 points, NaN at indices 2 and 3 — linear interpolation between
        // 1 (idx 1) and 16 (idx 4) should give 6.0 and 11.0.
        let timestamps: Vec<String> = (0..6)
            .map(|i| format!("2024-01-01 0{}:00:00", i))
            .collect();
        let values: Vec<Option<f64>> = vec![
            Some(0.0),
            Some(1.0),
            None,
            None,
            Some(16.0),
            Some(20.0),
        ];
        let df = DataFrame::new(vec![
            Series::new("TIMESTAMP".into(), &timestamps),
            Series::new("SF1".into(), &values),
        ])
        .unwrap();

        let (out, report) = fill_gaps_classical(
            &df,
            &["SF1".to_string()],
            GapFillMethod::LinearInterpolate,
        )
        .unwrap();

        let col_report = report.per_column.get("SF1").expect("SF1 report missing");
        assert_eq!(col_report.n_gaps_before, 2);
        assert_eq!(col_report.n_gaps_after, 0);
        assert_eq!(col_report.n_filled, 2);
        assert_eq!(col_report.longest_gap_before, 2);
        assert_eq!(report.total_filled, 2);

        // Verify the interpolated values landed where expected.
        let filled = out.column("SF1").unwrap().f64().unwrap();
        let v2 = filled.get(2).unwrap();
        let v3 = filled.get(3).unwrap();
        assert!((v2 - 6.0).abs() < 1e-9, "got {}", v2);
        assert!((v3 - 11.0).abs() < 1e-9, "got {}", v3);
    }

    #[test]
    fn backward_fill_propagates_next_finite() {
        let data = vec![f64::NAN, f64::NAN, 5.0, f64::NAN, 7.0, f64::NAN];
        let mask = vec![false; 6];
        let out = backward_fill(&data, &mask);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], 5.0);
        assert_eq!(out[2], 5.0);
        assert_eq!(out[3], 7.0);
        assert_eq!(out[4], 7.0);
        assert!(out[5].is_nan(), "trailing NaN with no next finite stays NaN");
    }

    #[test]
    fn detect_only_returns_indices_without_mutating_input() {
        // Build a tiny DataFrame: TIMESTAMP + a numeric column with one
        // obvious outlier at index 5.
        use polars::prelude::{DataFrame, NamedFrom, Series};
        let ts = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let timestamps: Vec<String> = (0..6)
            .map(|i| (ts + chrono::Duration::hours(i)).format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();
        let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 100.0];
        let df = DataFrame::new(vec![
            Series::new("TIMESTAMP".into(), &timestamps),
            Series::new("SF1".into(), &values),
        ])
        .unwrap();

        let cfg = IqrConfig {
            k: 1.5,
            target_columns: vec!["SF1".to_string()],
            replace_strategy: ReplaceStrategy::LinearInterpolate,
        };
        let method = CleaningMethod::Iqr(cfg);

        let per_col = detect_outliers_only(&df, &method).unwrap();
        let sf1 = per_col.get("SF1").expect("SF1 missing from output");
        assert!(sf1.contains(&5), "expected index 5 (value 100) flagged: {:?}", sf1);
        // Source DataFrame is untouched (height/columns intact).
        assert_eq!(df.height(), 6);
        assert_eq!(df.get_column_names(), &["TIMESTAMP", "SF1"]);
    }
}
