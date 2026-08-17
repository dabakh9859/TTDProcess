use polars::prelude::*;
use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate, Timelike};
use std::collections::{BTreeMap, HashMap};

use crate::core::types::{T0SmoothConfig, TmMethod};

/// Cyclic timestamp pattern (seconds between measurements).
#[allow(dead_code)]
pub const TIMESTAMP_PATTERN: [i64; 11] = [30, 30, 60, 180, 300, 30, 30, 60, 180, 300, 600];

// =============================================================================
// Helper
// =============================================================================

fn ts_col_to_datetimes(col: &Series) -> Result<Vec<chrono::DateTime<chrono::Utc>>> {
    let dt_series = match col.dtype() {
        DataType::Datetime(_, _) => col.clone(),
        DataType::Int64 => col.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
        _ => anyhow::bail!("TIMESTAMP must be Datetime or Int64 (microseconds)"),
    };
    let ca = dt_series.datetime()?;
    Ok(ca
        .into_iter()
        .filter_map(|ts| {
            ts.map(|t| {
                let secs = t / 1_000_000;
                let nsecs = ((t % 1_000_000) * 1000) as u32;
                chrono::DateTime::from_timestamp(secs, nsecs).unwrap_or_default()
            })
        })
        .collect())
}

fn get_or_cast_ts(data: &DataFrame) -> Result<Series> {
    let col = data.column("TIMESTAMP")?;
    match col.dtype() {
        DataType::Datetime(_, _) => Ok(col.clone()),
        DataType::Int64 => Ok(col.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?),
        _ => anyhow::bail!("TIMESTAMP must be Datetime or Int64"),
    }
}

// =============================================================================
// Step 1 – Tslope
// =============================================================================

/// Calculate Tslope for TC_ and SF_ columns.
///
/// For each 30-minute window [t, t+30min):
///   - Find the value at t and at t+20min
///   - Tslope = (v_end - v_start) / 1200 seconds
///   - Apply to all points in the window
pub fn calculate_tslope(data: &DataFrame, _pattern: &[i64]) -> Result<DataFrame> {
    if !data.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column is required");
    }

    let sf_cols: Vec<String> = data
        .get_column_names()
        .iter()
        .filter(|col| {
            let upper = col.to_uppercase();
            (upper.starts_with("TC_") || upper.starts_with("SF_"))
                && data.column(col).ok().and_then(|c| c.f64().ok()).is_some()
        })
        .map(|s| s.to_string())
        .collect();

    if sf_cols.is_empty() {
        anyhow::bail!("No TC_ or SF_ columns found for Tslope calculation");
    }

    let ts_series = get_or_cast_ts(data)?;
    let n = data.height();

    let timestamps = ts_col_to_datetimes(data.column("TIMESTAMP")?)?;
    if timestamps.len() != n {
        anyhow::bail!("TIMESTAMP length ({}) != data height ({})", timestamps.len(), n);
    }

    // Monotonic diffs: compute delta from last STRICTLY-LATER timestamp.
    // Backwards timestamps (diff ≤ 0) and duplicates get diff = 0 so they
    // are invisible to the 300/600 detectors.
    let diffs: Vec<i64> = {
        let mut v = Vec::with_capacity(n - 1);
        let mut last_valid = timestamps[0];
        for i in 1..n {
            let d = (timestamps[i] - last_valid).num_seconds();
            if d > 0 {
                v.push(d);
                last_valid = timestamps[i];
            } else {
                v.push(0);
            }
        }
        v
    };

    // Cycle boundaries: diff >= 550s (normal ~600s rest or gap).
    let mut cycle_boundaries: Vec<usize> = vec![0];
    for i in 1..n {
        if diffs[i - 1] >= 550 {
            cycle_boundaries.push(i);
        }
    }

    use rayon::prelude::*;
    let mut tslope_series: Vec<Series> = sf_cols
        .par_iter()
        .map(|sf_col| -> Result<Series> {
            let values = data.column(sf_col)?.f64()?;
            let mut tslope_values: Vec<Option<f64>> = vec![None; n];
            let mut last_slope: Option<f64> = None;

            for ci in 0..cycle_boundaries.len() {
                let start = cycle_boundaries[ci];
                let end = if ci + 1 < cycle_boundaries.len() {
                    cycle_boundaries[ci + 1]
                } else {
                    n
                };

                if ci > 0 && diffs[start - 1] > 700 {
                    last_slope = None;
                }

                let mut count_300 = 0usize;
                let mut found_slope: Option<f64> = None;
                for i in (start + 1)..end {
                    let d = diffs[i - 1];
                    if d >= 250 && d <= 350 {
                        count_300 += 1;
                        if count_300 == 2 {
                            let elapsed = (timestamps[i] - timestamps[start])
                                .num_seconds() as f64;
                            if elapsed > 0.0 {
                                if let (Some(v_start), Some(v_end)) =
                                    (values.get(start), values.get(i))
                                {
                                    found_slope =
                                        Some((v_end - v_start) / elapsed);
                                }
                            }
                            break;
                        }
                    }
                }

                let slope = found_slope.or(last_slope);
                if let Some(s) = slope {
                    for j in start..end {
                        tslope_values[j] = Some(s);
                    }
                }
                if found_slope.is_some() {
                    last_slope = found_slope;
                }
            }

            let tslope_name = format!("Tslope_{}", sf_col);
            Ok(Series::new(tslope_name.as_str(), tslope_values))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut result_data: Vec<Series> = Vec::with_capacity(sf_cols.len() + 1);
    result_data.push(ts_series);
    result_data.append(&mut tslope_series);

    DataFrame::new(result_data).context("Tslope DataFrame error")
}

// =============================================================================
// Step 2 – Baseline
// =============================================================================

/// Calculate Baseline from Tslope using the cyclic pattern.
///
/// baseline[i] = baseline[i-1] + tslope[i] * timestep
/// At 600s steps and every 2nd 300s step, reset to raw SF value.
pub fn calculate_baseline(
    data: &DataFrame,
    _tslope_data: &DataFrame,
    _pattern: &[i64],
) -> Result<DataFrame> {
    if !data.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column is required");
    }

    // Only f64 SF_ columns — exactly the set Tslope produced. A dead sensor
    // whose column is entirely empty loads as a non-f64 dtype; Tslope skips it,
    // so Baseline must skip it too (otherwise `Tslope_<dead>` is missing and the
    // step errors). Dead sensors are simply omitted from the pipeline output.
    let sf_cols: Vec<String> = data
        .get_column_names()
        .iter()
        .filter(|col| {
            col.starts_with("SF_")
                && data.column(col).ok().and_then(|c| c.f64().ok()).is_some()
        })
        .map(|s| s.to_string())
        .collect();

    if sf_cols.is_empty() {
        anyhow::bail!("No numeric SF_ columns found");
    }

    let ts_series = get_or_cast_ts(data)?;
    let n = data.height();

    let timestamps = ts_col_to_datetimes(data.column("TIMESTAMP")?)?;
    if timestamps.len() != n {
        anyhow::bail!("TIMESTAMP length ({}) != data height ({})", timestamps.len(), n);
    }

    // Monotonic diffs: backwards/duplicate timestamps get diff = 0.
    let diffs: Vec<i64> = {
        let mut v = Vec::with_capacity(n - 1);
        let mut last_valid = timestamps[0];
        for i in 1..n {
            let d = (timestamps[i] - last_valid).num_seconds();
            if d > 0 {
                v.push(d);
                last_valid = timestamps[i];
            } else {
                v.push(0);
            }
        }
        v
    };

    let mut result_data: Vec<Series> = vec![ts_series];

    for sf_col in &sf_cols {
        let sf_values = data.column(sf_col)?.f64()?;

        let mut baseline_values: Vec<Option<f64>> = vec![None; n];

        baseline_values[0] = sf_values.get(0);

        // Pass 1: detect anchors from monotonic timestamp diffs.
        let mut count_300_in_cycle = 0usize;
        for i in 1..n {
            let diff = diffs[i - 1];

            let is_anchor = if diff >= 550 && diff <= 650 {
                count_300_in_cycle = 0;
                true
            } else if diff >= 250 && diff <= 350 {
                count_300_in_cycle += 1;
                if count_300_in_cycle == 2 {
                    count_300_in_cycle = 0;
                    true
                } else {
                    false
                }
            } else if diff > 650 {
                count_300_in_cycle = 0;
                false
            } else {
                false
            };

            if is_anchor {
                baseline_values[i] = sf_values.get(i);
            }
        }

        // Pass 1b: reject anchors at heated positions.
        // A valid anchor (natural temp) is near the local minimum.
        // If it sits in the upper 40 % of the local value range, it is
        // likely a heated reading placed by a false cycle boundary.
        for i in 1..n {
            if let Some(anchor_val) = baseline_values[i] {
                let w_start = i.saturating_sub(5);
                let w_end = (i + 6).min(n);
                let mut min_v = f64::MAX;
                let mut max_v = f64::MIN;
                for j in w_start..w_end {
                    if let Some(v) = sf_values.get(j) {
                        if v < min_v { min_v = v; }
                        if v > max_v { max_v = v; }
                    }
                }
                let range = max_v - min_v;
                if range > 1.0 && anchor_val > min_v + 0.4 * range {
                    baseline_values[i] = None;
                }
            }
        }

        // Pass 2: linearly interpolate between anchors (skip gaps > 1h).
        let mut prev_anchor = 0usize;
        for i in 1..n {
            if baseline_values[i].is_some() {
                let time_span =
                    (timestamps[i] - timestamps[prev_anchor]).num_seconds();
                if time_span <= 3600 {
                    if let (Some(start_val), Some(end_val)) =
                        (baseline_values[prev_anchor], baseline_values[i])
                    {
                        let span = (i - prev_anchor) as f64;
                        for j in (prev_anchor + 1)..i {
                            let frac = (j - prev_anchor) as f64 / span;
                            baseline_values[j] =
                                Some(start_val + frac * (end_val - start_val));
                        }
                    }
                }
                prev_anchor = i;
            }
        }
        if let Some(last_val) = baseline_values[prev_anchor] {
            let max_trail =
                (timestamps.last().unwrap().clone() - timestamps[prev_anchor])
                    .num_seconds();
            if max_trail <= 3600 {
                for i in (prev_anchor + 1)..n {
                    baseline_values[i] = Some(last_val);
                }
            }
        }

        let baseline_name = format!("Baseline_{}", &sf_col[3..]);
        result_data.push(Series::new(baseline_name.as_str(), baseline_values));
    }

    DataFrame::new(result_data).context("Baseline DataFrame error")
}

// =============================================================================
// Step 3 – Delta-T
// =============================================================================

/// Delta-T = SF - Baseline
pub fn calculate_delta_t(data: &DataFrame, baseline_data: &DataFrame) -> Result<DataFrame> {
    let ts_series = get_or_cast_ts(data)?;
    let mut result_data: Vec<Series> = vec![ts_series];

    // Same f64-only selection as Baseline/Tslope: skip dead (non-numeric) sensors
    // so we never look up a Baseline_ column that was never produced.
    let sf_cols: Vec<String> = data
        .get_column_names()
        .iter()
        .filter(|col| {
            col.starts_with("SF_")
                && data.column(col).ok().and_then(|c| c.f64().ok()).is_some()
        })
        .map(|s| s.to_string())
        .collect();

    for sf_col in &sf_cols {
        let sf_values = data.column(sf_col)?.f64()?;
        let baseline_col_name = format!("Baseline_{}", &sf_col[3..]);
        let baseline_values = baseline_data
            .column(&baseline_col_name)
            .with_context(|| format!("Missing column {}", baseline_col_name))?
            .f64()?;

        let delta_t_values: Vec<Option<f64>> = (0..data.height())
            .map(|i| match (sf_values.get(i), baseline_values.get(i)) {
                (Some(sf), Some(bl)) => Some(sf - bl),
                _ => None,
            })
            .collect();

        let name = format!("DeltaT_{}", &sf_col[3..]);
        result_data.push(Series::new(name.as_str(), delta_t_values));
    }

    DataFrame::new(result_data).context("DeltaT DataFrame error")
}

// =============================================================================
// Step 4 – T600 (filter to half-hour timestamps)
// =============================================================================

/// Extract one T600 value per 30-minute heating cycle.
///
/// Standard 11-row cycles have the heated peak at the :00/:30 timestamp.
/// Alternate 9-row cycles (seen in some field files, e.g. Niakhar Oct 2024)
/// shift the heated peak to :25/:55.  A hard-coded :00/:30 filter picks the
/// wrong row in the 9-row case, producing an oscillating ("cisaillement")
/// T600 series.
///
/// This function splits the data into ~30-minute windows and selects the row
/// with the maximum ΔT in each window — the heated peak by definition.  For
/// standard 11-row data this is equivalent to the :00/:30 filter (the peak
/// IS at :00/:30); for 9-row data it correctly picks the :25/:55 peak.
pub fn create_t600(delta_t: &DataFrame) -> Result<DataFrame> {
    let timestamp_col = delta_t.column("TIMESTAMP")?;
    let datetimes = ts_col_to_datetimes(timestamp_col)?;
    let n = delta_t.height();

    if datetimes.len() != n {
        anyhow::bail!(
            "TIMESTAMP length ({}) != data height ({})",
            datetimes.len(),
            n
        );
    }

    let dt_col_names: Vec<String> = delta_t
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("DeltaT_"))
        .map(|s| s.to_string())
        .collect();
    if dt_col_names.is_empty() {
        anyhow::bail!("No DeltaT_ columns found");
    }
    let dt_arrays: Vec<&Float64Chunked> = dt_col_names
        .iter()
        .map(|name| delta_t.column(name).unwrap().f64().unwrap())
        .collect();

    let mut mask = vec![false; n];

    if n > 0 {
        let mut window_start = 0usize;
        for i in 1..=n {
            let end_of_data = i == n;
            let new_window = if end_of_data {
                true
            } else {
                let elapsed =
                    (datetimes[i] - datetimes[window_start]).num_seconds();
                elapsed >= 1700 || elapsed < 0
            };

            if new_window {
                let window_end = i;
                let window_rows = window_end - window_start;

                if window_rows >= 5 {
                    let mut best_idx = window_start;
                    let mut best_dt = f64::NEG_INFINITY;
                    for j in window_start..window_end {
                        // Max DeltaT across ALL sensors at row j.
                        let mut row_max = f64::NEG_INFINITY;
                        for arr in &dt_arrays {
                            if let Some(v) = arr.get(j) {
                                if v > row_max {
                                    row_max = v;
                                }
                            }
                        }
                        if row_max > best_dt {
                            best_dt = row_max;
                            best_idx = j;
                        }
                    }
                    if best_dt > 0.0 {
                        mask[best_idx] = true;
                    }
                }

                if !end_of_data {
                    window_start = i;
                }
            }
        }
    }

    let mask_ca = BooleanChunked::from_slice("mask".into(), &mask);
    let filtered = delta_t.filter(&mask_ca).context("T600 filter error")?;

    let mut result_data: Vec<Series> = Vec::new();
    for col_name in filtered.get_column_names() {
        let col = filtered.column(col_name)?;
        if col_name.starts_with("DeltaT_") {
            let new_name = col_name.replace("DeltaT_", "T600_");
            result_data.push(col.clone().with_name(new_name.as_str().into()));

            let inv_name = col_name.replace("DeltaT_", "InvT600_");
            let inv_values: Vec<Option<f64>> = col
                .f64()?
                .into_iter()
                .map(|v| match v {
                    Some(x) if x.is_finite() && x > 0.0 => Some(1.0 / x),
                    _ => None,
                })
                .collect();
            result_data.push(Series::new(inv_name.as_str().into(), inv_values));
        } else {
            result_data.push(col.clone());
        }
    }

    DataFrame::new(result_data).context("T600 DataFrame error")
}

// =============================================================================
// Step 5 – Tm (nightly max temperature)
// =============================================================================

/// Dispatch to appropriate Tm method.
///
/// `env_df` is required iff `method == TmMethod::VpdPar` and must contain the
/// columns referenced by the config (VPD, radiation, timestamp).
///
/// Returns the Tm DataFrame plus optional per-night diagnostics (only emitted
/// by RegressionDiurne — the CalculsPage uses these to render its inspection
/// charts).
/// Return type of `calculate_tm`: the Tm DataFrame plus any method-specific
/// diagnostic the user can inspect in the UI.
pub struct TmComputeOutput {
    pub df: DataFrame,
    pub diurnal_diagnostics: Option<HashMap<(String, String), crate::core::types::DiurnalDiagnostic>>,
    pub vpd_par_stats: Option<crate::core::types::VpdParStats>,
    /// Step-by-step verification tables, only produced by RegressionDiurne.
    pub diurnal_steps: Option<DiurnalStepTables>,
}

/// The five per-step audit tables for the RegressionDiurne T0 determination.
/// Each is a long-format DataFrame (one row per diurnal point, or per
/// night×sensor for the summary steps) so a reviewer can verify every stage
/// of the calculation from the Tableau tab.
pub struct DiurnalStepTables {
    pub regression: DataFrame, // Step 4
    pub result: DataFrame,     // Step 5
}

pub fn calculate_tm(
    t600: &DataFrame,
    method: &TmMethod,
    env_df: Option<&DataFrame>,
) -> Result<TmComputeOutput> {
    let none_out = |df| TmComputeOutput { df, diurnal_diagnostics: None, vpd_par_stats: None, diurnal_steps: None };
    match method {
        TmMethod::PreAube { nb_max_points } => calculate_tm_default(t600, *nb_max_points).map(none_out),
        TmMethod::FenetreGlissante {
            window_days,
            night_start_hour,
            night_end_hour,
        } => calculate_tm_moving_window(
            t600,
            *window_days,
            *night_start_hour,
            *night_end_hour,
        ).map(none_out),
        TmMethod::DoubleRegression { min_points } => {
            calculate_tm_double_regression(t600, *min_points).map(none_out)
        }
        TmMethod::VpdPar { config } => {
            let env = env_df.ok_or_else(|| {
                anyhow::anyhow!(
                    "VpdPar: aucun fichier environnemental chargé. Charge VPD/PAR dans l'onglet Calculs."
                )
            })?;
            let (df, stats) = crate::core::vpd_par::calculate_tm_vpd_par(t600, env, config)?;
            Ok(TmComputeOutput { df, diurnal_diagnostics: None, vpd_par_stats: Some(stats), diurnal_steps: None })
        }
        TmMethod::RegressionDiurne {
            alpha,
            beta,
            etp_column,
            x_min,
            x_max,
            hour_min,
            hour_max,
            double_regression,
        } => {
            let env = env_df.ok_or_else(|| {
                anyhow::anyhow!(
                    "RegressionDiurne: aucun fichier environnemental chargé. Charge ETo dans l'onglet Calculs."
                )
            })?;
            let (df, diag) = calculate_tm_regression_diurne(t600, env, *alpha, *beta, etp_column, *x_min, *x_max, *hour_min, *hour_max, *double_regression)?;
            let steps = match &diag {
                Some(d) => Some(build_diurnal_step_tables(d)?),
                None => None,
            };
            Ok(TmComputeOutput { df, diurnal_diagnostics: diag, vpd_par_stats: None, diurnal_steps: steps })
        }
    }
}

/// Default method: mean of top-N nightly maximums (20h–8h).
fn calculate_tm_default(t600: &DataFrame, nb_max_points: usize) -> Result<DataFrame> {
    let datetimes = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?;

    let t600_cols: Vec<String> = t600
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T600_"))
        .map(|s| s.to_string())
        .collect();

    if t600_cols.is_empty() {
        anyhow::bail!("No T600_ columns found");
    }

    // Group indices by night (key = date at 20h start)
    let mut nights: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, dt) in datetimes.iter().enumerate() {
        let hour = dt.hour();
        let night_key = if hour >= 20 {
            dt.date_naive().to_string()
        } else {
            (dt.date_naive() - Duration::days(1)).to_string()
        };
        if hour >= 20 || hour < 8 {
            nights.entry(night_key).or_default().push(idx);
        }
    }

    let mut sorted_nights: Vec<String> = nights.keys().cloned().collect();
    sorted_nights.sort();

    let mut result_timestamps: Vec<i64> = Vec::new();
    let mut result_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    for c in &t600_cols {
        result_values.insert(c.clone(), Vec::new());
    }

    for night_key in &sorted_nights {
        let indices = &nights[night_key];

        for col in &t600_cols {
            let col_values = t600.column(col)?.f64()?;
            let mut vals: Vec<f64> = indices
                .iter()
                .filter_map(|&i| col_values.get(i))
                .filter(|v| v.is_finite())
                .collect();

            if !vals.is_empty() {
                vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let top: Vec<f64> = vals.iter().take(nb_max_points).copied().collect();
                let tm = top.iter().sum::<f64>() / top.len() as f64;
                result_values.get_mut(col).unwrap().push(Some(tm));
            } else {
                result_values.get_mut(col).unwrap().push(None);
            }
        }

        if let Ok(date) = chrono::NaiveDate::parse_from_str(night_key, "%Y-%m-%d") {
            if let Some(ts) = date.and_hms_opt(6, 0, 0) {
                let dt = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ts, chrono::Utc);
                result_timestamps.push(dt.timestamp() * 1_000_000);
            }
        }
    }

    let mut out: Vec<Series> = Vec::new();
    let ts_series = Series::new("TIMESTAMP", result_timestamps)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    out.push(ts_series);
    for col in &t600_cols {
        let name = format!("T0_{}", &col[5..]);
        out.push(Series::new(name.as_str(), result_values.remove(col).unwrap()));
    }

    DataFrame::new(out).context("Tm DataFrame error")
}

/// Granier rolling-baseline method (Granier 1985, Lu 2004, Pasqualotto 2019).
///
/// Two passes:
/// 1. **Per-day nocturnal max.** For every calendar day D, compute
///    `night_max(D) = max ΔT` over rows whose timestamp falls in the
///    nocturnal window `[night_start, night_end)`. Wrap-around windows
///    (e.g. 20h–8h) are keyed on the **morning end** date, matching the
///    convention used by `calculate_tm_default` (PreAube) and VPD/PAR.
/// 2. **Rolling window over days.** For each day D, set
///    `Tm(D) = max{ night_max(D − w/2) … night_max(D + w/2) }` where
///    `w = window_days`. Centred window → no phase lag.
///
/// Output schema matches the other Tm methods: ONE row per night, timestamp
/// at 20h of `night_key`. Step 7 (Tmi interpolation) propagates Tm(D)
/// onto every T600 row of the day, the same way it does for PreAube.
fn calculate_tm_moving_window(
    t600: &DataFrame,
    window_days: usize,
    night_start_hour: u32,
    night_end_hour: u32,
) -> Result<DataFrame> {
    let datetimes = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?;

    let t600_cols: Vec<String> = t600
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T600_"))
        .map(|s| s.to_string())
        .collect();
    if t600_cols.is_empty() {
        anyhow::bail!("No T600_ columns found");
    }

    // Map each t600 row to its night-key date. We key on the EVENING date
    // (matches PreAube + DoubleRegression): h ≥ night_start → today;
    // h < night_end on a wrap-around window → yesterday. Rows outside
    // the nocturnal window get None and don't contribute to night_max.
    let wraps_midnight = night_start_hour >= night_end_hour;
    let row_night_key = |dt: &chrono::DateTime<chrono::Utc>| -> Option<NaiveDate> {
        let h = dt.hour();
        let in_night = if wraps_midnight {
            h >= night_start_hour || h < night_end_hour
        } else {
            h >= night_start_hour && h < night_end_hour
        };
        if !in_night {
            return None;
        }
        let naive = dt.naive_utc().date();
        if wraps_midnight && h < night_end_hour {
            // Morning-of-the-next-day side of a wrap-around window →
            // belongs to the night that started yesterday evening.
            naive.pred_opt()
        } else {
            Some(naive)
        }
    };

    // Group t600 row indices by night-key.
    let mut rows_by_night: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for (i, dt) in datetimes.iter().enumerate() {
        if let Some(d) = row_night_key(dt) {
            rows_by_night.entry(d).or_default().push(i);
        }
    }
    if rows_by_night.is_empty() {
        anyhow::bail!(
            "Aucune donnée nocturne trouvée dans la fenêtre {}h–{}h",
            night_start_hour, night_end_hour
        );
    }
    let night_keys: Vec<NaiveDate> = rows_by_night.keys().copied().collect();

    // Per-sensor: night_max(D), then rolling Tm(D) = max over ±half days.
    let half = (window_days / 2) as isize;
    let mut result_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    for c in &t600_cols {
        result_values.insert(c.clone(), Vec::with_capacity(night_keys.len()));
    }

    for col in &t600_cols {
        let col_values = t600.column(col)?.f64()?;

        // Pass 1: night_max(D) for each present night.
        let mut night_max: BTreeMap<NaiveDate, f64> = BTreeMap::new();
        for (day, idxs) in &rows_by_night {
            let m = idxs.iter()
                .filter_map(|&i| col_values.get(i))
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);
            if m.is_finite() {
                night_max.insert(*day, m);
            }
        }

        // Pass 2: for each night-key D, rolling max over [D-half .. D+half].
        let out_vec = result_values.get_mut(col).unwrap();
        for (i, _day) in night_keys.iter().enumerate() {
            let lo = (i as isize - half).max(0) as usize;
            let hi = ((i as isize + half) as usize).min(night_keys.len().saturating_sub(1));
            let mut wmax = f64::NEG_INFINITY;
            for j in lo..=hi {
                if let Some(v) = night_max.get(&night_keys[j]) {
                    if *v > wmax { wmax = *v; }
                }
            }
            out_vec.push(if wmax.is_finite() { Some(wmax) } else { None });
        }
    }

    // Build TIMESTAMP series at 06h of each night-key. The hour is only a
    // display label (predawn, within the 20h→8h window); Tmi/sTm index Tm by
    // DATE, so this hour change leaves every downstream calculation identical.
    let result_timestamps: Vec<i64> = night_keys.iter()
        .filter_map(|d| d.and_hms_opt(6, 0, 0)
            .map(|nd| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(nd, chrono::Utc)
                 .timestamp() * 1_000_000))
        .collect();

    let mut out: Vec<Series> = Vec::new();
    let ts_series = Series::new("TIMESTAMP", result_timestamps)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    out.push(ts_series);
    for col in &t600_cols {
        let name = format!("T0_{}", &col[5..]);
        out.push(Series::new(name.as_str(), result_values.remove(col).unwrap()));
    }

    DataFrame::new(out).context("Tm (MovingWindow) DataFrame error")
}

/// Ordinary least squares fit of `y = a·x + b`. Returns `(slope, intercept)`,
/// or `None` if x is constant (degenerate, no slope defined) or there are
/// fewer than 2 finite pairs.
fn linear_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    let pairs: Vec<(f64, f64)> = xs.iter().zip(ys.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    let n = pairs.len();
    if n < 2 { return None; }
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n as f64;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in &pairs {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x).powi(2);
    }
    if den < f64::EPSILON { return None; }
    let a = num / den;
    let b = mean_y - a * mean_x;
    Some((a, b))
}

/// Double linear regression method (Lu 2004 / Pasqualotto 2019).
///
/// For each night:
///   1. Fit a linear regression of ΔT vs. time on ALL nocturnal points →
///      this captures the average trend (sensor drift, slow cooling).
///   2. Keep only the points whose observed ΔT lies ABOVE the line — these
///      are the candidates for the no-flow ΔTmax envelope.
///   3. Re-fit a second linear regression on the filtered upper points.
///   4. Tm(night) = mean of the upper points (if `min_points` are present),
///      or fallback to the simple max when too few points pass step 2.
///
/// Output schema matches the other Tm methods: ONE row per night,
/// timestamp at 20h of the night-key.
fn calculate_tm_double_regression(t600: &DataFrame, min_points: usize) -> Result<DataFrame> {
    let datetimes = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?;

    let t600_cols: Vec<String> = t600
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T600_"))
        .map(|s| s.to_string())
        .collect();
    if t600_cols.is_empty() {
        anyhow::bail!("No T600_ columns found");
    }

    // Group t600 row indices by night-key. We use the EVENING date as the
    // key (h>=20 → today; h<8 → yesterday), matching PreAube. The output
    // timestamp is then "20h of the night_key", so downstream sTm/Tmi
    // treat all Tm methods uniformly.
    let mut nights: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, dt) in datetimes.iter().enumerate() {
        let hour = dt.hour();
        if !(hour >= 20 || hour < 8) { continue; }
        let night_key = if hour >= 20 {
            dt.date_naive().to_string()
        } else {
            (dt.date_naive() - Duration::days(1)).to_string()
        };
        nights.entry(night_key).or_default().push(idx);
    }

    let mut sorted_nights: Vec<String> = nights.keys().cloned().collect();
    sorted_nights.sort();

    let mut result_timestamps: Vec<i64> = Vec::new();
    let mut result_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    for c in &t600_cols { result_values.insert(c.clone(), Vec::new()); }

    let min_pts_for_step2 = min_points.max(3);

    for night_key in &sorted_nights {
        // Emit timestamp = 06h of the night-key date (display label only; Tmi
        // keys on the DATE, so the calc is unchanged — see calculate_tm_default).
        let date = match chrono::NaiveDate::parse_from_str(night_key, "%Y-%m-%d") {
            Ok(d) => d, Err(_) => continue,
        };
        let Some(ts_naive) = date.and_hms_opt(6, 0, 0) else { continue; };
        let ts_us = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ts_naive, chrono::Utc)
            .timestamp() * 1_000_000;
        result_timestamps.push(ts_us);

        let indices = &nights[night_key];

        for col in &t600_cols {
            let col_values = t600.column(col)?.f64()?;
            // Build (t_seconds, ΔT) pairs anchored at the first point of
            // the night so the OLS x-axis starts near 0. Drop NaN values.
            let pairs: Vec<(f64, f64)> = indices.iter()
                .filter_map(|&i| {
                    let v = col_values.get(i)?;
                    if !v.is_finite() { return None; }
                    Some((datetimes[i].timestamp() as f64, v))
                })
                .collect();

            if pairs.is_empty() {
                result_values.get_mut(col).unwrap().push(None);
                continue;
            }
            let t0 = pairs[0].0;
            let xs: Vec<f64> = pairs.iter().map(|(t, _)| t - t0).collect();
            let ys: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();

            // --- Step 1: regression on ALL nocturnal points -----------
            let tm = match linear_fit(&xs, &ys) {
                Some((a1, b1)) => {
                    // --- Step 2: keep points above the trend line -----
                    let upper: Vec<(f64, f64)> = xs.iter().zip(ys.iter())
                        .filter(|(x, y)| **y > a1 * **x + b1)
                        .map(|(x, y)| (*x, *y))
                        .collect();

                    if upper.len() >= min_pts_for_step2 {
                        // --- Step 3: re-fit on the filtered upper set -
                        // We don't actually need the second slope/intercept
                        // for the output; the upper set already captures
                        // the "no-flow envelope" and the second regression
                        // mostly serves as a sanity check / future hook
                        // for outlier rejection on the upper set.
                        let xu: Vec<f64> = upper.iter().map(|(x, _)| *x).collect();
                        let yu: Vec<f64> = upper.iter().map(|(_, y)| *y).collect();
                        let _ = linear_fit(&xu, &yu); // discarded for now
                        // Tm = mean of the upper points (Pasqualotto 2019).
                        let mean = yu.iter().sum::<f64>() / yu.len() as f64;
                        Some(mean)
                    } else {
                        // Too few "upper" points to trust the second
                        // regression — fall back to the simple nocturnal
                        // max so the night still gets a Tm value rather
                        // than a NaN propagating downstream.
                        ys.iter().cloned().filter(|v| v.is_finite()).reduce(f64::max)
                    }
                }
                None => {
                    // Step 1 degenerate (constant x or <2 points) → simple max.
                    ys.iter().cloned().filter(|v| v.is_finite()).reduce(f64::max)
                }
            };

            result_values.get_mut(col).unwrap().push(tm);
        }
    }

    let mut out: Vec<Series> = Vec::new();
    let ts_series = Series::new("TIMESTAMP", result_timestamps)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    out.push(ts_series);
    for col in &t600_cols {
        let name = format!("T0_{}", &col[5..]);
        out.push(Series::new(name.as_str(), result_values.remove(col).unwrap()));
    }

    DataFrame::new(out).context("Tm (DoubleRegression) DataFrame error")
}

/// Regaldo & Ritter 2007 method: Diurnal regression based on ETo.
///
/// This method implements the alternative approach to estimate ΔTmax (T600_max)
/// using diurnal measurements of T600 and potential evapotranspiration (ETo),
/// without assuming zero sap flow at night.
///
/// Algorithm (from Regaldo & Ritter 2007, Tree Physiology):
/// 1. Select diurnal data only (daytime hours, typically 6h-20h)
/// 2. Transform variables: Y = 1/T600, X = ETo^(1/β)
/// 3. Perform robust linear regression: 1/T600 = (1/m) * ETo^(1/β) + 1/T600_max
/// 4. T600_max = 1 / intercept
///
/// The robust regression uses RANSAC (RANdom SAmple Consensus) to down-weight
/// outliers, as suggested in the original paper.
fn calculate_tm_regression_diurne(
    t600: &DataFrame,
    env_df: &DataFrame,
    alpha: f64,
    beta: f64,
    etp_column: &str,
    x_min: Option<f64>,
    x_max: Option<f64>,
    hour_min: Option<u32>,
    hour_max: Option<u32>,
    double_regression: bool,
) -> Result<(DataFrame, Option<HashMap<(String, String), crate::core::types::DiurnalDiagnostic>>)> {
    use crate::core::types::{DiurnalDiagnostic, DiurnalPoint};
    let _ = alpha; // kept for API stability; the regression itself uses 1/intercept

    let t600_datetimes = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?;
    let env_datetimes = ts_col_to_datetimes(env_df.column("TIMESTAMP")?)?;

    // Diagnostics accumulator. Keyed by (night_date, t600_col).
    let mut diagnostics: HashMap<(String, String), DiurnalDiagnostic> = HashMap::new();

    // Get T600 columns
    let t600_cols: Vec<String> = t600
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T600_"))
        .map(|s| s.to_string())
        .collect();

    if t600_cols.is_empty() {
        anyhow::bail!("No T600_ columns found");
    }

    // Get ETo column from env data
    let etp_series = env_df.column(etp_column)
        .with_context(|| format!("ETo column '{}' not found in env data", etp_column))?;
    let etp_values: Vec<Option<f64>> = etp_series.f64()?.into_iter().collect();

    // Multi-resolution lookup tables so the join survives mismatched sampling
    // rates between T600 (5min) and the env file (often hourly or daily). On
    // each diurnal point we try the most specific level first and fall back.
    let mut etp_by_second: HashMap<i64, f64> = HashMap::new();
    let mut etp_by_hour:   HashMap<i64, (f64, u32)> = HashMap::new(); // (sum, count)
    let mut etp_by_date:   HashMap<chrono::NaiveDate, (f64, u32)> = HashMap::new();
    for (i, dt) in env_datetimes.iter().enumerate() {
        if let Some(etp) = etp_values.get(i).and_then(|v| *v) {
            if !etp.is_finite() || etp < 0.0 { continue; }
            let sec = dt.timestamp();
            etp_by_second.insert(sec, etp);
            let hour_bucket = sec - sec.rem_euclid(3600);
            let h = etp_by_hour.entry(hour_bucket).or_insert((0.0, 0));
            h.0 += etp; h.1 += 1;
            let d = etp_by_date.entry(dt.date_naive()).or_insert((0.0, 0));
            d.0 += etp; d.1 += 1;
        }
    }
    // Closure that tries second → hour → day. Returns Some(etp) if any level hits.
    let lookup_etp = |dt: &chrono::DateTime<chrono::Utc>| -> Option<f64> {
        let s = dt.timestamp();
        if let Some(v) = etp_by_second.get(&s).copied() { return Some(v); }
        let hour_bucket = s - s.rem_euclid(3600);
        if let Some(&(sum, n)) = etp_by_hour.get(&hour_bucket) {
            if n > 0 { return Some(sum / n as f64); }
        }
        if let Some(&(sum, n)) = etp_by_date.get(&dt.date_naive()) {
            if n > 0 { return Some(sum / n as f64); }
        }
        None
    };

    // Process each T600 column
    let mut result_timestamps: Vec<i64> = Vec::new();
    let mut result_values: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    for c in &t600_cols {
        result_values.insert(c.clone(), Vec::new());
    }

    // Group by night (20h-8h) for output format consistency with other methods
    let mut nights: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, dt) in t600_datetimes.iter().enumerate() {
        let hour = dt.hour();
        let night_key = if hour >= 20 {
            dt.date_naive().to_string()
        } else {
            (dt.date_naive() - Duration::days(1)).to_string()
        };
        nights.entry(night_key).or_default().push(idx);
    }

    for (night_key, indices) in &nights {
        // Emit timestamp = 06h of the night-key date (display label only; Tmi
        // keys on the DATE, so the calc is unchanged).
        let date = match chrono::NaiveDate::parse_from_str(night_key, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };
        let Some(ts_naive) = date.and_hms_opt(6, 0, 0) else { continue };
        let ts_us = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ts_naive, chrono::Utc)
            .timestamp() * 1_000_000;
        result_timestamps.push(ts_us);

        for col in &t600_cols {
            let col_values = t600.column(col)?.f64()?;

            // Collect raw diurnal points (6h-20h) AND the transformed pairs
            // in the same iteration so the diagnostic indices line up with
            // the regression input.
            let mut diur_points: Vec<DiurnalPoint> = Vec::new();
            let mut xy: Vec<(f64, f64)> = Vec::new(); // (ETo^(1/β), 1/T600), parallel to diur_points
            let mut n_in_window: usize = 0;
            let mut n_t600_valid: usize = 0;
            let mut n_etp_matched: usize = 0;

            // Diurnal window for COLLECTION stays hardcoded 6h–20h (Regaldo
            // & Ritter standard). The user's hour_min / hour_max bounds are
            // applied later, at sel_mask time — so every diurnal point still
            // appears on the regression chart (in/out of range = blue/gray),
            // matching the user's "display all, select only inside range"
            // mental model.
            for &idx in indices {
                let hour_u = t600_datetimes[idx].hour();
                if hour_u < 6 || hour_u >= 20 {
                    continue;
                }
                n_in_window += 1;
                let t600_val = match col_values.get(idx) {
                    Some(v) if v.is_finite() && v > 0.0 => v,
                    _ => continue,
                };
                n_t600_valid += 1;
                let etp = match lookup_etp(&t600_datetimes[idx]) {
                    Some(v) if v > 0.0 => v,
                    _ => continue,
                };
                n_etp_matched += 1;

                let x = etp.powf(1.0 / beta);
                let y = 1.0 / t600_val;
                xy.push((x, y));

                let hour_frac = hour_u as f64 + t600_datetimes[idx].minute() as f64 / 60.0;
                diur_points.push(DiurnalPoint {
                    timestamp: t600_datetimes[idx].to_rfc3339(),
                    hour: hour_frac,
                    etp,
                    t600: t600_val,
                });
            }

            // Nocturnal max (20h–8h) — independent diagnostic, useful regardless
            // of whether the regression has enough points.
            let t_night_max = {
                let m: f64 = indices
                    .iter()
                    .filter(|&&i| {
                        let h = t600_datetimes[i].hour();
                        h >= 20 || h < 8
                    })
                    .filter_map(|&i| col_values.get(i))
                    .filter(|v| v.is_finite())
                    .fold(0.0_f64, f64::max);
                if m > 0.0 { Some(m) } else { None }
            };

            // Guard: the multi-level ETo lookup may return the same daily
            // value for every diurnal point if the env file is daily. That
            // collapses X variance and Theil-Sen produces a meaningless Tm,
            // which then cascades into a broken K and Fd. Treat that case
            // as "no regression possible": T0 is left undefined for the night
            // (no fallback) so the user sees explicitly which nights were
            // skipped instead of a silent substitution.
            let distinct_x: usize = {
                let mut xs: Vec<f64> = xy.iter().map(|p| (p.0 * 1e6).round() / 1e6).collect();
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                xs.dedup();
                xs.len()
            };

            if xy.len() < 5 || distinct_x < 3 {
                // Not enough diurnal points → skip this night (T0 = None).
                result_values.get_mut(col).unwrap().push(None);
                // Still emit a diagnostic so the user sees why no regression ran.
                diagnostics.insert(
                    (night_key.clone(), col.clone()),
                    DiurnalDiagnostic {
                        date: night_key.clone(),
                        column: col.clone(),
                        points: diur_points,
                        selected: Vec::new(),
                        beta,
                        slope: 0.0,
                        intercept: 0.0,
                        r2: 0.0,
                        t_max: None,
                        t_night_max,
                        n_in_window,
                        n_t600_valid,
                        n_etp_matched,
                    },
                );
                continue;
            }

            // Selection mask. Optional manual filters (X-range AND/OR
            // hour-range) — when ANY bound is set, only points inside the
            // range(s) feed the regression. With every bound None, ALL
            // diurnal points (already restricted to the 6h–20h window) feed
            // the regression. Hours come from diur_points which is built in
            // parallel with xy, so indices line up.
            let manual_active = x_min.is_some() || x_max.is_some()
                || hour_min.is_some() || hour_max.is_some();
            let mut sel_mask: Vec<bool> = if manual_active {
                xy.iter().enumerate().map(|(i, (x, _))| {
                    let lo_x = x_min.map(|v| *x >= v).unwrap_or(true);
                    let hi_x = x_max.map(|v| *x <= v).unwrap_or(true);
                    let h = diur_points[i].hour;
                    let lo_h = hour_min.map(|v| h >= v as f64).unwrap_or(true);
                    let hi_h = hour_max.map(|v| h < v as f64).unwrap_or(true);
                    lo_x && hi_x && lo_h && hi_h
                }).collect()
            } else {
                vec![true; xy.len()]
            };
            let n_selected = sel_mask.iter().filter(|&&b| b).count();
            if n_selected < 5 {
                // Not enough selected points (manual range too narrow) →
                // skip this night (T0 = None). The policy is to make
                // insufficient nights explicit, not paper over them.
                result_values.get_mut(col).unwrap().push(None);
                diagnostics.insert(
                    (night_key.clone(), col.clone()),
                    DiurnalDiagnostic {
                        date: night_key.clone(),
                        column: col.clone(),
                        points: diur_points,
                        selected: Vec::new(),
                        beta,
                        slope: 0.0,
                        intercept: 0.0,
                        r2: 0.0,
                        t_max: None,
                        t_night_max,
                        n_in_window,
                        n_t600_valid,
                        n_etp_matched,
                    },
                );
                continue;
            }

            let regression_points: Vec<(f64, f64)> = xy
                .iter()
                .zip(sel_mask.iter())
                .filter_map(|((x, y), keep)| if *keep { Some((*x, *y)) } else { None })
                .collect();

            // Post-filter distinct-X guard. The raw-xy guard at the top of
            // this loop is computed BEFORE the manual range is applied; a
            // narrow user range can collapse to <3 distinct X values, and
            // Theil-Sen on a near-vertical column produces a meaningless
            // slope/intercept. Skip this night (T0 = None) with a clear
            // raison so the user sees what went wrong.
            let distinct_x_post: usize = {
                let mut xr: Vec<f64> = regression_points.iter().map(|(x, _)| (x * 1e6).round() / 1e6).collect();
                xr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                xr.dedup();
                xr.len()
            };
            if distinct_x_post < 3 {
                result_values.get_mut(col).unwrap().push(None);
                diagnostics.insert(
                    (night_key.clone(), col.clone()),
                    DiurnalDiagnostic {
                        date: night_key.clone(),
                        column: col.clone(),
                        points: diur_points,
                        selected: Vec::new(),
                        beta,
                        slope: 0.0,
                        intercept: 0.0,
                        r2: 0.0,
                        t_max: None,
                        t_night_max,
                        n_in_window,
                        n_t600_valid,
                        n_etp_matched,
                    },
                );
                continue;
            }

            // ---- First-pass Theil-Sen fit on the manually/tolerance-selected points.
            let xs: Vec<f64> = regression_points.iter().map(|(x, _)| *x).collect();
            let ys: Vec<f64> = regression_points.iter().map(|(_, y)| *y).collect();

            let (mut slope, mut intercept, mut t_max) = match robust_linear_fit(&xs, &ys) {
                Some((s, i)) => {
                    let t_max = if i > 0.0 { Some(1.0 / i) } else { None };
                    (s, i, t_max)
                }
                None => (0.0, 0.0, None),
            };

            // ---- Optional second-pass (Pasqualotto-style double regression).
            // Same idea as the nocturnal calculate_tm_double_regression but
            // mirrored: in diurne, Y = 1/T600, so points BELOW the first-pass
            // line carry the no-flow signal (lower 1/T600 = larger T600 =
            // close to ΔTmax). Keep them, re-fit, and use the new intercept.
            // The retained mask is also AND-ed onto sel_mask so the chart
            // and step-3 audit table reflect what really fed the final fit.
            // Falls back to the first-pass fit if the lower envelope is too
            // sparse to support a robust second fit (< 3 points).
            if double_regression && !xs.is_empty() {
                let lower_mask: Vec<bool> = xs.iter().zip(ys.iter())
                    .map(|(x, y)| *y <= slope * *x + intercept)
                    .collect();
                let n_lower = lower_mask.iter().filter(|&&b| b).count();
                if n_lower >= 3 {
                    let xs2: Vec<f64> = xs.iter().zip(lower_mask.iter())
                        .filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect();
                    let ys2: Vec<f64> = ys.iter().zip(lower_mask.iter())
                        .filter_map(|(y, k)| if *k { Some(*y) } else { None }).collect();
                    if let Some((s2, i2)) = robust_linear_fit(&xs2, &ys2) {
                        slope = s2;
                        intercept = i2;
                        t_max = if i2 > 0.0 { Some(1.0 / i2) } else { None };
                        // Project lower_mask back onto sel_mask: a point is
                        // RETAINED only if it was kept in pass 1 AND below
                        // the pass-1 line.
                        let mut k = 0usize;
                        for slot in sel_mask.iter_mut() {
                            if *slot {
                                *slot = lower_mask[k];
                                k += 1;
                            }
                        }
                    }
                }
                // If n_lower < 3: keep the first-pass fit and sel_mask as is.
            }

            // r² of the FINAL regression on the points that actually fed the
            // last fit (sel_mask, after the optional second-pass projection).
            let final_pts: Vec<(f64, f64)> = xy.iter().zip(sel_mask.iter())
                .filter_map(|((x, y), k)| if *k { Some((*x, *y)) } else { None }).collect();
            let r2 = if final_pts.len() >= 2 {
                let ys_f: Vec<f64> = final_pts.iter().map(|(_, y)| *y).collect();
                let mean_y = ys_f.iter().sum::<f64>() / ys_f.len() as f64;
                let ss_res: f64 = final_pts.iter()
                    .map(|(x, y)| {
                        let pred = slope * x + intercept;
                        (y - pred).powi(2)
                    })
                    .sum();
                let ss_tot: f64 = ys_f.iter().map(|y| (y - mean_y).powi(2)).sum();
                if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 }
            } else {
                0.0
            };

            result_values.get_mut(col).unwrap().push(t_max);

            diagnostics.insert(
                (night_key.clone(), col.clone()),
                DiurnalDiagnostic {
                    date: night_key.clone(),
                    column: col.clone(),
                    points: diur_points,
                    selected: sel_mask,
                    beta,
                    slope,
                    intercept,
                    r2,
                    t_max,
                    t_night_max,
                    n_in_window,
                    n_t600_valid,
                    n_etp_matched,
                },
            );
        }
    }

    let mut out: Vec<Series> = Vec::new();
    let ts_series = Series::new("TIMESTAMP", result_timestamps)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    out.push(ts_series);
    for col in &t600_cols {
        let name = format!("T0_{}", &col[5..]);
        out.push(Series::new(name.as_str(), result_values.remove(col).unwrap()));
    }

    let df = DataFrame::new(out).context("Tm (RegressionDiurne) DataFrame error")?;
    Ok((df, Some(diagnostics)))
}

/// Robust linear fit using Theil-Sen estimator (median of slopes)
/// followed by intercept calculation via median.
/// This is less sensitive to outliers than ordinary least squares.
fn robust_linear_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    if xs.len() < 2 || xs.len() != ys.len() {
        return None;
    }

    // Theil-Sen estimator: median of all pairwise slopes
    let mut slopes: Vec<f64> = Vec::new();
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            let dx = xs[j] - xs[i];
            let dy = ys[j] - ys[i];
            if dx.abs() > 1e-10 {
                slopes.push(dy / dx);
            }
        }
    }

    if slopes.is_empty() {
        return None;
    }

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let slope = slopes[slopes.len() / 2]; // median

    // Intercept: median of (y - slope * x)
    let mut intercepts: Vec<f64> = xs.iter().zip(ys.iter())
        .map(|(x, y)| y - slope * x)
        .collect();
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let intercept = intercepts[intercepts.len() / 2]; // median

    Some((slope, intercept))
}

// =============================================================================
// RegressionDiurne — step-by-step verification tables
// =============================================================================

/// Turn the per-(night, sensor) RegressionDiurne diagnostics into two
/// long-format audit tables surfaced in the Tableau tab so a reviewer can
/// re-derive T0 by hand:
///   4. regression — Theil-Sen slope / intercept / r² per night×sensor
///   5. result     — T0 = 1/intercept (None when the regression was skipped),
///                   nocturnal max kept as a side-by-side informational value
///
/// All numbers are recomputed from the stored diagnostic so the table matches
/// exactly what the pipeline used (the `selected` mask is authoritative).
fn build_diurnal_step_tables(
    diags: &HashMap<(String, String), crate::core::types::DiurnalDiagnostic>,
) -> Result<DiurnalStepTables> {
    // Stable order: by (night date, sensor column).
    let mut keys: Vec<&(String, String)> = diags.keys().collect();
    keys.sort();

    // Step 4 — regression (one row per night×sensor)
    let (mut s4_date, mut s4_cap) = (Vec::new(), Vec::new());
    let (mut s4_n, mut s4_nsel): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
    let (mut s4_beta, mut s4_slope, mut s4_intercept, mut s4_r2): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    // Step 5 — result (one row per night×sensor)
    let (mut s5_date, mut s5_cap) = (Vec::new(), Vec::new());
    let (mut s5_intercept, mut s5_t0, mut s5_nightmax): (Vec<f64>, Vec<Option<f64>>, Vec<Option<f64>>) =
        (Vec::new(), Vec::new(), Vec::new());

    for key in keys {
        let d = &diags[key];
        let n = d.points.len();
        let n_sel = d.selected.iter().filter(|&&b| b).count();

        // Step 4
        s4_date.push(d.date.clone()); s4_cap.push(d.column.clone());
        s4_n.push(n as i64); s4_nsel.push(n_sel as i64);
        s4_beta.push(d.beta); s4_slope.push(d.slope); s4_intercept.push(d.intercept); s4_r2.push(d.r2);

        // Step 5
        s5_date.push(d.date.clone()); s5_cap.push(d.column.clone());
        s5_intercept.push(d.intercept);
        // T0 = 1/intercept, None when the night was skipped (no fallback).
        s5_t0.push(d.t_max);
        s5_nightmax.push(d.t_night_max);
    }

    let regression = DataFrame::new(vec![
        Series::new("date_nuit", s4_date),
        Series::new("capteur", s4_cap),
        Series::new("n_points", s4_n),
        Series::new("n_retenus", s4_nsel),
        Series::new("beta", s4_beta),
        Series::new("pente", s4_slope),
        Series::new("ordonnee", s4_intercept),
        Series::new("r2", s4_r2),
    ])?;

    let result = DataFrame::new(vec![
        Series::new("date_nuit", s5_date),
        Series::new("capteur", s5_cap),
        Series::new("ordonnee", s5_intercept),
        Series::new("T0 (=1/ordonnee)", s5_t0),
        Series::new("max_nocturne", s5_nightmax),
    ])?;

    Ok(DiurnalStepTables { regression, result })
}

// =============================================================================
// Step 6 – sTm (slope of Tm between consecutive nights)
// =============================================================================

/// sTm = (T0_today – T0_yesterday) / 24
/// Applied only between 8h30 and 19h30 on the T600 grid.
/// Centred moving average over non-null values, window in samples.
fn smooth_moving_avg(vals: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let n = vals.len();
    let half = window / 2;
    (0..n).map(|i| {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let (mut s, mut c) = (0.0, 0usize);
        for v in &vals[lo..hi] { if let Some(x) = v { s += *x; c += 1; } }
        if c > 0 { Some(s / c as f64) } else { None }
    }).collect()
}

/// Centred rolling median over non-null values (robust to spikes).
fn smooth_median(vals: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let n = vals.len();
    let half = window / 2;
    (0..n).map(|i| {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let mut w: Vec<f64> = vals[lo..hi].iter().filter_map(|v| *v).collect();
        if w.is_empty() { return None; }
        w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = w.len();
        Some(if m % 2 == 1 { w[m / 2] } else { (w[m / 2 - 1] + w[m / 2]) / 2.0 })
    }).collect()
}

/// Exponential smoother (forward EWMA) over non-null values; α in (0,1].
fn smooth_ewma(vals: &[Option<f64>], alpha: f64) -> Vec<Option<f64>> {
    let a = alpha.clamp(0.01, 1.0);
    let mut prev: Option<f64> = None;
    vals.iter().map(|v| match v {
        Some(x) => {
            let y = match prev { Some(p) => a * x + (1.0 - a) * p, None => *x };
            prev = Some(y);
            Some(y)
        }
        None => None,
    }).collect()
}

/// Whittaker–Henderson smoother (discrete smoothing spline, 2nd-difference
/// penalty): minimise Σ(y−g)² + λ Σ(Δ²g)². Operates on the non-null subset
/// (treated as equally spaced) and writes the result back in place.
fn smooth_spline(vals: &[Option<f64>], lambda: f64) -> Vec<Option<f64>> {
    let idx: Vec<usize> = vals.iter().enumerate().filter_map(|(i, v)| v.map(|_| i)).collect();
    let y: Vec<f64> = idx.iter().map(|&i| vals[i].unwrap()).collect();
    let m = y.len();
    let mut out = vals.to_vec();
    if m < 4 || lambda <= 0.0 { return out; }
    // A = I + λ·DᵀD, with D the (m-2)×m second-difference operator.
    let mut a = vec![vec![0.0f64; m]; m];
    for i in 0..m { a[i][i] = 1.0; }
    for r in 0..(m - 2) {
        // row r of D has [1,-2,1] at columns r,r+1,r+2 → add λ·(dᵀd) contributions.
        let cols = [r, r + 1, r + 2];
        let d = [1.0, -2.0, 1.0];
        for p in 0..3 { for q in 0..3 { a[cols[p]][cols[q]] += lambda * d[p] * d[q]; } }
    }
    // Solve A·g = y by Gaussian elimination with partial pivoting (m ~ 110).
    let mut g = y.clone();
    for k in 0..m {
        let mut piv = k;
        for r in (k + 1)..m { if a[r][k].abs() > a[piv][k].abs() { piv = r; } }
        if a[piv][k].abs() < 1e-12 { return out; }
        a.swap(k, piv); g.swap(k, piv);
        for r in (k + 1)..m {
            let f = a[r][k] / a[k][k];
            if f == 0.0 { continue; }
            for c in k..m { a[r][c] -= f * a[k][c]; }
            g[r] -= f * g[k];
        }
    }
    for k in (0..m).rev() {
        let mut s = g[k];
        for c in (k + 1)..m { s -= a[k][c] * g[c]; }
        g[k] = s / a[k][k];
    }
    for (j, &i) in idx.iter().enumerate() { out[i] = Some(g[j]); }
    out
}

/// Smooth the final T0 columns according to `cfg` (all T0 methods).
/// TIMESTAMP and `T0_source_*` are left untouched.
pub fn smooth_tm(df: &DataFrame, cfg: &T0SmoothConfig) -> DataFrame {
    let method = cfg.method.as_str();
    if method.is_empty() || method == "none" {
        return df.clone();
    }
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T0_") && !c.starts_with("T0_source_"))
        .map(|s| s.to_string())
        .collect();
    let n = df.height();
    let mut out = df.clone();
    for name in &names {
        let ca = match df.column(name).and_then(|c| c.f64().map(|x| x.clone())) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let vals: Vec<Option<f64>> = (0..n).map(|i| ca.get(i)).collect();
        let smoothed = match method {
            "moving_avg" => smooth_moving_avg(&vals, cfg.window.max(1)),
            "median" => smooth_median(&vals, cfg.window.max(1)),
            "ewma" => smooth_ewma(&vals, cfg.alpha),
            "spline" => smooth_spline(&vals, cfg.lambda),
            _ => vals,
        };
        let _ = out.with_column(Series::new(name.as_str(), smoothed));
    }
    out
}

pub fn calculate_stm(tm: &DataFrame, t600: &DataFrame) -> Result<DataFrame> {
    if tm.height() < 2 {
        anyhow::bail!("Not enough Tm data to compute sTm (need >= 2 nights)");
    }

    let tm_datetimes = ts_col_to_datetimes(tm.column("TIMESTAMP")?)?;
    let t600_datetimes = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?;

    // VPD/PAR's Tm DataFrame contains TWO columns per sensor:
    //   T0_<sensor>         → f64 (the night's ΔTmax)
    //   T0_source_<sensor>  → string ("Valid"|"Interpolated"|"NoValidNight")
    // Both prefix-match "T0_", so we have to explicitly exclude the source
    // tag columns — otherwise the f64 cast below blows up on a string col.
    let tm_cols: Vec<String> = tm
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T0_") && !c.starts_with("T0_source_"))
        .map(|s| s.to_string())
        .collect();

    if tm_cols.is_empty() {
        anyhow::bail!("No T0_ columns found");
    }

    // Generate half-hour grid from T600 range
    let start = *t600_datetimes.first().context("T600 empty")?;
    let end = *t600_datetimes.last().context("T600 empty")?;
    let mut all_ts: Vec<i64> = Vec::new();
    let mut cur = start;
    while cur <= end {
        all_ts.push(cur.timestamp() * 1_000_000);
        cur += Duration::minutes(30);
    }

    // Index Tm values by date
    let mut tm_by_date: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (idx, dt) in tm_datetimes.iter().enumerate() {
        let date_key = dt.date_naive().to_string();
        let entry = tm_by_date.entry(date_key).or_default();
        for col in &tm_cols {
            if let Some(val) = tm.column(col)?.f64()?.get(idx) {
                entry.entry(col.clone()).or_insert(val);
            }
        }
    }

    let mut sorted_dates: Vec<String> = tm_by_date.keys().cloned().collect();
    sorted_dates.sort();

    // Map grid timestamps to datetimes
    let all_datetimes: Vec<chrono::DateTime<chrono::Utc>> = all_ts
        .iter()
        .map(|&ts| {
            let secs = ts / 1_000_000;
            let nsecs = ((ts % 1_000_000) * 1000) as u32;
            chrono::DateTime::from_timestamp(secs, nsecs).unwrap_or_default()
        })
        .collect();

    let mut out: Vec<Series> = Vec::new();
    let ts_series = Series::new("TIMESTAMP", all_ts.clone())
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    out.push(ts_series);

    for col in &tm_cols {
        let mut stm_values: Vec<Option<f64>> = vec![None; all_ts.len()];

        for i in 1..sorted_dates.len() {
            let cur_date = &sorted_dates[i];
            let prev_date = &sorted_dates[i - 1];

            if let (Some(cur_tm_map), Some(prev_tm_map)) =
                (tm_by_date.get(cur_date), tm_by_date.get(prev_date))
            {
                if let (Some(&cur_tm), Some(&prev_tm)) =
                    (cur_tm_map.get(col), prev_tm_map.get(col))
                {
                    let stm = (cur_tm - prev_tm) / 24.0;
                    if let Ok(prev_date_naive) =
                        chrono::NaiveDate::parse_from_str(prev_date, "%Y-%m-%d")
                    {
                        for (idx, dt) in all_datetimes.iter().enumerate() {
                            if dt.date_naive() == prev_date_naive {
                                let hour = dt.hour();
                                let minute = dt.minute();
                                let in_range = (hour == 8 && minute >= 30)
                                    || (hour > 8 && hour < 19)
                                    || (hour == 19 && minute <= 30);
                                if in_range {
                                    stm_values[idx] = Some(stm);
                                }
                            }
                        }
                    }
                }
            }
        }

        let name = format!("sT0_{}", &col[3..]);
        out.push(Series::new(name.as_str(), stm_values));
    }

    DataFrame::new(out).context("sTm DataFrame error")
}

// =============================================================================
// Step 7 – Tmi (Tm interpolated over the day)
// =============================================================================

/// Interpolation strategy per T600 timestamp:
/// - 00h – 8h30  : Tm of the previous night (constant)
/// - 8h30 – 19h30 : T0_yesterday + n * sTm  (n = half-hour steps since 8h30)
/// - 20h – 23h30  : Tm of the current night (constant)
pub fn calculate_tmi(tm: &DataFrame, stm: &DataFrame, t600: &DataFrame) -> Result<DataFrame> {
    if tm.is_empty() {
        anyhow::bail!("Tm data is empty");
    }

    let tm_datetimes = ts_col_to_datetimes(tm.column("TIMESTAMP")?)?;
    let stm_datetimes = ts_col_to_datetimes(stm.column("TIMESTAMP")?)?;
    let t600_ts_col = t600.column("TIMESTAMP")?.clone();
    let t600_datetimes = ts_col_to_datetimes(&t600_ts_col)?;

    // Same caveat as in calculate_stm: skip the per-sensor "T0_source_*"
    // string columns added by the VPD/PAR method.
    let tm_cols: Vec<String> = tm
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T0_") && !c.starts_with("T0_source_"))
        .map(|s| s.to_string())
        .collect();

    // Index Tm by date
    let mut tm_by_date: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (idx, dt) in tm_datetimes.iter().enumerate() {
        let date_key = dt.date_naive().to_string();
        let entry = tm_by_date.entry(date_key).or_default();
        for col in &tm_cols {
            if let Some(val) = tm.column(col)?.f64()?.get(idx) {
                entry.insert(col.clone(), val);
            }
        }
    }

    // Index sTm by date (constant within a day; take first available value)
    let stm_cols: Vec<String> = stm
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("sT0_"))
        .map(|s| s.to_string())
        .collect();

    let mut stm_by_date: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (idx, dt) in stm_datetimes.iter().enumerate() {
        let date_key = dt.date_naive().to_string();
        let entry = stm_by_date.entry(date_key).or_default();
        for col in &stm_cols {
            if let Some(val) = stm.column(col)?.f64()?.get(idx) {
                entry.entry(col.clone()).or_insert(val);
            }
        }
    }

    let mut out: Vec<Series> = vec![t600_ts_col];

    for tm_col in &tm_cols {
        let suffix = &tm_col[3..];
        let stm_col = format!("sT0_{}", suffix);
        let mut tmi_values: Vec<Option<f64>> = vec![None; t600.height()];

        for (idx, dt) in t600_datetimes.iter().enumerate() {
            let hour = dt.hour();
            let minute = dt.minute();
            let cur_date = dt.date_naive().to_string();
            let next_date = (dt.date_naive() + Duration::days(1)).to_string();

            if hour >= 20 {
                // Evening: Tmi = Tm of next day (= end of interpolation)
                if let Some(m) = tm_by_date.get(&next_date).and_then(|m| m.get(tm_col)) {
                    tmi_values[idx] = Some(*m);
                } else if let Some(base) = tm_by_date.get(&cur_date).and_then(|m| m.get(tm_col).copied()) {
                    if let Some(stm_v) = stm_by_date.get(&cur_date).and_then(|m| m.get(&stm_col).copied()) {
                        tmi_values[idx] = Some(base + 24.0 * stm_v);
                    }
                }
            } else if hour < 8 || (hour == 8 && minute < 30) {
                // Night: Tmi = Tm of current day (constant)
                if let Some(m) = tm_by_date.get(&cur_date).and_then(|m| m.get(tm_col)) {
                    tmi_values[idx] = Some(*m);
                }
            } else {
                // Daytime 08:30–19:30: Tmi = Tm_today + step * sTm
                let base_tm = tm_by_date.get(&cur_date).and_then(|m| m.get(tm_col).copied());
                let stm_val = stm_by_date
                    .get(&cur_date)
                    .and_then(|m| m.get(&stm_col).copied());

                if let (Some(base), Some(stm_v)) = (base_tm, stm_val) {
                    let minutes_since_830 =
                        (hour as i64 - 8) * 60 + minute as i64 - 30;
                    let n = minutes_since_830 as f64 / 30.0 + 1.0;
                    tmi_values[idx] = Some(base + n * stm_v);
                }
            }
        }

        let name = format!("T0i_{}", suffix);
        out.push(Series::new(name.as_str(), tmi_values));
    }

    DataFrame::new(out).context("Tmi DataFrame error")
}

// =============================================================================
// Step 8 – K index
// =============================================================================

/// K = (Tmi – T600) / T600, clamped to 0 for negatives.
/// Smallest |T600| accepted as a divisor when forming K.
///
/// T600 is a ΔT built by subtracting two nearly equal temperatures, so
/// catastrophic cancellation leaves floating-point residue: real Niakhar files
/// contain values of 3.55e-15 and 7.11e-15 °C. The old guard tested
/// `t600_v != 0.0`, which lets that residue through — dividing by 3.55e-15
/// yields K ≈ 9e14 and, at α=12.95 / β=1, a sap flow of 1.2e16 next to a
/// median of 0.15. A handful of such points makes every chart unreadable.
///
/// The threshold sits in the 12-order-of-magnitude gap between the residue
/// (~1e-15) and the smallest genuine readings (~1e-3), so it removes the
/// artefacts and nothing else. It is a NUMERICAL floor, not a physical one:
/// choosing a physically meaningful minimum ΔT600 is a calibration decision
/// and belongs in SapFlowParams, not here.
const T600_MIN_DIVISOR: f64 = 1e-9;

pub fn calculate_k(tmi: &DataFrame, t600: &DataFrame) -> Result<DataFrame> {
    let ts_col = tmi.column("TIMESTAMP")?.clone();
    let mut out: Vec<Series> = vec![ts_col];

    let tmi_cols: Vec<String> = tmi
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T0i_"))
        .map(|s| s.to_string())
        .collect();

    for tmi_col in &tmi_cols {
        let suffix = &tmi_col[4..];
        let t600_col_name = format!("T600_{}", suffix);

        let tmi_values = tmi.column(tmi_col)?.f64()?;
        let t600_col = t600
            .column(&t600_col_name)
            .with_context(|| format!("Missing {}", t600_col_name))?
            .f64()?;

        let k_values: Vec<Option<f64>> = (0..tmi.height())
            .map(|i| match (tmi_values.get(i), t600_col.get(i)) {
                (Some(tmi_v), Some(t600_v)) if t600_v.abs() >= T600_MIN_DIVISOR => {
                    let k = (tmi_v - t600_v) / t600_v;
                    Some(k.max(0.0))
                }
                _ => None,
            })
            .collect();

        let name = format!("K_{}", suffix);
        out.push(Series::new(name.as_str(), k_values));
    }

    DataFrame::new(out).context("K DataFrame error")
}

// =============================================================================
// Advanced — Jh → Jhp → Qh → Qd  (per-group aggregation chain)
//
// Reproduces the left-to-right chain of the reference FS_Synthese workbook:
//   Jh  = mean(Fd_<sensor>) across each group's sensors             [L/dm²/h]
//   Jhp = Jh × k_radial                                              [L/dm²/h]
//   Qh  = Jhp × A_sapwood_dm2                                        [L/h]
//   Qd  = ∑(Qh) × Δt/3600 over one calendar day                       [L/day]
// where Δt is the median consecutive timestamp delta of the input frame
// (so the daily integration adapts to 5-min / 30-min / 1-h sampling).
// =============================================================================

use crate::core::types::AdvancedGroup;

/// Jh = mean across each group's sensors of `Fd_<sensor>` (L/dm²/h).
/// Mirrors Excel's `=AVERAGE(...)`: blanks/NaN are silently ignored, so the
/// per-row average is over the *present* sensors only. A row where every
/// selected sensor is null produces null.
pub fn compute_jh(sap_flow: &DataFrame, groups: &[AdvancedGroup]) -> Result<DataFrame> {
    if !sap_flow.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column required in sap_flow");
    }
    if groups.is_empty() {
        anyhow::bail!("Aucun groupe défini — ajoute au moins un arbre ou un pool avant de calculer.");
    }

    let ts_col = sap_flow.column("TIMESTAMP")?.clone();
    let n = sap_flow.height();
    let mut out: Vec<Series> = vec![ts_col];

    for g in groups {
        if g.sensors.is_empty() {
            anyhow::bail!("Groupe '{}' n'a aucun capteur sélectionné.", g.name);
        }
        // Gather the per-sensor Fd ChunkedArrays for this group.
        let fd_arrays: Vec<&ChunkedArray<Float64Type>> = g.sensors.iter()
            .map(|sid| {
                let fd_col_name = format!("Fd_{}", sid);
                sap_flow.column(fd_col_name.as_str())
                    .with_context(|| format!("Colonne {} introuvable dans sap_flow", fd_col_name))
                    .and_then(|s| s.f64().map_err(Into::into))
            })
            .collect::<Result<Vec<_>>>()?;

        let jh_values: Vec<Option<f64>> = (0..n).map(|i| {
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            for arr in &fd_arrays {
                if let Some(v) = arr.get(i) {
                    if v.is_finite() {
                        sum += v;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 { Some(sum / cnt as f64) } else { None }
        }).collect();

        out.push(Series::new(g.name.as_str(), jh_values));
    }

    DataFrame::new(out).context("Jh DataFrame error")
}

/// Jhp = Jh × k_radial per group. Per-group scalar multiplication.
pub fn compute_jhp(jh: &DataFrame, groups: &[AdvancedGroup]) -> Result<DataFrame> {
    if !jh.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column required in Jh");
    }
    let ts_col = jh.column("TIMESTAMP")?.clone();
    let mut out: Vec<Series> = vec![ts_col];
    for g in groups {
        let jh_series = jh.column(g.name.as_str())
            .with_context(|| format!("Colonne {} absente de Jh", g.name))?;
        let jh_ca = jh_series.f64()?;
        let k = g.k_radial;
        let v: Vec<Option<f64>> = jh_ca.into_iter().map(|x| x.map(|y| y * k)).collect();
        out.push(Series::new(g.name.as_str(), v));
    }
    DataFrame::new(out).context("Jhp DataFrame error")
}

/// Qh = Jhp × A_sapwood_dm2 per group (L/h).
pub fn compute_qh(jhp: &DataFrame, groups: &[AdvancedGroup]) -> Result<DataFrame> {
    if !jhp.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column required in Jhp");
    }
    let ts_col = jhp.column("TIMESTAMP")?.clone();
    let mut out: Vec<Series> = vec![ts_col];
    for g in groups {
        let jhp_series = jhp.column(g.name.as_str())
            .with_context(|| format!("Colonne {} absente de Jhp", g.name))?;
        let jhp_ca = jhp_series.f64()?;
        let a = g.a_sapwood_dm2;
        let v: Vec<Option<f64>> = jhp_ca.into_iter().map(|x| x.map(|y| y * a)).collect();
        out.push(Series::new(g.name.as_str(), v));
    }
    DataFrame::new(out).context("Qh DataFrame error")
}

/// Qd = ∑(Qh) × Δt/3600 over each calendar day (L/day).
///
/// Δt is the median consecutive-timestamp delta in seconds, so the
/// integration adapts to 30-min / 1-h / 5-min sampling without manual
/// configuration. A day with fewer rows than expected (= incomplete) gets
/// integrated over whatever rows are present — the missing portion is
/// silently dropped, mirroring Excel's `SUM` behaviour. The completeness
/// stat is exposed through the `n_rows_in_day` column so the user can spot
/// truncated days.
pub fn compute_qd(qh: &DataFrame) -> Result<DataFrame> {
    if !qh.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column required in Qh");
    }
    let ts_vec = ts_col_to_datetimes(qh.column("TIMESTAMP")?)?;
    if ts_vec.len() < 2 {
        anyhow::bail!("Pas assez de lignes pour calculer Qd (au moins 2 timestamps requis).");
    }

    // Median Δt in seconds.
    let mut deltas: Vec<i64> = (1..ts_vec.len())
        .map(|i| (ts_vec[i] - ts_vec[i - 1]).num_seconds())
        .filter(|d| *d > 0)
        .collect();
    if deltas.is_empty() {
        anyhow::bail!("Impossible de déterminer le pas temporel.");
    }
    deltas.sort_unstable();
    let dt_secs = deltas[deltas.len() / 2] as f64; // seconds
    let dt_h = dt_secs / 3600.0;                    // hours

    // Bucket rows by calendar date.
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<chrono::NaiveDate, Vec<usize>> = BTreeMap::new();
    for (i, t) in ts_vec.iter().enumerate() {
        buckets.entry(t.date_naive()).or_default().push(i);
    }

    let qh_col_names: Vec<String> = qh.get_column_names().iter()
        .filter(|c| **c != "TIMESTAMP")
        .map(|s| s.to_string())
        .collect();

    let n_days = buckets.len();
    let mut s_date: Vec<chrono::NaiveDate> = Vec::with_capacity(n_days);
    let mut s_nrows: Vec<i64> = Vec::with_capacity(n_days);
    let mut s_cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n_days); qh_col_names.len()];

    for (date, idxs) in &buckets {
        s_date.push(*date);
        s_nrows.push(idxs.len() as i64);
        for (ci, col_name) in qh_col_names.iter().enumerate() {
            let qh_ca = qh.column(col_name.as_str())?.f64()?;
            let mut sum = 0.0_f64;
            let mut any = false;
            for &i in idxs {
                if let Some(v) = qh_ca.get(i) {
                    if v.is_finite() {
                        sum += v;
                        any = true;
                    }
                }
            }
            s_cols[ci].push(if any { Some(sum * dt_h) } else { None });
        }
    }

    // Build the DataFrame. DATE as a Polars Date series so users can sort/filter.
    let date_ints: Vec<i32> = s_date.iter()
        .map(|d| d.signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32)
        .collect();
    let date_series = Series::new("DATE", date_ints).cast(&DataType::Date)?;

    let mut out: Vec<Series> = vec![date_series, Series::new("n_rows_in_day", s_nrows)];
    for (col_name, vals) in qh_col_names.iter().zip(s_cols.into_iter()) {
        out.push(Series::new(col_name.as_str(), vals));
    }
    DataFrame::new(out).context("Qd DataFrame error")
}

// =============================================================================
// Step 9 – Sap flow (Granier formula)
// =============================================================================

/// Fd = alpha × K^beta
pub fn calculate_sapflow(k: &DataFrame, alpha: f64, beta: f64) -> Result<DataFrame> {
    if !k.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column is required in K DataFrame");
    }

    let ts_col = k.column("TIMESTAMP")?.clone();
    let mut out: Vec<Series> = vec![ts_col];

    let k_cols: Vec<String> = k
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("K_"))
        .map(|s| s.to_string())
        .collect();

    for k_col in &k_cols {
        let k_values = k.column(k_col)?.f64()?;
        let fd_values: Vec<Option<f64>> = k_values
            .into_iter()
            .map(|v| v.map(|kv| alpha * kv.powf(beta)))
            .collect();

        let name = format!("Fd_{}", &k_col[2..]);
        out.push(Series::new(name.as_str(), fd_values));
    }

    DataFrame::new(out).context("Sap flow DataFrame error")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_pattern() {
        assert_eq!(TIMESTAMP_PATTERN.len(), 11);
        // 30+30+60+180+300+30+30+60+180+300+600 = 1800 s = 30 min per cycle.
        assert_eq!(TIMESTAMP_PATTERN.iter().sum::<i64>(), 1800);
    }

    /// Floating-point residue in T600 must not become a K of 1e15.
    ///
    /// Regression test for the real Niakhar 1 file, where 230 rows carried a
    /// T600 of ~3.55e-15 °C (cancellation noise from the ΔT subtraction). The
    /// old `t600_v != 0.0` guard divided by them and produced sap-flow values
    /// of 1.2e16 against a median of 0.15.
    #[test]
    fn test_k_rejects_floating_point_residue_divisors() {
        let ts = Series::new(
            "TIMESTAMP",
            &[0i64, 1_800_000_000, 3_600_000_000, 5_400_000_000, 7_200_000_000],
        )
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
        .unwrap();

        // Row 0: healthy. 1: exact zero. 2 & 3: cancellation residue.
        // 4: small but genuine reading, must survive.
        let t600 = DataFrame::new(vec![
            ts.clone(),
            Series::new("T600_S1", &[2.0f64, 0.0, 3.552713678800501e-15, 7.105427357601002e-15, 1e-3]),
        ])
        .unwrap();
        let tmi = DataFrame::new(vec![
            ts,
            Series::new("T0i_S1", &[6.0f64, 6.0, 6.498333, 3.181875, 2e-3]),
        ])
        .unwrap();

        let k = calculate_k(&tmi, &t600).unwrap();
        let got = k.column("K_S1").unwrap().f64().unwrap();

        assert_eq!(got.get(0), Some(2.0), "(6-2)/2 must still be 2");
        assert_eq!(got.get(1), None, "an exact zero divisor stays rejected");
        assert_eq!(got.get(2), None, "3.55e-15 is residue, not a measurement");
        assert_eq!(got.get(3), None, "7.11e-15 is residue, not a measurement");
        assert_eq!(got.get(4), Some(1.0), "1e-3 is a real reading and must survive");

        // The bug in one assertion: without the guard this was ~1.8e15.
        let max = got.into_no_null_iter().fold(f64::MIN, f64::max);
        assert!(max < 1e3, "no K may explode past 1e3 here, got {max}");
    }

    /// 14-day synthetic dataset where each night has a known ΔT_max.
    /// Verifies the rolling baseline pulls the correct max across the
    /// configured window and emits one row per night-key (matching the
    /// schema produced by the other Tm methods).
    #[test]
    fn moving_window_tracks_per_day_max_with_rolling_smoothing() {
        // 14 days × 48 timestamps/day (30 min granularity). Day 0 starts
        // 1970-01-01 00:00 UTC. With wrap-around night 20h–8h, the row
        // at 02h on calendar day d belongs to night-key day d (the morning
        // it ends). The row at 22h on calendar day d-1 belongs to the
        // SAME night-key day d.
        const DAYS: usize = 14;
        const STEPS_PER_DAY: usize = 48;
        let n = DAYS * STEPS_PER_DAY;

        // Day_max indexed by calendar day d. The ΔT_max for night-key d
        // (the night ending on the morning of calendar day d) is whatever
        // we put at hour 02h of calendar day d (since that hour has the
        // same calendar day as the night-key).
        let night_max_target: Vec<f64> = (0..DAYS).map(|d| 5.0 + d as f64).collect();

        let mut ts_us: Vec<i64> = Vec::with_capacity(n);
        let mut t600_vals: Vec<f64> = Vec::with_capacity(n);
        for d in 0..DAYS {
            for s in 0..STEPS_PER_DAY {
                let ts = ((d * STEPS_PER_DAY + s) as i64) * 30 * 60 * 1_000_000;
                ts_us.push(ts);
                let hour = (s / 2) as u32;
                // Spike at 02h of calendar day d → night_max[d] = target[d].
                // Anywhere else (incl. 22h) → 1.0.
                let v = if hour == 2 { night_max_target[d] } else { 1.0 };
                t600_vals.push(v);
            }
        }

        let ts = Series::new("TIMESTAMP", ts_us)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let t600 = Series::new("T600_sensor", t600_vals);
        let df = DataFrame::new(vec![ts, t600]).unwrap();

        // Use a NON-wrapping nocturnal window (0h–6h) to keep the test
        // assertions simple: night-key D = calendar day D, exactly one
        // key per day. Window = 7 days centred on D.
        let result = calculate_tm_moving_window(&df, 7, 0, 6).unwrap();
        let tm_col = result.column("T0_sensor").unwrap().f64().unwrap();

        // Expected rolling max over [d-3 .. d+3] clamped to [0, DAYS-1].
        let expected = |d: usize| -> f64 {
            let lo = d.saturating_sub(3);
            let hi = (d + 3).min(DAYS - 1);
            night_max_target[lo..=hi].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        // One row per night-key; no wrap, so exactly DAYS keys.
        assert_eq!(result.height(), DAYS,
            "expected {} rows, got {}", DAYS, result.height());

        // Walk every output row to make any off-by-one obvious.
        for d in 0..DAYS {
            let got = tm_col.get(d).unwrap();
            let exp = expected(d);
            assert!((got - exp).abs() < 1e-9,
                "row {}: got {}, expected {}", d, got, exp);
        }
    }

    #[test]
    fn linear_fit_recovers_known_line() {
        // y = 2x + 5 — OLS should recover (2, 5) exactly on noiseless data.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 5.0).collect();
        let (a, b) = linear_fit(&xs, &ys).unwrap();
        assert!((a - 2.0).abs() < 1e-9);
        assert!((b - 5.0).abs() < 1e-9);
    }

    #[test]
    fn linear_fit_returns_none_on_constant_x() {
        let xs = vec![5.0, 5.0, 5.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(linear_fit(&xs, &ys).is_none());
    }

    /// Synthetic night with a known trend AND a clean ΔT_max envelope.
    /// We expect double-regression to recover a Tm close to the no-flow
    /// envelope mean rather than just the global max (which would be
    /// biased by the highest spike).
    #[test]
    fn double_regression_recovers_no_flow_envelope() {
        // Build 3 nights of 20 nocturnal points each. For night 1 we
        // engineer the data so the upper envelope mean is known.
        //
        // Trend: ΔT(t) = 8.0 + 0.0·t (constant baseline). On top, half
        // the points are "above the trend" at ~9.0, half are "below" at
        // ~7.0 (representing residual transpiration / slow recharge).
        // The double regression should:
        //   - step 1: fit a flat line at y ≈ 8.0
        //   - step 2: keep the upper points (~9.0)
        //   - return Tm ≈ 9.0 (mean of upper points)
        // A naive "simple max" would return 9.5 (a single high spike we
        // sneak in). DR's mean should be much closer to 9.0.
        const POINTS_PER_NIGHT: usize = 24;
        const NIGHTS: usize = 3;

        let mut ts_us: Vec<i64> = Vec::new();
        let mut t600_vals: Vec<f64> = Vec::new();

        // Anchor at 1970-01-05 to have full 24h cycles. Each night we
        // span 20h day d → 8h day d+1, every 30 minutes. Evening = day d.
        for d in 0..NIGHTS {
            let day_offset_secs = (4 + d) as i64 * 86400; // start at Jan 5
            for s in 0..POINTS_PER_NIGHT {
                // 30-min steps from 20h00 of day d.
                let t = day_offset_secs + 20 * 3600 + (s as i64) * 30 * 60;
                ts_us.push(t * 1_000_000);
                let v = if s == POINTS_PER_NIGHT - 1 {
                    // One final spike of 9.5 to test the "mean ≠ max" bit.
                    9.5
                } else if s % 2 == 0 {
                    9.0  // upper envelope
                } else {
                    7.0  // lower (residual flow)
                };
                t600_vals.push(v);
            }
        }

        let ts = Series::new("TIMESTAMP", ts_us)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let t600 = Series::new("T600_sensor", t600_vals);
        let df = DataFrame::new(vec![ts, t600]).unwrap();

        let result = calculate_tm_double_regression(&df, 3).unwrap();
        let tm_col = result.column("T0_sensor").unwrap().f64().unwrap();

        // We should get one row per night. With 3 nights, expect 3 rows.
        assert_eq!(result.height(), NIGHTS);

        for d in 0..NIGHTS {
            let tm = tm_col.get(d).expect("Tm should be defined");
            // Expected: mean of upper-points (9.0) plus the single 9.5
            // spike. With ~12 points at 9.0 and 1 spike at 9.5,
            // mean ≈ (12·9.0 + 9.5)/13 ≈ 9.038. Allow ±0.2 of slack.
            assert!(
                (tm - 9.04).abs() < 0.2,
                "night {}: got Tm={}, expected ~9.04 (upper-envelope mean)",
                d, tm
            );
            // Sanity: must be strictly less than the global max (9.5)
            // and strictly greater than 8.0 (the trend), proving the
            // algorithm is doing more than just max() or mean().
            assert!(tm < 9.5, "Tm {} should be < global max 9.5", tm);
            assert!(tm > 8.0, "Tm {} should be > trend 8.0", tm);
        }
    }

    /// Full pipeline smoke test for the DataEnv (VpdPar) Tm method.
    /// Builds a synthetic 7-day dataset with realistic environmental
    /// conditions (low VPD + low PAR at night, high values during the
    /// day), runs Tm via VPD/PAR, then chains the next two pipeline
    /// steps (sTm, Tmi). Verifies the whole chain produces non-empty
    /// frames with the expected column shapes — guards against the
    /// `T0_source_*` regression we hit earlier.
    #[test]
    fn data_env_pipeline_runs_through_tm_stm_tmi() {
        use crate::core::types::VpdParConfig;
        use crate::core::vpd_par::calculate_tm_vpd_par;

        const DAYS: usize = 7;
        const STEPS_PER_DAY: usize = 48; // 30-min granularity

        // Build TIMESTAMPS for both DataFrames (same grid, same column).
        let n = DAYS * STEPS_PER_DAY;
        let mut ts_us: Vec<i64> = Vec::with_capacity(n);
        for d in 0..DAYS {
            for s in 0..STEPS_PER_DAY {
                let ts = ((d * STEPS_PER_DAY + s) as i64) * 30 * 60 * 1_000_000;
                ts_us.push(ts);
            }
        }

        // T600: nocturnal plateau at 9.5 (ΔTmax-like) with tiny symmetric
        // noise → CV well under 1%. Daytime ramps up to ~5 (active
        // transpiration → far from Tm). Symmetric noise (sin) so the
        // mean of any 4-point window stays right at 9.5.
        let mut t600_vals: Vec<f64> = Vec::with_capacity(n);
        for s_global in 0..n {
            let s = s_global % STEPS_PER_DAY;
            let hour = (s / 2) as u32;
            let nocturnal = hour >= 20 || hour < 6;
            let v = if nocturnal {
                // ±0.005 noise centred on 9.5 → CV ≈ 5e-4, well below 1%.
                9.5 + 0.005 * ((s_global as f64) * 0.7).sin()
            } else {
                2.0 + (hour as f64) * 0.3
            };
            t600_vals.push(v);
        }

        // Env: matches the t600 timestamps exactly. Low VPD + low PAR at
        // night, high during day. Both columns named generically so the
        // VpdParConfig can target them.
        let mut vpd_vals: Vec<f64> = Vec::with_capacity(n);
        let mut par_vals: Vec<f64> = Vec::with_capacity(n);
        for s_global in 0..n {
            let s = s_global % STEPS_PER_DAY;
            let hour = (s / 2) as u32;
            let nocturnal = hour >= 20 || hour < 6;
            // Low VPD at night (well under 0.1 kPa), high during day.
            vpd_vals.push(if nocturnal { 0.05 } else { 1.5 });
            // PAR: 0 at night, ramping to ~1500 at midday.
            par_vals.push(if nocturnal {
                0.0
            } else {
                let t = (hour as f64 - 6.0) / 14.0; // 0..1 across daylight
                1500.0 * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.0)
            });
        }

        let ts_t600 = Series::new("TIMESTAMP", ts_us.clone())
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let t600 = DataFrame::new(vec![
            ts_t600,
            Series::new("T600_sensor", t600_vals),
        ]).unwrap();

        let ts_env = Series::new("TIMESTAMP", ts_us)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let env = DataFrame::new(vec![
            ts_env,
            Series::new("VPD", vpd_vals),
            Series::new("PAR", par_vals),
        ]).unwrap();

        // --- Step 5: Tm via VPD/PAR ----------------------------------
        let cfg = VpdParConfig {
            vpd_column: "VPD".to_string(),
            radiation_column: "PAR".to_string(),
            timestamp_column: "TIMESTAMP".to_string(),
            ..VpdParConfig::default()
        };
        // `calculate_tm_vpd_par` gained a per-filter stats return value; this
        // test still bound the whole tuple as if it were the DataFrame, which
        // stopped the entire `core` test module from compiling.
        let (tm, _vpd_par_stats) = calculate_tm_vpd_par(&t600, &env, &cfg)
            .expect("Tm via VPD/PAR should succeed");

        // Output shape: TIMESTAMP + T0_sensor + T0_source_sensor.
        let names: Vec<&str> = tm.get_column_names();
        assert!(names.contains(&"TIMESTAMP"),
            "Tm DataFrame missing TIMESTAMP column");
        assert!(names.contains(&"T0_sensor"),
            "Tm DataFrame missing T0_sensor column");
        assert!(names.contains(&"T0_source_sensor"),
            "Tm DataFrame missing T0_source_sensor (provenance) column");
        assert!(tm.height() >= 5,
            "expected >=5 nights of Tm output, got {}", tm.height());

        // The T0_source column must be a string with values from the
        // documented set. If we get any "NoValidNight" the test data
        // is misconfigured — the entire week was set up to be valid.
        let src_col = tm.column("T0_source_sensor").unwrap();
        assert!(matches!(src_col.dtype(), DataType::String),
            "T0_source_sensor should be String, got {:?}", src_col.dtype());
        let any_no_valid = src_col.str().unwrap()
            .into_iter()
            .any(|s| matches!(s, Some("NoValidNight")));
        assert!(!any_no_valid,
            "no night should be marked NoValidNight on this clean synthetic data");

        // Tm values: every night should be near the 9.5 plateau (algo
        // takes mean over the most stable 2h window inside no-flow).
        let tm_col = tm.column("T0_sensor").unwrap().f64().unwrap();
        for v in tm_col.into_iter().flatten() {
            assert!(
                (v - 9.5).abs() < 0.5,
                "Tm value {} too far from expected ~9.5 plateau", v
            );
        }

        // --- Step 6: sTm — must NOT trip on the T0_source_* string col.
        // (This is the bug we fixed earlier: the filter has to skip
        // string-typed source columns when iterating numeric T0_*.)
        let stm = calculate_stm(&tm, &t600)
            .expect("sTm should succeed on a VpdPar Tm DataFrame");
        let stm_names: Vec<&str> = stm.get_column_names();
        assert!(stm_names.contains(&"sT0_sensor"),
            "sTm DataFrame missing sT0_sensor column (got {:?})", stm_names);

        // --- Step 7: Tmi (interpolation) ----------------------------
        let tmi = calculate_tmi(&tm, &stm, &t600)
            .expect("Tmi should succeed");
        let tmi_names: Vec<&str> = tmi.get_column_names();
        assert!(tmi_names.contains(&"T0i_sensor"),
            "Tmi DataFrame missing T0i_sensor column (got {:?})", tmi_names);
        assert_eq!(tmi.height(), t600.height(),
            "Tmi should have one row per t600 timestamp");
    }

    #[test]
    fn test_sapflow_formula() {
        // K=0 → Fd=0
        let ts = Series::new("TIMESTAMP", vec![0i64])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let k = Series::new("K_test", vec![Some(0.0f64)]);
        let df = DataFrame::new(vec![ts, k]).unwrap();
        let result = calculate_sapflow(&df, 12.95, 1.231).unwrap();
        let fd = result.column("Fd_test").unwrap().f64().unwrap();
        assert_eq!(fd.get(0), Some(0.0));
    }
}
