//! VPD/PAR method for Tm (ΔTmax) determination.
//!
//! Implements the Oishi et al. 2008/2016 algorithm, with Rabbel et al. 2016 and
//! Ward et al. 2017 (TRACC) extensions. The core idea: the Granier nocturnal-
//! zero-flow assumption is unreliable in many biomes (residual nocturnal
//! transpiration, hydraulic recharge, growth). Instead, ΔTmax is identified
//! only when three cumulative physical no-flow conditions are met:
//!
//! 1. Night — low radiation (PAR or Rg below threshold) AND clock-hour in
//!    the nocturnal window (belt-and-braces: avoids false positives during
//!    overcast daytime at Sahelian sites).
//! 2. Low VPD — below threshold (no evaporative demand).
//! 3. Thermal stability — coefficient of variation of ΔT (= T600 here) over a
//!    rolling window smaller than threshold, AND at least
//!    `min_consecutive_pts` valid points in a row.
//!
//! For each night, the most stable window (min CV) is kept; Tm is the mean ΔT
//! over that window. Nights without a valid window are linearly interpolated
//! (TRACC), with constant-hold extrapolation at the series edges.
//!
//! Output shape: one row per night (indexed by the date of the morning end of
//! the nocturnal window), two columns per sensor: `T0_<sensor>` and
//! `T0_source_<sensor>` ∈ { "Valid", "Interpolated", "NoValidNight" }.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use chrono::{Duration, NaiveDate, NaiveDateTime, Timelike};
use polars::prelude::*;

use crate::core::timestamp_utils::ts_col_to_datetimes;
use crate::core::types::{VpdParConfig, VpdParStats};

/// Per-night result for one sensor.
#[derive(Debug, Clone)]
struct NightOutcome {
    tm: f64,
    source: TmSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TmSource {
    Valid,
    /// No env-qualified stable window for this night (e.g. dry-season nights
    /// where VPD never drops below the threshold). Anchored to the night's OWN
    /// nocturnal ΔTmax instead of being interpolated from distant valid nights,
    /// which used to drag the envelope far below the real plateau.
    Fallback,
    Interpolated,
    NoValidNight,
}

impl TmSource {
    fn as_str(&self) -> &'static str {
        match self {
            TmSource::Valid => "Valid",
            TmSource::Fallback => "Fallback",
            TmSource::Interpolated => "Interpolated",
            TmSource::NoValidNight => "NoValidNight",
        }
    }
}

/// Entry point. Builds a Tm DataFrame from the T600 series and environmental
/// data (VPD + PAR/Rg on a shared 30-min time grid).
pub fn calculate_tm_vpd_par(
    t600: &DataFrame,
    env: &DataFrame,
    cfg: &VpdParConfig,
) -> Result<(DataFrame, VpdParStats)> {
    cfg.validate().map_err(anyhow::Error::msg)?;

    // Sensor columns in t600 are the T600_<sensor> series (plus TIMESTAMP).
    let t600_sensor_cols: Vec<String> = t600
        .get_column_names()
        .iter()
        .filter(|c| c.starts_with("T600_"))
        .map(|s| s.to_string())
        .collect();
    if t600_sensor_cols.is_empty() {
        bail!("No T600_ sensor columns found in the T600 DataFrame");
    }

    // Align T600 and env on timestamp by building a hashmap keyed on the
    // t600 timestamp. Env is assumed to be at ≤ 30-min granularity; for each
    // t600 timestamp we pick the env row with the same timestamp, falling back
    // to the nearest previous env row within 30 min.
    // `ts_col_to_datetimes` returns DateTime<Utc>; this module's helpers all
    // work on NaiveDateTime, so normalize at the boundary.
    let t600_datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(t600.column("TIMESTAMP")?)?
        .into_iter()
        .map(|d| d.naive_utc())
        .collect();
    // The env loader normalizes the timestamp column to "TIMESTAMP" — fall
    // back to the user-provided name for backward compatibility.
    let env_ts_col = env.column("TIMESTAMP")
        .or_else(|_| env.column(&cfg.timestamp_column))?;
    let env_datetimes: Vec<NaiveDateTime> = ts_col_to_datetimes(env_ts_col)?
        .into_iter()
        .map(|d| d.naive_utc())
        .collect();
    if t600_datetimes.is_empty() {
        bail!("T600 DataFrame has no timestamps");
    }
    if env_datetimes.is_empty() {
        bail!("Env DataFrame has no timestamps");
    }

    // Ensure the env columns exist before we get deep into the algorithm.
    let vpd_series = extract_f64_column(env, &cfg.vpd_column)?;

    // Resolve PAR / Rg columns. The new schema uses dedicated `par_column`
    // and `solar_column`; the legacy single `radiation_column` is routed
    // to the matching slot via `use_par` for backward compat.
    let par_col_name = if !cfg.par_column.is_empty() {
        cfg.par_column.clone()
    } else if !cfg.radiation_column.is_empty() && cfg.use_par {
        cfg.radiation_column.clone()
    } else {
        String::new()
    };
    let solar_col_name = if !cfg.solar_column.is_empty() {
        cfg.solar_column.clone()
    } else if !cfg.radiation_column.is_empty() && !cfg.use_par {
        cfg.radiation_column.clone()
    } else {
        String::new()
    };
    if par_col_name.is_empty() && solar_col_name.is_empty() {
        bail!("Aucune colonne de rayonnement (PAR ou Rg) sélectionnée");
    }
    let par_series_opt = if par_col_name.is_empty() {
        None
    } else {
        Some(extract_f64_column(env, &par_col_name)?)
    };
    let solar_series_opt = if solar_col_name.is_empty() {
        None
    } else {
        Some(extract_f64_column(env, &solar_col_name)?)
    };

    // Precompute env lookup: map timestamp (i64 ms) → (vpd, par_opt, solar_opt).
    // Each radiation slot is `None` when the user didn't pick that column.
    let mut env_lookup: BTreeMap<i64, (Option<f64>, Option<f64>, Option<f64>)> = BTreeMap::new();
    for (i, dt) in env_datetimes.iter().enumerate() {
        let key = dt.and_utc().timestamp_millis();
        let par_v = par_series_opt.as_ref().and_then(|s| s.get(i));
        let solar_v = solar_series_opt.as_ref().and_then(|s| s.get(i));
        env_lookup.insert(key, (vpd_series.get(i), par_v, solar_v));
    }

    // Build per-row env flags aligned to the t600 timeline.
    let n = t600_datetimes.len();
    let mut is_night_flags = vec![false; n];
    let mut is_low_vpd_flags = vec![false; n];
    let mut stats = VpdParStats { n_total: n, ..Default::default() };
    for (i, dt) in t600_datetimes.iter().enumerate() {
        let hour = dt.hour();
        let clock_in_window = if cfg.night_start_hour < cfg.night_end_hour {
            hour >= cfg.night_start_hour && hour < cfg.night_end_hour
        } else {
            // wraps midnight (e.g. 20 → 6)
            hour >= cfg.night_start_hour || hour < cfg.night_end_hour
        };

        let key = dt.and_utc().timestamp_millis();
        // Pick the env row with same key, or the closest prior one within 30 min.
        let env_row = env_lookup
            .range(..=key)
            .next_back()
            .filter(|(k, _)| key - **k <= 30 * 60 * 1000)
            .map(|(_, v)| *v);
        let (vpd_opt, par_opt, solar_opt) = env_row.unwrap_or((None, None, None));

        // Cumulative AND across whichever radiation columns were chosen:
        //   - PAR set + value < par_threshold   → contributes "low light"
        //   - Rg  set + value < solar_threshold → contributes "low light"
        // The row is "no flow" only if EVERY active filter passes. If
        // the env row is missing a value (NaN / outside 30-min window),
        // the corresponding filter fails — conservative on purpose.
        let par_active = par_series_opt.is_some();
        let solar_active = solar_series_opt.is_some();
        let par_ok = if par_active {
            par_opt.map(|r| r < cfg.par_threshold).unwrap_or(false)
        } else {
            true
        };
        let solar_ok = if solar_active {
            solar_opt.map(|r| r < cfg.solar_threshold).unwrap_or(false)
        } else {
            true
        };
        let rad_ok = par_ok && solar_ok;
        is_night_flags[i] = clock_in_window && rad_ok;

        // Clip negative VPD to 0 (sensor artefact) before comparing.
        let vpd_clipped = vpd_opt.map(|v| v.max(0.0));
        is_low_vpd_flags[i] = vpd_clipped.map(|v| v < cfg.vpd_threshold).unwrap_or(false);

        // Diagnostic counters: cumulative drops along the filter chain.
        if clock_in_window { stats.n_in_clock_window += 1; }
        if rad_ok { stats.n_rad_ok += 1; }
        if is_night_flags[i] { stats.n_night_flag += 1; }
        if is_low_vpd_flags[i] { stats.n_vpd_ok += 1; }
        if is_night_flags[i] && is_low_vpd_flags[i] { stats.n_env_passed += 1; }
    }

    // Estimate the time step in hours from the first few intervals. Assume
    // regular spacing (validated in data_loader). Default 0.5h if unknown.
    let dt_hours = estimate_step_hours(&t600_datetimes).unwrap_or(0.5);
    let window_pts = ((cfg.stability_window_h / dt_hours).round() as usize).max(cfg.min_consecutive_pts);

    // Group indices by night. Convention from the spec: a night spans
    // [20h of day N, 6h of day N+1), indexed on day N+1 (the morning key).
    let nights = group_by_night(&t600_datetimes, &is_night_flags, cfg);
    if nights.is_empty() {
        bail!("VpdPar: no complete nights found (need ≥ 24 h of data covering the nocturnal window)");
    }

    // Per-sensor result columns we will concatenate at the end.
    let mut out_columns: Vec<Series> = Vec::with_capacity(1 + t600_sensor_cols.len() * 2);

    // Timestamp column (one entry per night, at 00:00 of the morning key).
    let ts_values: Vec<i64> = nights
        .keys()
        .map(|date| {
            date.and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_micros()
        })
        .collect();
    let ts_series = Series::new("TIMESTAMP".into(), ts_values);
    let ts_series = ts_series
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
        .unwrap_or(ts_series);
    out_columns.push(ts_series);

    // Process each sensor.
    for sensor_col in &t600_sensor_cols {
        let values = extract_f64_column(t600, sensor_col)?;

        let mut per_night: Vec<Option<NightOutcome>> = Vec::with_capacity(nights.len());
        for (_night_key, night_indices) in &nights {
            let outcome = pick_best_window(
                night_indices,
                &values,
                &is_night_flags,
                &is_low_vpd_flags,
                window_pts,
                cfg.min_consecutive_pts,
                cfg.cv_threshold,
            )
            // No env-qualified window → fall back to this night's own nocturnal
            // ΔTmax rather than leaving it to be interpolated from distant (and,
            // in the dry season, lower) valid nights — the cause of the "low
            // envelope". Only nights with no finite ΔT at all stay None and get
            // interpolated.
            .or_else(|| nightly_max(night_indices, &values)
                .map(|mx| NightOutcome { tm: mx, source: TmSource::Fallback }));
            per_night.push(outcome);
        }

        // Interpolate missing nights.
        let finalized = interpolate_missing(cfg.interpolation_method, &per_night, cfg.spline_smooth_window);

        // Extract Tm + source columns.
        let tm_vals: Vec<Option<f64>> = finalized
            .iter()
            .map(|o| o.as_ref().map(|n| n.tm))
            .collect();
        let source_vals: Vec<String> = finalized
            .iter()
            .map(|o| {
                let s = o.as_ref()
                    .map(|n| n.source)
                    .unwrap_or(TmSource::NoValidNight);
                match s {
                    TmSource::Valid => stats.n_valid_nights += 1,
                    // Fallback nights are anchored to real data (the night's own
                    // ΔTmax), not neighbour-interpolated; count them with the
                    // "valid" tally for the diagnostic so the no-data count stays
                    // meaningful.
                    TmSource::Fallback => stats.n_valid_nights += 1,
                    TmSource::Interpolated => stats.n_interpolated_nights += 1,
                    TmSource::NoValidNight => stats.n_no_valid_nights += 1,
                }
                s.as_str().to_string()
            })
            .collect();

        // Rename: T600_<sensor> → T0_<sensor>
        let base = sensor_col.strip_prefix("T600_").unwrap_or(sensor_col);
        let tm_name = format!("T0_{}", base);
        let src_name = format!("T0_source_{}", base);
        out_columns.push(Series::new(tm_name.as_str().into(), tm_vals));
        out_columns.push(Series::new(src_name.as_str().into(), source_vals));
    }

    // Warn via the errors field is handled upstream. Here we just build the DF.
    let df = DataFrame::new(out_columns).context("VpdPar: DataFrame assembly failed")?;
    Ok((df, stats))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cast a column to f64 and return an owned Vec<Option<f64>>.
struct Col {
    data: Vec<Option<f64>>,
}

impl Col {
    fn get(&self, i: usize) -> Option<f64> {
        self.data.get(i).copied().flatten()
    }
}

fn extract_f64_column(df: &DataFrame, name: &str) -> Result<Col> {
    let s = df
        .column(name)
        .with_context(|| format!("Column '{}' not found in DataFrame", name))?;
    let casted = s
        .cast(&DataType::Float64)
        .with_context(|| format!("Cannot cast column '{}' to f64", name))?;
    let ca = casted.f64().context("f64 chunked array expected")?;
    let data: Vec<Option<f64>> = ca.into_iter().collect();
    Ok(Col { data })
}

/// Estimate the median time step between consecutive datetimes in hours.
fn estimate_step_hours(datetimes: &[chrono::NaiveDateTime]) -> Option<f64> {
    if datetimes.len() < 2 {
        return None;
    }
    let sample: Vec<f64> = datetimes
        .windows(2)
        .take(100)
        .map(|pair| (pair[1] - pair[0]).num_seconds() as f64 / 3600.0)
        .filter(|h| *h > 0.0)
        .collect();
    if sample.is_empty() {
        return None;
    }
    let mut sorted = sample.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(sorted[sorted.len() / 2])
}

/// Group t600 row indices by night key (= morning date).
/// Convention: every row where hour ∈ [night_start, 24) belongs to the night
/// that ends the NEXT morning; rows with hour < night_end belong to the night
/// that ends on the SAME morning. Rows outside the nocturnal window are
/// skipped entirely (their night key doesn't matter for Tm selection).
fn group_by_night(
    datetimes: &[chrono::NaiveDateTime],
    is_night: &[bool],
    cfg: &VpdParConfig,
) -> BTreeMap<NaiveDate, Vec<usize>> {
    let mut groups: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for (i, dt) in datetimes.iter().enumerate() {
        // We only consider points that COULD be candidates (in the clock
        // window). Low-radiation flag is already folded into is_night, but we
        // still want to keep rows where clock is in the window even if radia-
        // tion exceeds the threshold — otherwise group_by_night would lose the
        // structure on cloudy nights. Decision: keep clock-window rows only.
        let hour = dt.hour();
        let clock_in_window = if cfg.night_start_hour < cfg.night_end_hour {
            hour >= cfg.night_start_hour && hour < cfg.night_end_hour
        } else {
            hour >= cfg.night_start_hour || hour < cfg.night_end_hour
        };
        if !clock_in_window {
            continue;
        }

        let date = dt.date();
        let morning_key = if hour >= cfg.night_start_hour {
            date + Duration::days(1)
        } else {
            date
        };
        groups.entry(morning_key).or_default().push(i);

        // Silence unused warning on is_night in this helper — it's consumed
        // downstream by pick_best_window. (Keep arg for signature symmetry.)
        let _ = is_night;
    }
    groups
}

/// For one night, find the most stable window: contiguous run of ≥
/// min_consecutive points that all satisfy (is_night AND is_low_vpd), with
/// CV(ΔT) below cv_threshold, picking the window with MINIMUM CV and taking
/// the mean ΔT over it. Returns None if no window qualifies.
fn pick_best_window(
    night_indices: &[usize],
    values: &Col,
    is_night: &[bool],
    is_low_vpd: &[bool],
    window_pts: usize,
    min_consecutive: usize,
    cv_threshold: f64,
) -> Option<NightOutcome> {
    // Keep only indices that pass both env flags AND have a finite value.
    let eligible: Vec<usize> = night_indices
        .iter()
        .copied()
        .filter(|&i| {
            is_night[i]
                && is_low_vpd[i]
                && values.get(i).map(|v| v.is_finite()).unwrap_or(false)
        })
        .collect();
    if eligible.len() < min_consecutive {
        return None;
    }

    // Split eligible into maximal consecutive runs (consecutive in the raw
    // timeline — we rely on the index difference being 1 for "consecutive").
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for &idx in &eligible {
        if current.is_empty() || idx == *current.last().unwrap() + 1 {
            current.push(idx);
        } else {
            runs.push(std::mem::take(&mut current));
            current.push(idx);
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }

    // Slide a window of `window_pts` over each run; compute CV; track minimum.
    let mut best: Option<(f64, f64)> = None; // (cv, mean)
    for run in &runs {
        if run.len() < min_consecutive {
            continue;
        }
        let effective_window = window_pts.min(run.len()).max(min_consecutive);
        for start in 0..=(run.len() - effective_window) {
            let slice = &run[start..start + effective_window];
            let xs: Vec<f64> = slice
                .iter()
                .filter_map(|&i| values.get(i))
                .collect();
            if xs.len() < min_consecutive {
                continue;
            }
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            if mean.abs() < 1e-12 {
                continue; // avoid CV blowup
            }
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
            let std = var.sqrt();
            let cv = (std / mean).abs();
            if cv < cv_threshold && best.as_ref().map(|(c, _)| cv < *c).unwrap_or(true) {
                best = Some((cv, mean));
            }
        }
    }

    best.map(|(_cv, mean)| NightOutcome {
        tm: mean,
        source: TmSource::Valid,
    })
}

/// The night's own nocturnal ΔTmax — the maximum finite ΔT over the night's
/// clock-window rows. Used as the fallback Tm when no env-qualified stable
/// window exists, so a dry-season night anchors to its real plateau instead of
/// being interpolated low.
fn nightly_max(night_indices: &[usize], values: &Col) -> Option<f64> {
    night_indices
        .iter()
        .filter_map(|&i| values.get(i))
        .filter(|v| v.is_finite())
        .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
}

/// Linear interpolation of missing nights between valid values. Constant-hold
/// Fills the `NoValidNight` slots according to the chosen strategy. Six
/// methods are supported — see `NightInterpolationMethod` for definitions.
/// The 1st arg is the method to apply; the 2nd is the per-night outcomes
/// (`None` = no stable window found for that night). Returns one
/// `Option<NightOutcome>` per input position, where `None` means there was
/// never a valid neighbour to interpolate from (whole series invalid).
fn interpolate_missing(
    method: crate::core::types::NightInterpolationMethod,
    per_night: &[Option<NightOutcome>],
    spline_smooth_window: usize,
) -> Vec<Option<NightOutcome>> {
    use crate::core::types::NightInterpolationMethod as M;
    let n = per_night.len();
    let mut result: Vec<Option<NightOutcome>> = per_night.to_vec();

    // Collect (index, value) pairs of valid nights. If empty, nothing to do.
    let valid: Vec<(usize, f64)> = per_night
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.as_ref().map(|x| (i, x.tm)))
        .collect();
    if valid.is_empty() {
        return result;
    }

    // Helper to mark a position interpolated with a given value.
    let mark = |out: &mut Vec<Option<NightOutcome>>, i: usize, v: f64| {
        out[i] = Some(NightOutcome { tm: v, source: TmSource::Interpolated });
    };

    match method {
        M::Linear => {
            // Same as the original TRACC behaviour: nearest prev + next, then
            // y = y0 + t * (y1 - y0). Hold-left / hold-right at series edges.
            let idxs: Vec<usize> = valid.iter().map(|(i, _)| *i).collect();
            for i in 0..n {
                if result[i].is_some() { continue; }
                let prev = idxs.iter().rev().find(|&&v| v < i).copied();
                let next = idxs.iter().find(|&&v| v > i).copied();
                let value = match (prev, next) {
                    (Some(p), Some(nx)) => {
                        let y0 = per_night[p].as_ref().unwrap().tm;
                        let y1 = per_night[nx].as_ref().unwrap().tm;
                        let t = (i - p) as f64 / (nx - p) as f64;
                        y0 + t * (y1 - y0)
                    }
                    (Some(p), None) => per_night[p].as_ref().unwrap().tm,
                    (None, Some(nx)) => per_night[nx].as_ref().unwrap().tm,
                    (None, None) => continue,
                };
                mark(&mut result, i, value);
            }
        }

        M::Nearest => {
            // Snap to the closest valid neighbour in time (no interpolation).
            // Tie → pick the previous one.
            for i in 0..n {
                if result[i].is_some() { continue; }
                let mut best: Option<(usize, f64)> = None;
                for &(j, v) in &valid {
                    let d = (i as isize - j as isize).abs();
                    match best {
                        None => best = Some((d as usize, v)),
                        Some((bd, _)) if (d as usize) < bd => best = Some((d as usize, v)),
                        _ => {}
                    }
                }
                if let Some((_, v)) = best { mark(&mut result, i, v); }
            }
        }

        M::Hold => {
            // Last-observation-carried-forward, with backward fill for
            // positions before the first valid night.
            let first_valid_value = valid[0].1;
            let mut last_value = first_valid_value;
            let mut last_valid_idx: Option<usize> = None;
            for i in 0..n {
                if let Some(o) = result[i].as_ref() {
                    last_value = o.tm;
                    last_valid_idx = Some(i);
                } else if last_valid_idx.is_some() {
                    mark(&mut result, i, last_value);
                } else {
                    // before any valid → use the first valid value
                    mark(&mut result, i, first_valid_value);
                }
            }
        }

        M::Spline => {
            // Natural cubic spline through the valid (x=index, y=tm) points,
            // evaluated at the missing indices. With < 2 valids we fall back
            // to constant; with exactly 2 we fall back to linear.
            //
            // `spline_smooth_window` (in nights) optionally pre-smooths the
            // valid y-values with a centred moving average before fitting, so
            // the curve no longer passes through every noisy night but follows
            // the trend. 1 (or 0) = no smoothing (interpolating spline).
            let xs: Vec<f64> = valid.iter().map(|(i, _)| *i as f64).collect();
            let raw_ys: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
            let ys = smooth_moving_average(&raw_ys, spline_smooth_window);
            let spline = NaturalCubicSpline::build(&xs, &ys);
            for i in 0..n {
                if result[i].is_some() { continue; }
                let v = spline.eval(i as f64);
                mark(&mut result, i, v);
            }
        }

        M::LinearRegression => {
            // Global OLS y = a + b * x on the valid nights, evaluated at missing.
            let xs: Vec<f64> = valid.iter().map(|(i, _)| *i as f64).collect();
            let ys: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
            if let Some((a, b)) = ols_fit(&xs, &ys) {
                for i in 0..n {
                    if result[i].is_some() { continue; }
                    mark(&mut result, i, a + b * (i as f64));
                }
            } else {
                // Fall back to hold if OLS is degenerate (single point).
                let v = ys[0];
                for i in 0..n {
                    if result[i].is_some() { continue; }
                    mark(&mut result, i, v);
                }
            }
        }

        M::DoubleRegression => {
            // Pasqualotto-style two-pass. Pass 1: OLS on all valids. Pass 2:
            // keep only points ABOVE the line (upper envelope — the no-flow
            // ceiling we're trying to reconstruct) and refit. Evaluate the
            // second line at missing positions. Falls back to single OLS
            // when the upper envelope is too sparse (< 3 points).
            let xs: Vec<f64> = valid.iter().map(|(i, _)| *i as f64).collect();
            let ys: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
            let (a, b) = match ols_fit(&xs, &ys) {
                Some(p) => p,
                None => {
                    let v = ys[0];
                    for i in 0..n {
                        if result[i].is_some() { continue; }
                        mark(&mut result, i, v);
                    }
                    return result;
                }
            };
            // Upper envelope = points strictly above the pass-1 line.
            let xs_up: Vec<f64> = xs.iter().zip(ys.iter())
                .filter(|(x, y)| **y >= a + b * **x).map(|(x, _)| *x).collect();
            let ys_up: Vec<f64> = xs.iter().zip(ys.iter())
                .filter(|(x, y)| **y >= a + b * **x).map(|(_, y)| *y).collect();
            let (a2, b2) = if xs_up.len() >= 3 {
                ols_fit(&xs_up, &ys_up).unwrap_or((a, b))
            } else { (a, b) };
            for i in 0..n {
                if result[i].is_some() { continue; }
                mark(&mut result, i, a2 + b2 * (i as f64));
            }
        }
    }

    result
}

/// Least-squares straight-line fit `y = a + b·x`. Returns None when all
/// xs collapse to one point (slope undefined).
fn ols_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    let n = xs.len();
    if n < 2 { return None; }
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0; let mut sxy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        sxx += dx * dx;
        sxy += dx * (y - mean_y);
    }
    if sxx == 0.0 { return None; }
    let b = sxy / sxx;
    let a = mean_y - b * mean_x;
    Some((a, b))
}

/// Centred moving-average smoother used to soften the spline control points.
/// `window` is in samples (nights here); <= 1 (or a series shorter than 3)
/// returns the input unchanged. Near the edges the window shrinks to the
/// available neighbours so endpoints aren't dragged toward the interior.
fn smooth_moving_average(ys: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || ys.len() < 3 {
        return ys.to_vec();
    }
    let half = window / 2;
    let n = ys.len();
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let slice = &ys[lo..hi];
            slice.iter().sum::<f64>() / slice.len() as f64
        })
        .collect()
}

/// Natural cubic spline (boundary: y''=0 at the ends). Built once, evaluated
/// in O(log n) per query. Knots must be strictly increasing.
struct NaturalCubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    m: Vec<f64>, // second derivatives at the knots
}

impl NaturalCubicSpline {
    fn build(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        if n <= 1 {
            return Self { xs: xs.to_vec(), ys: ys.to_vec(), m: vec![0.0; n] };
        }
        // Tridiagonal system from CR (natural BC).
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        b[0] = 1.0; b[n - 1] = 1.0;
        for i in 1..n - 1 {
            let hi_1 = xs[i] - xs[i - 1];
            let hi = xs[i + 1] - xs[i];
            a[i] = hi_1;
            b[i] = 2.0 * (hi_1 + hi);
            c[i] = hi;
            d[i] = 6.0 * ((ys[i + 1] - ys[i]) / hi - (ys[i] - ys[i - 1]) / hi_1);
        }
        // Thomas algorithm.
        for i in 1..n {
            let w = if b[i - 1] != 0.0 { a[i] / b[i - 1] } else { 0.0 };
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
        let mut m = vec![0.0; n];
        m[n - 1] = if b[n - 1] != 0.0 { d[n - 1] / b[n - 1] } else { 0.0 };
        for i in (0..n - 1).rev() {
            m[i] = if b[i] != 0.0 { (d[i] - c[i] * m[i + 1]) / b[i] } else { 0.0 };
        }
        Self { xs: xs.to_vec(), ys: ys.to_vec(), m }
    }

    fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 { return 0.0; }
        if n == 1 { return self.ys[0]; }
        if x <= self.xs[0] {
            // Linear extrapolation tangent to the spline at the left edge.
            let dx = self.xs[1] - self.xs[0];
            let slope = (self.ys[1] - self.ys[0]) / dx - dx * self.m[1] / 6.0;
            return self.ys[0] + slope * (x - self.xs[0]);
        }
        if x >= self.xs[n - 1] {
            let dx = self.xs[n - 1] - self.xs[n - 2];
            let slope = (self.ys[n - 1] - self.ys[n - 2]) / dx + dx * self.m[n - 2] / 6.0;
            return self.ys[n - 1] + slope * (x - self.xs[n - 1]);
        }
        // Binary search for the interval.
        let mut lo = 0usize; let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] > x { hi = mid; } else { lo = mid; }
        }
        let h = self.xs[hi] - self.xs[lo];
        let a = (self.xs[hi] - x) / h;
        let b = (x - self.xs[lo]) / h;
        a * self.ys[lo] + b * self.ys[hi]
            + ((a.powi(3) - a) * self.m[lo] + (b.powi(3) - b) * self.m[hi]) * h.powi(2) / 6.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn outcome(tm: f64, src: TmSource) -> Option<NightOutcome> {
        Some(NightOutcome { tm, source: src })
    }

    #[test]
    fn interpolate_fills_gap_linearly() {
        let input = vec![
            outcome(10.0, TmSource::Valid),
            None,
            None,
            None,
            outcome(14.0, TmSource::Valid),
        ];
        let out = interpolate_missing(crate::core::types::NightInterpolationMethod::Linear, &input, 1);
        assert_eq!(out[0].as_ref().unwrap().tm, 10.0);
        assert!((out[1].as_ref().unwrap().tm - 11.0).abs() < 1e-9);
        assert!((out[2].as_ref().unwrap().tm - 12.0).abs() < 1e-9);
        assert!((out[3].as_ref().unwrap().tm - 13.0).abs() < 1e-9);
        assert_eq!(out[4].as_ref().unwrap().tm, 14.0);
        assert_eq!(out[1].as_ref().unwrap().source, TmSource::Interpolated);
    }

    #[test]
    fn interpolate_holds_at_edges() {
        let input = vec![
            None,
            None,
            outcome(5.0, TmSource::Valid),
            None,
            None,
        ];
        let out = interpolate_missing(crate::core::types::NightInterpolationMethod::Linear, &input, 1);
        assert_eq!(out[0].as_ref().unwrap().tm, 5.0);
        assert_eq!(out[1].as_ref().unwrap().tm, 5.0);
        assert_eq!(out[4].as_ref().unwrap().tm, 5.0);
        assert_eq!(out[0].as_ref().unwrap().source, TmSource::Interpolated);
    }

    #[test]
    fn interpolate_all_invalid_returns_all_none() {
        let input: Vec<Option<NightOutcome>> = vec![None, None, None];
        let out = interpolate_missing(crate::core::types::NightInterpolationMethod::Linear, &input, 1);
        assert!(out.iter().all(|o| o.is_none()));
    }

    #[test]
    fn pick_best_window_rejects_when_cv_too_high() {
        let values = Col {
            data: vec![
                Some(10.0),
                Some(20.0),
                Some(10.0),
                Some(20.0),
                Some(10.0),
            ],
        };
        let is_night = vec![true; 5];
        let is_low_vpd = vec![true; 5];
        let indices: Vec<usize> = (0..5).collect();
        let out = pick_best_window(&indices, &values, &is_night, &is_low_vpd, 4, 4, 0.01);
        assert!(out.is_none()); // CV is ~33% ≫ 1%
    }

    #[test]
    fn pick_best_window_accepts_stable_series() {
        let values = Col {
            data: vec![
                Some(10.0),
                Some(10.01),
                Some(9.99),
                Some(10.0),
                Some(10.01),
            ],
        };
        let is_night = vec![true; 5];
        let is_low_vpd = vec![true; 5];
        let indices: Vec<usize> = (0..5).collect();
        let out = pick_best_window(&indices, &values, &is_night, &is_low_vpd, 4, 4, 0.01)
            .expect("should accept stable series");
        assert!((out.tm - 10.0).abs() < 0.1);
        assert_eq!(out.source, TmSource::Valid);
    }

    #[test]
    fn group_by_night_folds_night_spanning_midnight() {
        let cfg = VpdParConfig {
            night_start_hour: 20,
            night_end_hour: 6,
            ..Default::default()
        };
        let dt = |y, m, d, h| NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, 0, 0).unwrap();
        let datetimes = vec![
            dt(2024, 1, 1, 20),
            dt(2024, 1, 1, 23),
            dt(2024, 1, 2, 2),
            dt(2024, 1, 2, 5),
            dt(2024, 1, 2, 12), // daytime — should be skipped
        ];
        let flags = vec![true; 5];
        let groups = group_by_night(&datetimes, &flags, &cfg);
        let morning = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        assert_eq!(groups.get(&morning).unwrap().len(), 4);
        assert_eq!(groups.len(), 1);
    }
}
