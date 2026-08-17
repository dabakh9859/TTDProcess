use anyhow::{anyhow, Result};
use polars::prelude::*;
use std::f64::consts::PI;

use super::types::TtdPlusParams;

/// Run the full TTD+ pipeline on the input DataFrame.
/// Returns (fourier_df, references_df, sap_flow_df)
pub fn run_ttdplus_pipeline(
    df: &DataFrame,
    params: &TtdPlusParams,
) -> Result<(DataFrame, DataFrame, DataFrame)> {
    let n = params.num_samples;
    if n == 0 || n % 2 != 0 {
        return Err(anyhow!("num_samples (N) must be a positive even number, got {}", n));
    }

    // Get timestamp column (first column assumed datetime)
    let ts_col = df.get_columns().iter().find(|c| {
        matches!(c.dtype(), DataType::Datetime(_, _) | DataType::Date)
    });
    let ts_name = ts_col.map(|c| c.name().to_string())
        .unwrap_or_else(|| df.get_column_names()[0].to_string());

    // Find numeric (temperature) columns
    let temp_cols: Vec<String> = df.get_column_names().iter()
        .filter(|name| {
            let s = name.to_string();
            s != ts_name && df.column(&s).map(|c| c.dtype().is_float()).unwrap_or(false)
        })
        .map(|s| s.to_string())
        .collect();

    if temp_cols.is_empty() {
        return Err(anyhow!("No numeric temperature columns found"));
    }

    let height = df.height();
    let num_cycles = height / n;
    if num_cycles == 0 {
        return Err(anyhow!("Not enough data for even one cycle (need {} rows, have {})", n, height));
    }

    // ── Step 1-5: Compute Fourier coefficients, amplitude, phase per cycle per column ──
    let mut fourier_timestamps: Vec<String> = Vec::with_capacity(num_cycles);
    let mut col_alpha: Vec<Vec<f64>> = vec![Vec::with_capacity(num_cycles); temp_cols.len()];
    let mut col_phi: Vec<Vec<f64>> = vec![Vec::with_capacity(num_cycles); temp_cols.len()];

    // Get timestamp series for cycle midpoint timestamps
    let ts_series = df.column(&ts_name).unwrap();

    for cycle_idx in 0..num_cycles {
        let start = cycle_idx * n;

        // Cycle midpoint timestamp
        let mid = start + n / 2;
        let ts_val = format!("{}", ts_series.get(mid).unwrap());
        fourier_timestamps.push(ts_val);

        for (col_i, col_name) in temp_cols.iter().enumerate() {
            let series = df.column(col_name)?;

            // Collect N values for this cycle
            let mut valid = true;
            let mut temps: Vec<f64> = Vec::with_capacity(n);
            for j in 0..n {
                let idx = start + j;
                match series.get(idx)? {
                    AnyValue::Float64(v) if v.is_finite() => temps.push(v),
                    AnyValue::Float32(v) if v.is_finite() => temps.push(v as f64),
                    AnyValue::Null => { valid = false; break; }
                    _ => { valid = false; break; }
                }
            }

            if !valid || temps.len() != n {
                // Invalid cycle: push NaN
                col_alpha[col_i].push(f64::NAN);
                col_phi[col_i].push(f64::NAN);
                continue;
            }

            // Fourier coefficients (k=1, first harmonic)
            let mut a = 0.0_f64;
            let mut b = 0.0_f64;
            let n_f = n as f64;
            for j in 0..n {
                let angle = 2.0 * PI * (j as f64) / n_f;
                a += temps[j] * angle.cos();
                b += temps[j] * angle.sin();
            }
            a /= n_f;
            b /= n_f;

            // Amplitude and phase
            let alpha = (a * a + b * b).sqrt();
            let phi = b.atan2(a);

            col_alpha[col_i].push(alpha);
            col_phi[col_i].push(phi);
        }
    }

    // Build fourier DataFrame
    let mut fourier_columns: Vec<Series> = Vec::new();
    fourier_columns.push(
        Series::new("timestamp".into(), &fourier_timestamps)
    );
    for (i, col_name) in temp_cols.iter().enumerate() {
        fourier_columns.push(Series::new(format!("alpha_{}", col_name).as_str().into(), &col_alpha[i]));
        fourier_columns.push(Series::new(format!("phi_{}", col_name).as_str().into(), &col_phi[i]));
    }
    let fourier_df = DataFrame::new(fourier_columns)?;

    // ── Step 6: Daily reference values (α₀, φ₀) ──
    // Parse timestamps to find day boundaries at ref_hour
    let refs_df = compute_daily_references(
        &fourier_timestamps,
        &col_alpha,
        &col_phi,
        &temp_cols,
        params,
        num_cycles,
    )?;

    // ── Step 7-8: Compute sap flux density J per cycle ──
    let sap_flow_df = compute_sap_flux(
        &fourier_timestamps,
        &col_alpha,
        &col_phi,
        &temp_cols,
        params,
        num_cycles,
    )?;

    Ok((fourier_df, refs_df, sap_flow_df))
}

/// Step 6: Compute daily α₀ and φ₀ (top decile averages), then interpolate.
fn compute_daily_references(
    timestamps: &[String],
    col_alpha: &[Vec<f64>],
    col_phi: &[Vec<f64>],
    temp_cols: &[String],
    params: &TtdPlusParams,
    num_cycles: usize,
) -> Result<DataFrame> {
    // Group cycles into 24h windows ending at ref_hour each day
    // For simplicity, we compute one reference set per day using all cycles in 24h
    // Then interpolate linearly between daily values for each cycle

    let num_cols = temp_cols.len();

    // Per-column: compute daily α₀, φ₀ then interpolate to per-cycle
    let mut ref_alpha0: Vec<Vec<f64>> = vec![vec![f64::NAN; num_cycles]; num_cols];
    let mut ref_phi0: Vec<Vec<f64>> = vec![vec![f64::NAN; num_cycles]; num_cols];

    // Determine day boundaries: every `cycles_per_day` cycles
    let period = params.period;
    let cycles_per_day = (86400.0 / period).round() as usize;
    if cycles_per_day == 0 {
        return Err(anyhow!("Period too large for daily reference calculation"));
    }

    for col_i in 0..num_cols {
        let alphas = &col_alpha[col_i];
        let phis = &col_phi[col_i];

        // Collect daily reference values
        let mut daily_alpha0: Vec<f64> = Vec::new();
        let mut daily_phi0: Vec<f64> = Vec::new();
        let mut daily_center_idx: Vec<usize> = Vec::new();

        let mut day_start = 0;
        while day_start < num_cycles {
            let day_end = (day_start + cycles_per_day).min(num_cycles);

            // Collect valid (α, φ) pairs for this day
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for i in day_start..day_end {
                if alphas[i].is_finite() && phis[i].is_finite() {
                    pairs.push((alphas[i], phis[i]));
                }
            }

            if pairs.is_empty() {
                daily_alpha0.push(f64::NAN);
                daily_phi0.push(f64::NAN);
            } else {
                // Sort by φ ascending
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Top decile (highest 10% of φ values = last 10%)
                let decile_count = (pairs.len() as f64 * 0.1).ceil() as usize;
                let decile_start = pairs.len().saturating_sub(decile_count);
                let top = &pairs[decile_start..];

                let a0 = top.iter().map(|(a, _)| a).sum::<f64>() / top.len() as f64;
                let p0 = top.iter().map(|(_, p)| p).sum::<f64>() / top.len() as f64;

                daily_alpha0.push(a0);
                daily_phi0.push(p0);
            }

            daily_center_idx.push((day_start + day_end) / 2);
            day_start = day_end;
        }

        // Interpolate daily values to per-cycle
        interpolate_daily_to_cycles(
            &daily_alpha0, &daily_center_idx, num_cycles, &mut ref_alpha0[col_i],
        );
        interpolate_daily_to_cycles(
            &daily_phi0, &daily_center_idx, num_cycles, &mut ref_phi0[col_i],
        );
    }

    // Build references DataFrame
    let mut ref_columns: Vec<Series> = Vec::new();
    ref_columns.push(Series::new("timestamp".into(), timestamps));
    for (i, col_name) in temp_cols.iter().enumerate() {
        ref_columns.push(Series::new(format!("alpha0_{}", col_name).as_str().into(), &ref_alpha0[i]));
        ref_columns.push(Series::new(format!("phi0_{}", col_name).as_str().into(), &ref_phi0[i]));
    }

    Ok(DataFrame::new(ref_columns)?)
}

/// Linear interpolation from daily values to per-cycle values.
fn interpolate_daily_to_cycles(
    daily_vals: &[f64],
    daily_centers: &[usize],
    num_cycles: usize,
    output: &mut [f64],
) {
    if daily_vals.is_empty() {
        return;
    }
    if daily_vals.len() == 1 {
        // Only one day: use constant
        let v = daily_vals[0];
        for o in output.iter_mut() {
            *o = v;
        }
        return;
    }

    for cycle_i in 0..num_cycles {
        // Find surrounding daily indices
        let mut left = 0;
        let mut right = daily_vals.len() - 1;
        for (di, &center) in daily_centers.iter().enumerate() {
            if center <= cycle_i {
                left = di;
            }
            if center >= cycle_i && di < right {
                right = di;
                break;
            }
        }

        if left == right || daily_centers[left] == daily_centers[right] {
            output[cycle_i] = daily_vals[left];
        } else {
            let t = (cycle_i as f64 - daily_centers[left] as f64)
                / (daily_centers[right] as f64 - daily_centers[left] as f64);
            let v_left = daily_vals[left];
            let v_right = daily_vals[right];
            if v_left.is_finite() && v_right.is_finite() {
                output[cycle_i] = v_left + t * (v_right - v_left);
            } else if v_left.is_finite() {
                output[cycle_i] = v_left;
            } else {
                output[cycle_i] = v_right;
            }
        }
    }
}

/// Steps 7-8: Compute sap flux density J for each cycle.
fn compute_sap_flux(
    timestamps: &[String],
    col_alpha: &[Vec<f64>],
    col_phi: &[Vec<f64>],
    temp_cols: &[String],
    params: &TtdPlusParams,
    num_cycles: usize,
) -> Result<DataFrame> {
    let period = params.period;
    let x = params.distance;
    let num_cols = temp_cols.len();

    // First, recompute references (same logic)
    let cycles_per_day = (86400.0 / period).round() as usize;
    let mut ref_a0 = vec![vec![f64::NAN; num_cycles]; num_cols];
    let mut ref_p0 = vec![vec![f64::NAN; num_cycles]; num_cols];

    for col_i in 0..num_cols {
        let alphas = &col_alpha[col_i];
        let phis = &col_phi[col_i];
        let mut daily_a0: Vec<f64> = Vec::new();
        let mut daily_p0: Vec<f64> = Vec::new();
        let mut daily_center: Vec<usize> = Vec::new();

        let mut ds = 0;
        while ds < num_cycles {
            let de = (ds + cycles_per_day).min(num_cycles);
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for i in ds..de {
                if alphas[i].is_finite() && phis[i].is_finite() {
                    pairs.push((alphas[i], phis[i]));
                }
            }
            if pairs.is_empty() {
                daily_a0.push(f64::NAN);
                daily_p0.push(f64::NAN);
            } else {
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let dc = (pairs.len() as f64 * 0.1).ceil() as usize;
                let start = pairs.len().saturating_sub(dc);
                let top = &pairs[start..];
                daily_a0.push(top.iter().map(|(a, _)| a).sum::<f64>() / top.len() as f64);
                daily_p0.push(top.iter().map(|(_, p)| p).sum::<f64>() / top.len() as f64);
            }
            daily_center.push((ds + de) / 2);
            ds = de;
        }
        interpolate_daily_to_cycles(&daily_a0, &daily_center, num_cycles, &mut ref_a0[col_i]);
        interpolate_daily_to_cycles(&daily_p0, &daily_center, num_cycles, &mut ref_p0[col_i]);
    }

    // Compute J for each cycle and column
    // J = (2π / T) × (x / φ) × [(α - C)² - φ²] / [(α - C)² + φ²]
    // C = α₀ + φ₀ (using C⁺)
    let two_pi_over_t = 2.0 * PI / period;

    let mut sap_columns: Vec<Series> = Vec::new();
    sap_columns.push(Series::new("timestamp".into(), timestamps));

    for (col_i, col_name) in temp_cols.iter().enumerate() {
        let mut j_values: Vec<f64> = Vec::with_capacity(num_cycles);

        for i in 0..num_cycles {
            let alpha = col_alpha[col_i][i];
            let phi = col_phi[col_i][i];
            let a0 = ref_a0[col_i][i];
            let p0 = ref_p0[col_i][i];

            if !alpha.is_finite() || !phi.is_finite() || !a0.is_finite() || !p0.is_finite() || phi.abs() < 1e-12 {
                j_values.push(f64::NAN);
                continue;
            }

            let c = a0 + p0; // C⁺
            let alpha_minus_c = alpha - c;
            let num = alpha_minus_c * alpha_minus_c - phi * phi;
            let den = alpha_minus_c * alpha_minus_c + phi * phi;

            if den.abs() < 1e-15 {
                j_values.push(f64::NAN);
                continue;
            }

            let j = two_pi_over_t * (x / phi) * (num / den);
            j_values.push(j);
        }

        sap_columns.push(Series::new(format!("J_{}", col_name).as_str().into(), &j_values));
    }

    Ok(DataFrame::new(sap_columns)?)
}
