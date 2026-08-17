use polars::prelude::*;
use anyhow::Result;

use crate::core::types::TimestampValidation;

/// Drop rows whose TIMESTAMP is null/blank.
///
/// Some Excel exports (and the loader) carry entirely-empty rows with no
/// timestamp. `ts_col_to_datetimes` silently drops them when building the time
/// index, which then misaligns every pipeline step that maps a *compacted*
/// datetime index back onto the *full* dataframe (the T600 :00/:30 filter, the
/// Tm nightly grouping, …) — producing zeroed or garbage outputs. Removing them
/// up front keeps `row index == time index` for the whole pipeline.
///
/// Returns the filtered frame and the number of rows removed. A no-op (Ok with
/// 0 removed) when there is no TIMESTAMP column.
pub fn drop_null_timestamp_rows(data: &DataFrame) -> Result<(DataFrame, usize)> {
    if !data.get_column_names().contains(&"TIMESTAMP") {
        return Ok((data.clone(), 0));
    }
    let mask = data.column("TIMESTAMP")?.is_not_null();
    let filtered = data.filter(&mask)?;
    let removed = data.height().saturating_sub(filtered.height());
    Ok((filtered, removed))
}

/// Check whether the timestamp column follows the expected cyclic pattern.
///
/// `pattern` is a slice of expected intervals in seconds (e.g., [30, 30, 60, 180, ...]).
#[allow(dead_code)]
pub fn check_timestamp_pattern(
    data: &DataFrame,
    pattern: &[i64],
) -> Result<TimestampValidation> {
    if !data.get_column_names().contains(&"TIMESTAMP") {
        anyhow::bail!("TIMESTAMP column is required");
    }

    if pattern.is_empty() {
        anyhow::bail!("Pattern must not be empty");
    }
    if pattern.iter().any(|&p| p <= 0) {
        anyhow::bail!("Pattern must contain only strictly positive values (seconds)");
    }

    let timestamp_col = data.column("TIMESTAMP")?;
    let ts_vals: Vec<i64> = match timestamp_col.dtype() {
        DataType::Datetime(_, _) => timestamp_col
            .datetime()?
            .into_iter()
            .flatten()
            .collect(),
        DataType::Int64 => timestamp_col
            .i64()?
            .into_iter()
            .flatten()
            .collect(),
        _ => anyhow::bail!(
            "Unsupported TIMESTAMP type: {:?}",
            timestamp_col.dtype()
        ),
    };

    let total_rows = ts_vals.len();
    let mut errors = Vec::new();
    let mut anomalies_count = 0;

    if total_rows < 2 {
        return Ok(TimestampValidation {
            is_valid: true,
            errors,
            anomalies_count: 0,
            total_rows,
        });
    }

    let pattern_len = pattern.len();
    for i in 1..total_rows {
        let diff_us = ts_vals[i] - ts_vals[i - 1];
        let diff_secs = diff_us / 1_000_000; // microseconds to seconds
        let expected = pattern[(i - 1) % pattern_len];

        if diff_secs != expected {
            anomalies_count += 1;
            if errors.len() < 20 {
                errors.push(format!(
                    "Row {}: expected {}s, got {}s",
                    i, expected, diff_secs
                ));
            }
        }
    }

    Ok(TimestampValidation {
        is_valid: anomalies_count == 0,
        errors,
        anomalies_count,
        total_rows,
    })
}

/// Common datetime formats we encounter in field-data exports. Tried in
/// order until one matches. Covers ISO-8601, French / US locales, with or
/// without seconds.
const DATETIME_FORMATS: &[&str] = &[
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y",
];

pub fn parse_string_to_datetime(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let s = s.trim();
    if s.is_empty() { return None; }
    for fmt in DATETIME_FORMATS {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc());
        }
        // Date-only format: combine with midnight.
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, fmt) {
            if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                return Some(dt.and_utc());
            }
        }
    }
    None
}

/// Convert a Polars TIMESTAMP column into a vector of chrono `DateTime<Utc>`.
/// Accepts native Datetime, Int64 microseconds, and String columns parsed
/// with the common formats above. Used wherever we need to inspect hours,
/// dates, or build date-indexed HashMaps.
pub fn ts_col_to_datetimes(col: &Series) -> Result<Vec<chrono::DateTime<chrono::Utc>>> {
    // String case: parse each cell.
    if matches!(col.dtype(), DataType::String) {
        let ca = col.str()?;
        let mut out: Vec<chrono::DateTime<chrono::Utc>> = Vec::with_capacity(ca.len());
        let mut sample: Option<String> = None;
        for opt in ca.into_iter() {
            if let Some(s) = opt {
                if let Some(dt) = parse_string_to_datetime(s) {
                    out.push(dt);
                } else if sample.is_none() {
                    sample = Some(s.to_string());
                }
            }
        }
        if out.is_empty() {
            anyhow::bail!(
                "TIMESTAMP column couldn't be parsed as date — got '{}' (try ISO format YYYY-MM-DD HH:MM:SS)",
                sample.unwrap_or_default()
            );
        }
        return Ok(out);
    }

    let dt_series = match col.dtype() {
        DataType::Datetime(_, _) => col.clone(),
        DataType::Int64 => col.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
        _ => anyhow::bail!("TIMESTAMP must be Datetime, Int64 (microseconds), or a String like 'YYYY-MM-DD HH:MM:SS'"),
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

/// Drop rows whose timestamp is exactly 10 seconds after the previous row.
///
/// The 2019 datalogger configuration recorded one extra sample 10 seconds
/// after each heating-cycle start (e.g. `HH:20:10` right after `HH:20:00`).
/// The 2020+ configuration dropped that read. Since the heating-cycle pattern
/// (see `Config::pattern`, default `[30, 30, 60, 180, 300, ...]`) has a
/// minimum legitimate delta of 30 seconds, any consecutive pair separated by
/// exactly 10 seconds is a 2019-config artefact.
///
/// The baseline step (`calculate_baseline`) advances its pattern-position
/// counter by one row per loop iteration, so an extra row knocks every
/// subsequent row out of phase — corrupting baselines for hours. Pre-filtering
/// these extras here keeps the row count aligned with the pattern for files
/// that mix both configurations (2019 + early 2020).
///
/// Since ~June 2024, the datalogger also records duplicate rows (0 s delta)
/// and occasionally backwards timestamps (negative delta). These pollute
/// baseline anchoring and produce a second T600 value per cycle, creating an
/// oscillating ("cisaillement") pattern on multi-year files.  Dropping
/// `delta <= 0` here removes them before any calculation sees the data.
///
/// Returns `(filtered_df, n_dropped)`.
pub fn drop_intra_cycle_extras(data: &DataFrame) -> Result<(DataFrame, usize)> {
    if data.height() < 2 {
        return Ok((data.clone(), 0));
    }
    let ts_col = data.column("TIMESTAMP")?;
    let datetimes = ts_col_to_datetimes(ts_col)?;
    if datetimes.len() != data.height() {
        return Ok((data.clone(), 0));
    }
    let mut keep = vec![true; data.height()];
    let mut last_kept = 0usize;
    for i in 1..data.height() {
        let delta = (datetimes[i] - datetimes[last_kept]).num_seconds();
        if delta == 10 || delta <= 0 {
            keep[i] = false;
        } else {
            last_kept = i;
        }
    }
    let n_dropped = keep.iter().filter(|&&k| !k).count();
    if n_dropped == 0 {
        return Ok((data.clone(), 0));
    }
    let mask = BooleanChunked::from_slice("mask".into(), &keep);
    let filtered = data.filter(&mask)?;
    Ok((filtered, n_dropped))
}

/// Infer the heating-cycle pattern (in seconds) from the actual consecutive
/// timestamp deltas of the loaded data.
///
/// **Why this exists.** The baseline step (`calculate_baseline`) walks the
/// rows of the DataFrame and reads the expected inter-row delta from a
/// hard-coded pattern at position `(i - 1) % len`. If the file's real cadence
/// has a different cycle length than the configured pattern, every row past
/// the first mismatch gets the wrong delta — baselines become arbitrary,
/// `ΔT = SF − baseline` flips sign on alternating samples, and the half-hour
/// T600 series looks like a zigzag instead of a smooth diurnal curve.
///
/// Common cadences we see in field files:
///   - 11 deltas / cycle: `[30,30,60,180,300, 30,30,60,180,300, 600]` — two
///     heat starts per 30 min (default).
///   - 7 deltas / cycle: `[30,30,60,180,300, 600, 600]` — one heat start per
///     30 min (e.g. heats at HH:20 and HH:50 only).
///
/// **Algorithm.** For each candidate cycle length L in a curated list, slice
/// the deltas into stripes of length L, take the median at each position,
/// and score how many deltas the resulting candidate matches at their own
/// position. Keep the shortest L whose score ≥ 95 %. Returns `None` when no
/// candidate is convincing — the caller should keep the configured pattern.
///
/// `drop_intra_cycle_extras` MUST run before this so the +10 s legacy samples
/// don't pollute the deltas.
pub fn infer_heating_pattern(data: &DataFrame) -> Result<Option<Vec<i64>>> {
    let ts = ts_col_to_datetimes(data.column("TIMESTAMP")?)?;
    if ts.len() < 30 {
        return Ok(None);
    }
    let deltas: Vec<i64> = (1..ts.len())
        .map(|i| (ts[i] - ts[i - 1]).num_seconds())
        .collect();

    let candidates = [5usize, 7, 9, 11, 13, 15, 21];
    let mut best: Option<(Vec<i64>, f64)> = None;
    for &l in &candidates {
        if l * 3 > deltas.len() {
            continue;
        }
        let mut cand: Vec<i64> = Vec::with_capacity(l);
        for p in 0..l {
            let mut col: Vec<i64> = deltas.iter().skip(p).step_by(l).copied().collect();
            col.sort_unstable();
            cand.push(col[col.len() / 2]);
        }
        let matches = deltas.iter().enumerate()
            .filter(|(i, &d)| d == cand[i % l])
            .count();
        let score = matches as f64 / deltas.len() as f64;
        let better = match &best {
            None => true,
            // Prefer the shortest L; only switch to a longer one when its
            // score is meaningfully (>1 pp) higher.
            Some((_, s)) => score > *s + 0.01,
        };
        if better {
            best = Some((cand, score));
        }
    }
    Ok(best.filter(|(_, s)| *s >= 0.95).map(|(p, _)| p))
}

/// Compute the difference in seconds between consecutive timestamps.
#[allow(dead_code)]
pub fn compute_timestamp_diffs(data: &DataFrame) -> Result<Vec<i64>> {
    let timestamp_col = data.column("TIMESTAMP")?;
    let ts_vals: Vec<i64> = match timestamp_col.dtype() {
        DataType::Datetime(_, _) => timestamp_col
            .datetime()?
            .into_iter()
            .flatten()
            .collect(),
        DataType::Int64 => timestamp_col
            .i64()?
            .into_iter()
            .flatten()
            .collect(),
        _ => anyhow::bail!("Unsupported TIMESTAMP type"),
    };

    let diffs: Vec<i64> = ts_vals
        .windows(2)
        .map(|w| (w[1] - w[0]) / 1_000_000)
        .collect();

    Ok(diffs)
}
