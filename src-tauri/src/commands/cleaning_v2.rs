//! Tauri commands for the two-path data cleaning workflow (Voie A / Voie B).
//!
//! These commands expose the new `CleaningMethod` + `MLCleaner` pipeline to
//! the frontend. They coexist with the legacy `detect_outliers` / `fill_gaps`
//! commands for now — the UI chooses which API to call.
//!
//! Workflow from the frontend:
//!   1. `cleaning_list_columns(dataset)` → get available columns
//!   2. Voie A: `cleaning_run_classical(dataset, method)`    (one shot)
//!   2. Voie B: `cleaning_train_ml(method)` → metrics
//!              `cleaning_apply_ml(threshold_factor)` → stats
//!   3. `cleaning_reset_ml()` clears a trained model

use polars::prelude::{DataFrame, NamedFrom, Series};
use tauri::State;

use crate::core::data_cleaning;
use crate::core::data_cleaning::{GapFillMethod, GapFillReport};
use crate::core::ml_cleaning::{MLCleanMode, MLCleaner, MLDetectionPerColumn};
use crate::core::types::{CleaningMethod, CleaningReport, CleaningTrainingMetrics, LogLevel};
use crate::state::AppState;
use crate::utils::logger;

/// Write a cleaned DataFrame back into the slot it came from. For raw/cleaned
/// the result lands in `cleaned_data` (so the TTD pipeline picks it up).
/// For derived datasets we mutate the corresponding `results.*` slot so the
/// user can inspect the corrected derived series without breaking the
/// pipeline (which still runs from raw → recomputes everything).
fn write_cleaned_to_slot(
    app: &mut crate::state::AppData,
    key: &str,
    df: DataFrame,
) -> Result<(), String> {
    match key {
        "raw" | "cleaned" => { app.cleaned_data = Some(df); }
        "tslope"   => { app.results.tslope = Some(df); }
        "baseline" => { app.results.baseline = Some(df); }
        "delta_t"  => { app.results.delta_t = Some(df); }
        "t600"     => { app.results.t600 = Some(df); }
        "tm"       => { app.results.tm = Some(df); }
        "stm"      => { app.results.stm = Some(df); }
        "tmi"      => { app.results.tmi = Some(df); }
        "k"        => { app.results.k = Some(df); }
        "sap_flow" => { app.results.sap_flow = Some(df); }
        other => return Err(format!("dataset inconnu '{}'", other)),
    }
    Ok(())
}

/// The frame a cleaning run should START from.
///
/// For raw/cleaned that is `cleaned_data` whenever it exists. `dataset_ref`
/// resolves "raw" to the pristine import, but `write_cleaned_to_slot` writes to
/// `cleaned_data` — so reading through `dataset_ref` makes every pass restart
/// from the untouched file and silently discard the previous one. Painting a
/// few zones, checking, then painting more would keep only the last pass.
///
/// Derived slots are cleaned in place, so their own slot is already the
/// latest version and needs no special case.
fn working_df(app: &crate::state::AppData, key: &str) -> Result<DataFrame, String> {
    if matches!(key, "raw" | "cleaned") {
        if let Some(df) = app.cleaned_data.as_ref() {
            return Ok(df.clone());
        }
    }
    dataset_ref(app, key).map(|d| d.clone())
}

fn dataset_ref<'a>(
    app: &'a crate::state::AppData,
    key: &str,
) -> Result<&'a DataFrame, String> {
    let r = match key {
        "raw" => app.raw_data.as_ref(),
        "cleaned" => app.cleaned_data.as_ref(),
        // Calculated datasets from the TTD pipeline — available once the user
        // has run the pipeline at least once. These let ML cleaning target
        // intermediate signals (e.g. T600 before Tm) as scientifically
        // recommended by SapFlower.
        "tslope" => app.results.tslope.as_ref(),
        "baseline" => app.results.baseline.as_ref(),
        "delta_t" => app.results.delta_t.as_ref(),
        "t600" => app.results.t600.as_ref(),
        "tm" => app.results.tm.as_ref(),
        "stm" => app.results.stm.as_ref(),
        "tmi" => app.results.tmi.as_ref(),
        "k" => app.results.k.as_ref(),
        "sap_flow" => app.results.sap_flow.as_ref(),
        other => return Err(format!("dataset inconnu '{}'", other)),
    };
    r.ok_or_else(|| format!("dataset '{}' non chargé", key))
}

// =============================================================================
// List columns (used by the UI to populate predictor / target selectors)
// =============================================================================

#[tauri::command]
pub fn cleaning_list_columns(
    state: State<'_, AppState>,
    dataset: String,
) -> Result<serde_json::Value, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let df = dataset_ref(&app, &dataset)?;
    let cols: Vec<serde_json::Value> = df
        .get_column_names()
        .iter()
        .map(|n| {
            let name = n.to_string();
            let is_ts = name == "TIMESTAMP";
            let dtype = df
                .column(&name)
                .map(|c| format!("{:?}", c.dtype()))
                .unwrap_or_else(|_| "Unknown".to_string());
            let is_numeric = df
                .column(&name)
                .map(|c| c.dtype().is_numeric())
                .unwrap_or(false);
            serde_json::json!({
                "name": name,
                "dtype": dtype,
                "is_timestamp": is_ts,
                "is_numeric": is_numeric,
            })
        })
        .collect();

    let env_cols: Vec<String> = app
        .env_data
        .as_ref()
        .map(|df| df.get_column_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    Ok(serde_json::json!({
        "columns": cols,
        "env_columns": env_cols,
        "env_loaded": app.env_data.is_some(),
        "ml_cleaner_loaded": app.ml_cleaner.is_some(),
    }))
}

// =============================================================================
// Manual cleaning — user-driven NaN marking (Voie C)
// =============================================================================
// Used by:
//   - Voie C (purely manual workflow): the user brushes zones / clicks points
//     directly on the chart with no algorithm involved.
//   - Voies A/B as a supplement: after running the algo, the user can flag
//     additional outliers the algo missed; the frontend chains
//     classical/ml apply → mark_manual_nan in one click.

#[derive(Debug, serde::Deserialize)]
pub struct ManualColumnSelection {
    pub column: String,
    /// Inclusive [start, end] time ranges in UTC milliseconds. Every row
    /// whose TIMESTAMP falls inside any range gets NaN-ed for this column.
    #[serde(default)]
    pub time_ranges: Vec<(i64, i64)>,
    /// Individual point timestamps in UTC milliseconds — exact match
    /// against the row's TIMESTAMP. Useful for kicking out single bad
    /// samples without having to brush a tiny zone.
    #[serde(default)]
    pub point_timestamps: Vec<i64>,
    /// 2D selection boxes as (t_start_ms, t_end_ms, y_min, y_max). Every row
    /// whose TIMESTAMP is inside [t_start, t_end] AND whose value is inside
    /// [y_min, y_max] gets NaN-ed.
    ///
    /// This exists because the chart is STRIDE-DOWNSAMPLED (105k rows shown as
    /// ~7.5k points). The box brush used to resolve client-side and send the
    /// matched points via `point_timestamps` — which could only ever name rows
    /// the chart had actually rendered, so ~13 of every 14 rows inside the box
    /// were silently spared. Sending the box itself lets us resolve it here,
    /// against the FULL data.
    #[serde(default)]
    pub value_boxes: Vec<(i64, i64, f64, f64)>,
}

#[tauri::command]
pub async fn cleaning_mark_manual_nan(
    state: State<'_, AppState>,
    dataset: Option<String>,
    selections: Vec<ManualColumnSelection>,
) -> Result<serde_json::Value, String> {
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    if selections.is_empty() {
        return Err("Aucune sélection manuelle fournie".to_string());
    }

    // Pull the source DF + take a pre-cleaning snapshot if this slot has
    // never been cleaned before (so the Gap Filling tab can show before/
    // after). Same convention as the algo apply paths.
    let mut df = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        // Snapshot the PRISTINE slot the first time (before/after comparison),
        // but work from the latest cleaned version so passes accumulate.
        let pristine = dataset_ref(&app, &dataset_key)?.clone();
        app.cleaning_pre_snapshots
            .entry(dataset_key.clone())
            .or_insert(pristine);
        working_df(&app, &dataset_key)?
    };

    // Resolve TIMESTAMP → millis once, off-lock.
    let (df_out, per_column_stats): (DataFrame, Vec<(String, usize)>) =
        tokio::task::spawn_blocking(move || -> Result<(DataFrame, Vec<(String, usize)>), String> {
            use polars::prelude::*;
            let ts_col = df.column("TIMESTAMP")
                .map_err(|e| format!("Colonne TIMESTAMP requise pour le marquage manuel: {}", e))?;
            let datetimes = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
                .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
            let row_ms: Vec<i64> = datetimes
                .iter()
                .map(|d| d.timestamp_millis())
                .collect();

            let mut stats: Vec<(String, usize)> = Vec::with_capacity(selections.len());
            for sel in &selections {
                // Read the column FIRST: `value_boxes` needs the values to decide
                // which rows fall inside a box. Only Float64 / Float32 columns
                // make sense for NaN-ing; integer / string are rejected to keep
                // the schema clean.
                let col = df.column(&sel.column)
                    .map_err(|e| format!("Colonne '{}' introuvable: {}", sel.column, e))?;
                let is_f32 = match col.dtype() {
                    DataType::Float64 => false,
                    DataType::Float32 => true,
                    other => return Err(format!(
                        "Colonne '{}' de type {:?} — seuls les types numériques peuvent être marqués NaN",
                        sel.column, other,
                    )),
                };
                let vals: Vec<Option<f64>> = if is_f32 {
                    col.f32().map_err(|e| e.to_string())?
                        .into_iter().map(|o| o.map(|v| v as f64)).collect()
                } else {
                    col.f64().map_err(|e| e.to_string())?.into_iter().collect()
                };

                // Build a boolean mask: which rows belong to this selection.
                let mut hit = vec![false; row_ms.len()];
                for (i, &ts) in row_ms.iter().enumerate() {
                    if sel.time_ranges.iter().any(|&(lo, hi)| {
                        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        ts >= lo && ts <= hi
                    }) {
                        hit[i] = true;
                        continue;
                    }
                    // Point match — tolerate ±1ms float-rounding noise.
                    if sel.point_timestamps.iter().any(|&p| (p - ts).abs() <= 1) {
                        hit[i] = true;
                        continue;
                    }
                    // 2D box match — resolved here, against every row, so a
                    // downsampled chart can't spare the rows it didn't draw.
                    if let Some(v) = vals[i] {
                        if sel.value_boxes.iter().any(|&(t0, t1, y0, y1)| {
                            let (t0, t1) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                            let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
                            ts >= t0 && ts <= t1 && v >= y0 && v <= y1
                        }) {
                            hit[i] = true;
                        }
                    }
                }
                let n_hit = hit.iter().filter(|&&b| b).count();
                stats.push((sel.column.clone(), n_hit));
                if n_hit == 0 { continue; }

                let new_series = if is_f32 {
                    let v: Vec<Option<f32>> = vals.iter().zip(&hit)
                        .map(|(o, &h)| if h { None } else { o.map(|x| x as f32) })
                        .collect();
                    Series::new(sel.column.as_str().into(), v)
                } else {
                    let v: Vec<Option<f64>> = vals.iter().zip(&hit)
                        .map(|(o, &h)| if h { None } else { *o })
                        .collect();
                    Series::new(sel.column.as_str().into(), v)
                };
                df.replace(&sel.column, new_series)
                    .map_err(|e| format!("Échec remplacement '{}': {}", sel.column, e))?;
            }
            Ok((df, stats))
        })
        .await
        .map_err(|e| e.to_string())??;

    let target_cols: Vec<String> = per_column_stats.iter().map(|(c, _)| c.clone()).collect();
    let n_total_marked: usize = per_column_stats.iter().map(|(_, n)| n).sum();

    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        write_cleaned_to_slot(&mut app, &dataset_key, df_out)?;
        // Provenance: manual is a SUPPLEMENT when it refines a run on the same
        // slot — keep the algorithm's label in that case. But when the user
        // moves to a DIFFERENT dataset the old provenance becomes plain wrong:
        // it would keep pointing the Calculs tab at a stage the user has left,
        // so "Recalculer à partir de T600" would be offered after a raw
        // cleaning and quietly ignore it.
        let same_slot = app.cleaning_source_dataset.as_deref() == Some(dataset_key.as_str());
        if app.cleaning_path.is_none() || !same_slot {
            app.cleaning_path = Some("manual".to_string());
            app.cleaning_method_label = Some("Manual".to_string());
            app.cleaning_source_dataset = Some(dataset_key.clone());
        }
        // Union the touched columns with whatever was already there.
        for c in &target_cols {
            if !app.cleaning_target_columns.iter().any(|x| x == c) {
                app.cleaning_target_columns.push(c.clone());
            }
        }
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!(
                "Nettoyage manuel sur {} : {} cellules → NaN sur {} colonne(s)",
                dataset_key, n_total_marked, target_cols.len(),
            ),
        );
        crate::utils::session_persist::save(&app);
    }

    let per_col_json: Vec<serde_json::Value> = per_column_stats
        .into_iter()
        .map(|(c, n)| serde_json::json!({ "column": c, "n_marked": n }))
        .collect();

    Ok(serde_json::json!({
        "dataset": dataset_key,
        "n_marked": n_total_marked,
        "per_column": per_col_json,
    }))
}

// =============================================================================
// Time gaps — missing ROWS (as opposed to NaN cells)
// =============================================================================
// A logger outage leaves no row at all, so there is nothing for gap filling to
// fill: both the classical methods and SAITS only ever write into rows that
// already exist (see commands/ai.rs — the prediction map is walked against the
// existing TIMESTAMP index, and predictions with no matching row are dropped).
// `cleaning_reindex_time` materialises the missing instants as all-NaN rows so
// the normal fill step can then work on them.

/// Nominal step of a series = the most frequent gap between consecutive rows.
/// Field data jitters (25/35/38 min around a 30 min cycle), so the mode is far
/// more robust here than the mean or the min.
fn infer_step_seconds(ms: &[i64]) -> Option<i64> {
    if ms.len() < 3 { return None; }
    let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for w in ms.windows(2) {
        let d = (w[1] - w[0]) / 1000;
        if d > 0 { *counts.entry(d).or_insert(0) += 1; }
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(d, _)| d)
}

/// The timing rule a dataset's rows are supposed to follow.
///
///  - `Fixed` — one row every N seconds, measured from the previous row. Used
///    for T600 and downstream: one row per heating cycle, nominally 30 min
///    apart, but landing on the cycle's peak-ΔT instant rather than a round
///    clock time (see `create_t600`), so there is no absolute grid to snap to.
///
///  - `Clock` — the raw logger cycle. Its steps are deliberately uneven
///    (30, 30, 60, 180, 300, 600 …) BUT the cycle is anchored to wall-clock
///    half-hours: on this data 11 offsets — 0, 30, 60, 120, 300, 600, 1200,
///    1230, 1260, 1320, 1500 s past each half hour — account for 99.4% of
///    1.29 M rows. Anchoring on the clock rather than on the row index is what
///    makes this usable: an index-anchored pattern (`infer_heating_pattern`)
///    loses its phase at every outage, which is precisely where we need it —
///    it scored only 0.79 here, below its own 0.95 acceptance threshold.
enum Grid {
    Fixed(i64),
    Clock { period_s: i64, offsets_s: Vec<i64> },
}

impl Grid {
    /// Shortest legitimate step — the "don't land on an existing row" tolerance.
    fn min_step_ms(&self) -> i64 {
        match self {
            Grid::Fixed(s) => s * 1000,
            Grid::Clock { period_s, offsets_s } => {
                let mut min = *period_s;
                for w in offsets_s.windows(2) {
                    min = min.min(w[1] - w[0]);
                }
                if let (Some(f), Some(l)) = (offsets_s.first(), offsets_s.last()) {
                    min = min.min(period_s - l + f);
                }
                min.max(1) * 1000
            }
        }
    }

    /// Instants to splice between row `i` and row `i + 1`, in order. Empty when
    /// the two rows are already consecutive per the rule.
    fn instants_between(&self, ms: &[i64], i: usize) -> Vec<i64> {
        let (a, b) = (ms[i], ms[i + 1]);
        let mut out = Vec::new();
        match self {
            Grid::Fixed(step) => {
                let step_ms = step * 1000;
                if step_ms <= 0 || b - a <= (step_ms * 3) / 2 { return out; }
                let mut t = a + step_ms;
                while b - t > step_ms / 2 {
                    out.push(t);
                    t += step_ms;
                }
            }
            Grid::Clock { period_s, offsets_s } => {
                let period_ms = period_s * 1000;
                if period_ms <= 0 || offsets_s.is_empty() { return out; }
                let tol = (self.min_step_ms() / 2).max(1);
                if b - a <= tol { return out; }
                // Walk the clock cycles spanned by the hole and emit every
                // expected instant strictly inside it. Epoch is half-hour
                // aligned, so `ms % period` is the offset past the boundary.
                let mut cycle = a - a.rem_euclid(period_ms);
                while cycle < b {
                    for off in offsets_s {
                        let t = cycle + off * 1000;
                        if t - a > tol && b - t > tol {
                            out.push(t);
                        }
                    }
                    cycle += period_ms;
                    if out.len() > 5_000_000 { break; }
                }
                out.sort_unstable();
            }
        }
        out
    }
}

/// Dominant offsets of `ms` within `period_s`, plus the share of rows they
/// explain. An offset is "dominant" when it is at least half as frequent as the
/// most frequent one — on this data that cleanly separates the 11 real cycle
/// slots (each ~8.5-9.4% of rows) from the legacy +10 s drift (0.6%).
fn clock_offsets(ms: &[i64], period_s: i64) -> (Vec<i64>, f64) {
    let period_ms = period_s * 1000;
    if period_ms <= 0 || ms.is_empty() { return (Vec::new(), 0.0); }
    let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for &m in ms {
        *counts.entry(m.rem_euclid(period_ms) / 1000).or_insert(0) += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    if max == 0 { return (Vec::new(), 0.0); }
    let mut kept: Vec<(i64, usize)> = counts.into_iter().filter(|(_, n)| *n * 2 >= max).collect();
    kept.sort_by_key(|(o, _)| *o);
    let covered: usize = kept.iter().map(|(_, n)| *n).sum();
    (kept.into_iter().map(|(o, _)| o).collect(), covered as f64 / ms.len() as f64)
}

/// Pick the timing rule for a slot: the clock-anchored heating cycle for
/// raw-shaped data, a fixed step for the computed stages.
fn grid_for(_df: &DataFrame, key: &str, ms: &[i64], step_override: Option<i64>) -> Result<Grid, String> {
    if let Some(s) = step_override.filter(|s| *s > 0) {
        return Ok(Grid::Fixed(s));
    }
    if matches!(key, "raw" | "cleaned") {
        // Shortest period that explains ≥95% of rows with a stable slot set.
        for period_s in [1800i64, 3600, 900, 600, 300] {
            let (offsets, coverage) = clock_offsets(ms, period_s);
            if coverage >= 0.95 && !offsets.is_empty() {
                return Ok(Grid::Clock { period_s, offsets_s: offsets });
            }
        }
        return Err(
            "Motif d'horodatage non détectable sur ces données brutes : les lignes ne se répartissent \
             pas sur des positions stables dans le cycle. Indique un pas fixe si tu en connais un."
                .to_string(),
        );
    }
    infer_step_seconds(ms)
        .filter(|s| *s > 0)
        .map(Grid::Fixed)
        .ok_or_else(|| "Pas de temps indéterminable.".to_string())
}

/// Sorted row timestamps (millis) for a dataset slot.
fn row_millis(df: &DataFrame) -> Result<Vec<i64>, String> {
    let ts_col = df.column("TIMESTAMP")
        .map_err(|e| format!("Colonne TIMESTAMP requise: {}", e))?;
    let dts = crate::core::timestamp_utils::ts_col_to_datetimes(ts_col)
        .map_err(|e| format!("Impossible de parser TIMESTAMP: {}", e))?;
    Ok(dts.iter().map(|d| d.timestamp_millis()).collect())
}

/// Report the holes in the time index WITHOUT modifying anything, so the UI can
/// show what would be inserted before the user commits.
#[tauri::command]
pub async fn cleaning_time_gaps(
    state: State<'_, AppState>,
    dataset: Option<String>,
) -> Result<serde_json::Value, String> {
    let key = dataset.unwrap_or_else(|| "raw".to_string());
    let df = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        working_df(&app, &key)?
    };
    let ms = row_millis(&df)?;
    if ms.len() < 3 {
        return Ok(serde_json::json!({
            "dataset": key, "step_seconds": 0, "n_rows": ms.len(),
            "n_missing": 0, "n_gaps": 0, "gaps": []
        }));
    }
    let grid = match grid_for(&df, &key, &ms, None) {
        Ok(g) => g,
        // No detectable rule (too short / too irregular) — report "no gaps"
        // rather than failing the whole page.
        Err(_) => return Ok(serde_json::json!({
            "dataset": key, "step_seconds": 0, "n_rows": ms.len(),
            "n_missing": 0, "n_gaps": 0, "gaps": []
        })),
    };
    let (step, pattern_json) = match &grid {
        Grid::Fixed(s) => (*s, serde_json::Value::Null),
        // For raw, "step" is the cycle length; the intra-cycle steps are sent
        // so the UI can show which rule is being applied.
        Grid::Clock { period_s, offsets_s } => {
            let mut steps: Vec<i64> = offsets_s.windows(2).map(|w| w[1] - w[0]).collect();
            if let (Some(f), Some(l)) = (offsets_s.first(), offsets_s.last()) {
                steps.push(period_s - l + f);
            }
            (*period_s, serde_json::json!(steps))
        }
    };

    let mut n_missing = 0usize;
    let mut gaps: Vec<serde_json::Value> = Vec::new();
    for i in 0..ms.len() - 1 {
        let inserts = grid.instants_between(&ms, i);
        if inserts.is_empty() { continue; }
        n_missing += inserts.len();
        gaps.push(serde_json::json!({
            "start_ms": ms[i],
            "end_ms": ms[i + 1],
            "missing": inserts.len(),
            "hours": (ms[i + 1] - ms[i]) as f64 / 3_600_000.0,
        }));
    }
    // Count BEFORE truncation — the list below is only the head of it.
    let n_gaps_total = gaps.len();
    // Longest first — that's what the user needs to eyeball before filling.
    gaps.sort_by(|a, b| {
        let (x, y) = (a["missing"].as_u64().unwrap_or(0), b["missing"].as_u64().unwrap_or(0));
        y.cmp(&x)
    });
    gaps.truncate(50);

    Ok(serde_json::json!({
        "dataset": key,
        "step_seconds": step,
        "pattern": pattern_json,
        "n_rows": ms.len(),
        "n_missing": n_missing,
        "n_gaps": n_gaps_total,
        "gaps": gaps,
    }))
}

/// Insert the missing instants as all-NaN rows so gap filling has something to
/// fill. Existing rows are never moved or resampled — we only splice rows INTO
/// the holes, which keeps the series' natural jitter intact (snapping
/// everything onto a perfect grid would displace the ~2 000 rows that sit at
/// 25/35/38 min offsets).
#[tauri::command]
pub async fn cleaning_reindex_time(
    state: State<'_, AppState>,
    dataset: Option<String>,
    step_seconds: Option<i64>,
) -> Result<serde_json::Value, String> {
    let key = dataset.unwrap_or_else(|| "raw".to_string());

    let df = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        let pristine = dataset_ref(&app, &key)?.clone();
        app.cleaning_pre_snapshots.entry(key.clone()).or_insert(pristine);
        working_df(&app, &key)?
    };

    let key_for_task = key.clone();
    let (df_out, inserted, step, n_before, n_after) =
        tokio::task::spawn_blocking(move || -> Result<(DataFrame, usize, i64, usize, usize), String> {
            use polars::prelude::*;
            let key = key_for_task;
            let ms = row_millis(&df)?;
            let n_before = ms.len();
            if n_before < 3 { return Err("Série trop courte pour déduire un pas.".into()); }
            let grid = grid_for(&df, &key, &ms, step_seconds)?;
            let step = match &grid {
                Grid::Fixed(s) => *s,
                Grid::Clock { period_s, .. } => *period_s,
            };

            // Build the output row plan: Some(i) = existing row i, None = a row
            // to materialise at `ts`. Existing rows are never moved, so the
            // series' own jitter is preserved — we only splice into the holes.
            let mut idx: Vec<Option<IdxSize>> = Vec::with_capacity(n_before + 8192);
            let mut ts_out: Vec<i64> = Vec::with_capacity(n_before + 8192);
            for i in 0..n_before {
                idx.push(Some(i as IdxSize));
                ts_out.push(ms[i]);
                if i + 1 >= n_before { break; }
                for t in grid.instants_between(&ms, i) {
                    idx.push(None);
                    ts_out.push(t);
                }
            }
            let inserted = idx.iter().filter(|o| o.is_none()).count();
            if inserted == 0 {
                return Ok((df, 0, step, n_before, n_before));
            }

            // `take` with a null-carrying index gathers rows and yields NULL
            // everywhere the index is null — dtype-agnostic, so every column
            // (floats, ints, strings) gets a proper missing value.
            let idx_ca = IdxCa::from_iter(idx.iter().copied());
            let mut out = df.take(&idx_ca)
                .map_err(|e| format!("Insertion des lignes manquantes: {}", e))?;

            // The gathered TIMESTAMP is null on inserted rows — write the real
            // instants back, preserving the column's original dtype.
            let ts_dtype = df.column("TIMESTAMP").map_err(|e| e.to_string())?.dtype().clone();
            let ts_series = Series::new("TIMESTAMP".into(), &ts_out)
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .map_err(|e| format!("cast TIMESTAMP: {}", e))?
                .cast(&ts_dtype)
                .map_err(|e| format!("cast TIMESTAMP -> {:?}: {}", ts_dtype, e))?;
            out.replace("TIMESTAMP", ts_series)
                .map_err(|e| format!("replace TIMESTAMP: {}", e))?;

            let n_after = out.height();
            Ok((out, inserted, step, n_before, n_after))
        })
        .await
        .map_err(|e| e.to_string())??;

    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        write_cleaned_to_slot(&mut app, &key, df_out)?;
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!(
                "Ré-indexation de {} : {} lignes insérées (pas {} s) — {} -> {} lignes",
                key, inserted, step, n_before, n_after
            ),
        );
        crate::utils::session_persist::save(&app);
    }

    Ok(serde_json::json!({
        "dataset": key,
        "inserted": inserted,
        "step_seconds": step,
        "n_before": n_before,
        "n_after": n_after,
    }))
}

// =============================================================================
// Voie A — detect-only preview (no state mutation)
// =============================================================================
// Lets the UI show the user which points would be flagged before they commit
// to running the cleaning. Returns per-column indices of outliers + the
// timestamp/value for each so the chart can scatter-plot them on top of the
// raw curve without a second fetch.

#[tauri::command]
pub async fn cleaning_detect_only(
    state: State<'_, AppState>,
    dataset: String,
    method: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let method: CleaningMethod =
        serde_json::from_value(method).map_err(|e| format!("méthode invalide: {}", e))?;
    method.validate().map_err(|e| e.to_string())?;

    let df = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        dataset_ref(&app, &dataset)?.clone()
    };
    let n_total = df.height();

    let per_col = tokio::task::spawn_blocking(move || {
        data_cleaning::detect_outliers_only(&df, &method)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let mut total = 0usize;
    let per_col_json: Vec<serde_json::Value> = per_col
        .into_iter()
        .map(|(col, indices)| {
            total += indices.len();
            serde_json::json!({
                "column": col,
                "n_outliers": indices.len(),
                "indices": indices,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "n_total": n_total,
        "n_outliers": total,
        "pct_data_modified": if n_total == 0 { 0.0 } else { 100.0 * total as f64 / n_total as f64 },
        "per_column": per_col_json,
    }))
}

// =============================================================================
// Voie A — classical (one-shot)
// =============================================================================

#[tauri::command]
pub async fn cleaning_run_classical(
    state: State<'_, AppState>,
    dataset: String,
    method: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let method: CleaningMethod =
        serde_json::from_value(method).map_err(|e| format!("méthode invalide: {}", e))?;
    method.validate().map_err(|e| e.to_string())?;

    // Clone the source DataFrame out of the lock + capture a one-shot
    // pre-cleaning snapshot so the UI can render before/after comparisons
    // after the slot has been overwritten in-place.
    let df = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        let pristine = dataset_ref(&app, &dataset)?.clone();
        app.cleaning_pre_snapshots
            .entry(dataset.clone())
            .or_insert(pristine);
        // Start from the latest cleaned version so successive runs stack
        // instead of each one restarting from the untouched import.
        working_df(&app, &dataset)?
    };

    let method_name = method.name().to_string();
    // Extract target columns BEFORE moving the method into the worker thread
    // so we can attach them to the cleaning provenance after the run.
    let targets: Vec<String> = match &method {
        CleaningMethod::Iqr(c) => c.target_columns.clone(),
        CleaningMethod::ZScore(c) => c.target_columns.clone(),
        CleaningMethod::Mad(c) => c.target_columns.clone(),
        CleaningMethod::RollingWindow(c) => c.target_columns.clone(),
        CleaningMethod::ReverseDetection(c) => c.target_columns.clone(),
        CleaningMethod::IsolationForest(c) => c.target_columns.clone(),
        CleaningMethod::AbsoluteBounds { target_columns, .. } => target_columns.clone(),
        _ => Vec::new(),
    };
    let (cleaned_df, report): (DataFrame, CleaningReport) =
        tokio::task::spawn_blocking(move || {
            data_cleaning::apply_cleaning_method(&df, &method)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    // Fall back to the report's per-column keys if `targets` is empty (e.g.
    // user passed an empty list and the method auto-targeted all numerics).
    let targets = if targets.is_empty() {
        report.per_column.keys().cloned().collect()
    } else {
        targets
    };

    // Store the cleaned result. Writing to the right slot depends on which
    // dataset the user cleaned: cleaning the raw / cleaned slot updates
    // `cleaned_data` (used by the TTD pipeline as input). Cleaning a derived
    // dataset (T600, Tslope, …) writes back to that derived slot, leaving
    // `cleaned_data` untouched so the pipeline can still run from raw.
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        write_cleaned_to_slot(&mut app, &dataset, cleaned_df)?;
        app.cleaning_source_dataset = Some(dataset.clone());
        app.cleaning_method_label = Some(method_name.clone());
        app.cleaning_path = Some("classical".to_string());
        // Union with whatever was already there — the same slot may have
        // been touched by a previous Voie B / manual run. Overwriting
        // would silently drop columns the user already cleaned.
        for c in targets {
            if !app.cleaning_target_columns.iter().any(|x| x == &c) {
                app.cleaning_target_columns.push(c);
            }
        }
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!(
                "Nettoyage classique '{}' appliqué: {} outliers sur {} points ({:.2}%)",
                method_name, report.n_outliers_replaced, report.n_total, report.pct_data_modified
            ),
        );
        crate::utils::session_persist::save(&app);
    }

    Ok(serde_json::to_value(report).map_err(|e| e.to_string())?)
}

// =============================================================================
// Gap filling — Voie A (classical fill of NaN holes, no detection)
// =============================================================================
// Used by the Gap Filling page when the user picks the "classique" path:
// they select columns + a fill method (linear / ffill / bfill / rolling mean)
// and we bridge whatever NaN holes already exist. No model training needed.

#[tauri::command]
pub async fn gap_filling_run_classical(
    state: State<'_, AppState>,
    dataset: String,
    columns: Vec<String>,
    method: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let method: GapFillMethod = serde_json::from_value(method)
        .map_err(|e| format!("méthode de gap-filling invalide: {}", e))?;

    // Pull the latest version of the source dataset. If a previous step
    // cleaned the SAME dataset, pull that version (so successive fills stack).
    // Otherwise fall back to the original `dataset_ref` slot. Also stash a
    // one-shot pre-fill snapshot if none exists yet for this slot.
    let df = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        let same_source = app.cleaning_source_dataset.as_deref() == Some(dataset.as_str());
        let original = if same_source {
            match dataset.as_str() {
                "raw" | "cleaned" => app
                    .cleaned_data
                    .clone()
                    .unwrap_or_else(|| dataset_ref(&app, &dataset).unwrap().clone()),
                _ => dataset_ref(&app, &dataset)?.clone(),
            }
        } else {
            dataset_ref(&app, &dataset)?.clone()
        };
        app.cleaning_pre_snapshots
            .entry(dataset.clone())
            .or_insert_with(|| original.clone());
        original
    };

    let (filled_df, report): (DataFrame, GapFillReport) =
        tokio::task::spawn_blocking(move || {
            data_cleaning::fill_gaps_classical(&df, &columns, method)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    let filled_columns: Vec<String> = report.per_column.keys().cloned().collect();
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        write_cleaned_to_slot(&mut app, &dataset, filled_df)?;
        // Only overwrite the source-dataset hint if we don't already have one
        // (gap-fill stacks on top of a previous cleaning step — preserve its
        // origin so the UI keeps showing where the cleaning started).
        if app.cleaning_source_dataset.is_none() {
            app.cleaning_source_dataset = Some(dataset.clone());
        }
        app.cleaning_method_label = Some(report.method.clone());
        app.cleaning_path = Some("gapfill_classical".to_string());
        // Merge with any prior cleaning target list so the chip set keeps
        // growing as the user touches additional columns.
        for col in filled_columns {
            if !app.cleaning_target_columns.contains(&col) {
                app.cleaning_target_columns.push(col);
            }
        }
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!(
                "Gap filling classique '{}': {} trous comblés sur {} (reste {})",
                report.method, report.total_filled, report.total_gaps_before, report.total_gaps_after
            ),
        );
        crate::utils::session_persist::save(&app);
    }

    Ok(serde_json::to_value(report).map_err(|e| e.to_string())?)
}

// =============================================================================
// Voie B — train ML cleaner
// =============================================================================

#[tauri::command]
pub async fn cleaning_train_ml(
    state: State<'_, AppState>,
    method: serde_json::Value,
    dataset: Option<String>,
) -> Result<serde_json::Value, String> {
    let method: CleaningMethod =
        serde_json::from_value(method).map_err(|e| format!("méthode invalide: {}", e))?;
    method.validate().map_err(|e| e.to_string())?;
    let method_name = method.name().to_string();
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // Clone data + env out of the lock. `dataset_key` selects which DataFrame
    // drives training (raw/cleaned/tslope/…/sap_flow) — useful when cleaning a
    // derived signal like T600 rather than raw sensor data.
    let (source_df, env_df, ds_key) = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let df = dataset_ref(&app, &dataset_key)?.clone();
        (df, app.env_data.clone(), dataset_key.clone())
    };

    // Train on a blocking thread. MLCleaner is constructed inside the
    // closure so Send bounds stay happy.
    let result: Result<(MLCleaner, Option<CleaningTrainingMetrics>), String> =
        tokio::task::spawn_blocking(move || -> Result<(MLCleaner, Option<CleaningTrainingMetrics>), String> {
            let mut cleaner = MLCleaner::new(method).map_err(|e| e.to_string())?;
            cleaner
                .train(&source_df, env_df.as_ref())
                .map_err(|e| e.to_string())?;
            let metrics = cleaner.metrics().cloned();
            Ok((cleaner, metrics))
        })
        .await
        .map_err(|e| e.to_string())?;

    let (cleaner, metrics) = result?;

    // Store trained model in state.
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        app.ml_cleaner = Some(cleaner);
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!(
                "Modèle '{}' entraîné sur {}: RMSE={:.4}, R²={:.4}",
                method_name,
                ds_key,
                metrics.as_ref().map(|m| m.rmse).unwrap_or(f64::NAN),
                metrics.as_ref().map(|m| m.r_squared).unwrap_or(f64::NAN),
            ),
        );
    }

    Ok(serde_json::json!({
        "model": method_name,
        "dataset": ds_key,
        "metrics": metrics,
    }))
}

// =============================================================================
// Voie B — detection-only preview (no replacement)
// =============================================================================

/// Run the trained ML cleaner against the dataset and return outlier indices
/// per column WITHOUT writing predictions back. Mirrors Voie A's
/// `cleaning_detect_only` so the frontend can show a "validate before
/// apply" preview with red dots, then call `cleaning_apply_ml` to commit.
#[tauri::command]
pub async fn cleaning_detect_ml(
    state: State<'_, AppState>,
    threshold_factor: Option<f64>,
    dataset: Option<String>,
) -> Result<serde_json::Value, String> {
    let k = threshold_factor.unwrap_or(3.0).clamp(1.0, 10.0);
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // Borrow the cleaner without consuming it — detection is read-only so we
    // don't need to take/reinsert like cleaning_apply_ml does.
    let (cleaner_clone_input, source_df, env_df) = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        if app.ml_cleaner.is_none() {
            return Err("Aucun modèle entraîné — appelle cleaning_train_ml d'abord".to_string());
        }
        let df = dataset_ref(&app, &dataset_key)?.clone();
        // We pass the cleaner by reference inside spawn_blocking; clone() is
        // not implemented on MLCleaner (Box<dyn Predictor> is not Clone), so
        // we *take* it and put it back at the end like apply does. Detection
        // is fast though — we hold the lock briefly each side.
        (true, df, app.env_data.clone())
    };
    let _ = cleaner_clone_input;

    // Take ownership for the detection run, restore afterwards. This keeps
    // the MLCleaner type Sync-free and avoids deep refactors.
    let cleaner = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        app.ml_cleaner.take().ok_or_else(|| {
            "Aucun modèle entraîné — appelle cleaning_train_ml d'abord".to_string()
        })?
    };

    let result: Result<(MLCleaner, Vec<MLDetectionPerColumn>), String> =
        tokio::task::spawn_blocking(move || {
            let detected = cleaner
                .detect(&source_df, env_df.as_ref(), k)
                .map_err(|e| e.to_string())?;
            Ok((cleaner, detected))
        })
        .await
        .map_err(|e| e.to_string())?;

    let (cleaner, per_col) = match result {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let target_cols = cleaner.target_columns().to_vec();
    let n_targets = target_cols.len();
    let n_total = source_df_height(&state, &dataset_key)?;
    // Aggregate stats over targets only (test columns are reported separately
    // in per_col but excluded from the global pct so the user sees the impact
    // on the columns that will actually be modified by apply).
    let n_outliers: usize = per_col[..n_targets.min(per_col.len())]
        .iter()
        .map(|c| c.n_outliers)
        .sum();
    let pct = if n_total == 0 { 0.0 } else { 100.0 * (n_outliers as f64) / (n_total as f64) };

    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        app.ml_cleaner = Some(cleaner);
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            format!(
                "Détection ML: {} outliers détectés sur {} lignes (seuil={}σ)",
                n_outliers, n_total, k
            ),
        );
    }

    Ok(serde_json::json!({
        "n_total": n_total,
        "n_outliers": n_outliers,
        "pct_data_modified": pct,
        "per_column": per_col,
        "threshold_factor": k,
    }))
}

fn source_df_height(state: &State<'_, AppState>, key: &str) -> Result<usize, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    Ok(dataset_ref(&app, key)?.height())
}

// =============================================================================
// Voie B — apply trained ML cleaner (detection + gap filling in one shot)
// =============================================================================

#[tauri::command]
pub async fn cleaning_apply_ml(
    state: State<'_, AppState>,
    threshold_factor: Option<f64>,
    dataset: Option<String>,
    mode: Option<MLCleanMode>,
) -> Result<serde_json::Value, String> {
    let k = threshold_factor.unwrap_or(3.0).clamp(1.0, 10.0);
    // Default mode = MarkOnly. Detection page uses this; Gap Filling page
    // sends GapFillOnly; the legacy one-shot is opt-in via DetectAndReplace.
    let mode = mode.unwrap_or(MLCleanMode::MarkOnly);
    let dataset_key = dataset.unwrap_or_else(|| "raw".to_string());

    // Take the cleaner + data out of the lock.
    let (cleaner, source_df, env_df) = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        let c = app
            .ml_cleaner
            .take()
            .ok_or_else(|| "Aucun modèle entraîné — appelle cleaning_train_ml d'abord".to_string())?;
        let df = dataset_ref(&app, &dataset_key)?.clone();
        (c, df, app.env_data.clone())
    };

    // Apply on a blocking thread.
    let (cleaner, cleaned_y, per_col): (MLCleaner, ndarray::Array2<f64>, Vec<(usize, usize)>) =
        match tokio::task::spawn_blocking(
            move || -> Result<(MLCleaner, ndarray::Array2<f64>, Vec<(usize, usize)>), String> {
                let (y, stats) = cleaner
                    .clean(&source_df, env_df.as_ref(), k, mode)
                    .map_err(|e| e.to_string())?;
                Ok((cleaner, y, stats))
            },
        )
        .await
        .map_err(|e| e.to_string())?
        {
            Ok(t) => t,
            Err(e) => {
                // On failure, re-insert a fresh empty slot; model is lost.
                return Err(e);
            }
        };

    // Merge cleaned target columns back into a full DataFrame. We read the
    // same dataset we trained on so row counts align with the predictions.
    // Test columns are NOT overwritten — their stats are reported separately.
    let target_cols = cleaner.target_columns().to_vec();
    let test_cols = cleaner.test_columns().to_vec();
    let all_cols = cleaner.all_predicted_columns();
    // Read the latest version of the source dataset to merge predictions
    // into + snapshot the original for before/after comparisons.
    let mut merged = {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        let is_raw_or_cleaned = matches!(dataset_key.as_str(), "raw" | "cleaned");
        let base = if is_raw_or_cleaned {
            if let Some(df) = app.cleaned_data.as_ref() {
                df.clone()
            } else {
                dataset_ref(&app, &dataset_key)?.clone()
            }
        } else {
            dataset_ref(&app, &dataset_key)?.clone()
        };
        app.cleaning_pre_snapshots
            .entry(dataset_key.clone())
            .or_insert_with(|| base.clone());
        base
    };

    // Replace only target columns (j < target_cols.len() in cleaned_y).
    for (j, col_name) in target_cols.iter().enumerate() {
        let col_vals: Vec<Option<f64>> = (0..cleaned_y.nrows())
            .map(|i| {
                let v = cleaned_y[[i, j]];
                if v.is_finite() { Some(v) } else { None }
            })
            .collect();
        let new_s = Series::new(col_name.as_str().into(), &col_vals);
        merged
            .replace(col_name, new_s)
            .map_err(|e| format!("replace column {}: {}", col_name, e))?;
    }

    // Count only target-column outliers/gaps for the global totals — test
    // columns get their own stats so lab users can compare cross-validation
    // impact vs target impact.
    let n_outliers: usize = per_col[..target_cols.len()].iter().map(|(o, _)| o).sum();
    let n_gaps: usize = per_col[..target_cols.len()].iter().map(|(_, g)| g).sum();

    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        // Route the merged dataframe to the slot it came from, same as the
        // classical path, so the TTD pipeline doesn't end up trying to read
        // T600 columns out of cleaned_data.
        write_cleaned_to_slot(&mut app, &dataset_key, merged)?;
        app.ml_cleaner = Some(cleaner); // put it back for re-use
        app.cleaning_source_dataset = Some(dataset_key.clone());
        app.cleaning_method_label = Some("ML".to_string());
        app.cleaning_path = Some("ml".to_string());
        // Union, same rationale as the classical path — preserves columns
        // touched by previous Voie A / manual operations on this slot.
        for c in &target_cols {
            if !app.cleaning_target_columns.iter().any(|x| x == c) {
                app.cleaning_target_columns.push(c.clone());
            }
        }
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            match mode {
                MLCleanMode::MarkOnly => format!(
                    "Détection ML : {} outliers marqués NaN, {} trous laissés tels quels (seuil={}σ) — utilise Gap Filling pour combler",
                    n_outliers, n_gaps, k
                ),
                MLCleanMode::GapFillOnly => format!(
                    "Gap Filling ML : {} trous comblés par le modèle",
                    n_gaps
                ),
                MLCleanMode::DetectAndReplace => format!(
                    "Nettoyage ML appliqué : {} outliers remplacés, {} gaps comblés (seuil={}σ)",
                    n_outliers, n_gaps, k
                ),
            },
        );
        crate::utils::session_persist::save(&app);
    }

    // Report per-column stats for ALL predicted cols (target + test) with a
    // flag so the UI can display them in two sections.
    let per_col_json: Vec<serde_json::Value> = all_cols
        .iter()
        .zip(per_col.iter())
        .enumerate()
        .map(|(j, (c, (o, g)))| {
            let is_test = j >= target_cols.len();
            serde_json::json!({
                "column": c,
                "outliers_replaced": if is_test { 0 } else { *o },
                "outliers_detected": o,
                "gaps_filled": if is_test { 0 } else { *g },
                "gaps_detected": g,
                "is_test": is_test,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "total_outliers_replaced": n_outliers,
        "total_gaps_filled": n_gaps,
        "per_column": per_col_json,
        "threshold_factor": k,
        "n_target_cols": target_cols.len(),
        "n_test_cols": test_cols.len(),
    }))
}

// =============================================================================
// Voie B — reset (drop the trained cleaner)
// =============================================================================

/// Drop the cleaned_data slot so the pipeline / table will fall back to
/// raw_data on the next read. Lets the user undo a botched cleaning without
/// re-importing the file.
#[tauri::command]
pub fn cleaning_reset_to_raw(state: State<'_, AppState>) -> Result<(), String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    let had = app.cleaned_data.is_some();
    app.cleaned_data = None;
    app.cleaning_source_dataset = None;
    app.cleaning_method_label = None;
    app.cleaning_path = None;
    app.cleaning_target_columns.clear();
    app.cleaning_locked_stages.clear();
    app.cleaning_pre_snapshots.clear();
    logger::add_log(
        &mut app.logs,
        LogLevel::Info,
        if had { "Cleaned data dropped — fallback to raw" } else { "No cleaned data to drop" }.to_string(),
    );
    crate::utils::session_persist::save(&app);
    Ok(())
}

/// Wipe the entire on-disk session: clears every cached parquet under
/// the OS session dir AND resets the in-memory state to defaults. Used
/// by the "Réinitialiser tout" UI control when the user wants to start
/// fresh after the auto-restore has rehydrated stale data.
#[tauri::command]
pub fn clear_session(state: State<'_, AppState>) -> Result<(), String> {
    {
        let mut app = state.inner.lock().map_err(|e| e.to_string())?;
        *app = crate::state::AppData::default();
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            "Session wiped (in-memory state + disk)".to_string(),
        );
    }
    crate::utils::session_persist::clear();
    Ok(())
}

// =============================================================================
// Cleaning provenance — surfaces which source dataset was cleaned + by what
// method, so the Gap Filling page can show "the cleaning was done on T600"
// without the user having to remember.
// =============================================================================

/// Return the first `page_size` rows of the pre-cleaning snapshot for a
/// given dataset slot. Used by the Gap Filling page to render an "avant
/// nettoyage" overlay against the current (cleaned) state. Returns null if
/// no snapshot exists for that key.
#[tauri::command]
pub async fn get_cleaning_pre_rows(
    state: State<'_, AppState>,
    dataset: String,
    page_size: usize,
) -> Result<serde_json::Value, String> {
    let df_opt = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        app.cleaning_pre_snapshots.get(&dataset).cloned()
    };
    let df = match df_opt {
        Some(d) => d,
        None => return Ok(serde_json::json!({ "rows": null })),
    };
    let limit = page_size.min(df.height());
    let slice = df.slice(0, limit);
    let rows = crate::utils::dataframe_serde::dataframe_page_to_json(&slice)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "rows": rows }))
}

#[tauri::command]
pub fn get_cleaning_info(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let src = app.cleaning_source_dataset.as_deref();
    // Pipeline can only be re-run from raw — cleaning a derived slot doesn't
    // feed back into upstream computations, so the UI should hide the
    // "Re-run pipeline on Cleaned" CTA in that case.
    let pipeline_rerun_safe = matches!(src, Some("raw") | Some("cleaned") | None);
    let has_anything = app.cleaned_data.is_some()
        || app.cleaning_source_dataset.is_some();
    Ok(serde_json::json!({
        "has_cleaned":           has_anything,
        "source_dataset":        app.cleaning_source_dataset,
        "method_label":          app.cleaning_method_label,
        "path":                  app.cleaning_path,
        "n_rows":                app.cleaned_data.as_ref().map(|df| df.height()),
        "target_columns":        app.cleaning_target_columns,
        "pipeline_rerun_safe":   pipeline_rerun_safe,
        "locked_stages":         app.cleaning_locked_stages,
    }))
}

// =============================================================================
// Cleaning locks — "Rendre permanent"
// =============================================================================
// A cleaning applied to a DERIVED stage (T600, T0, …) lives in that stage's
// `results.*` slot and is used everywhere (table / viz / export). Its only
// threat is a full `run_pipeline`, which rebuilds every stage from raw and
// silently wipes it. Locking a stage makes that full run REFUSE to overwrite
// it unless the user explicitly forces it (see calculations::run_pipeline).
//
// Only derived stages can be locked: raw/cleaned cleanings already survive a
// full run (cleaned_data is the pipeline's input), so a lock there is moot.

const LOCKABLE_STAGES: [&str; 9] =
    ["tslope", "baseline", "delta_t", "t600", "tm", "stm", "tmi", "k", "sap_flow"];

/// Lock a cleaned derived stage so a full recompute can't silently wipe it.
/// Idempotent; returns the full locked-stage list.
#[tauri::command]
pub fn cleaning_lock_permanent(
    state: State<'_, AppState>,
    dataset: String,
) -> Result<serde_json::Value, String> {
    if !LOCKABLE_STAGES.contains(&dataset.as_str()) {
        return Err(format!(
            "Seuls les étages calculés peuvent être verrouillés (reçu « {} »). Un nettoyage sur les données brutes est déjà conservé par le recalcul.",
            dataset
        ));
    }
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    if !app.cleaning_locked_stages.iter().any(|s| s == &dataset) {
        app.cleaning_locked_stages.push(dataset.clone());
        logger::add_log(
            &mut app.logs,
            LogLevel::Success,
            format!("Nettoyage de « {} » rendu permanent (verrouillé contre le recalcul complet)", dataset),
        );
        crate::utils::session_persist::save(&app);
    }
    Ok(serde_json::json!({ "locked_stages": app.cleaning_locked_stages }))
}

/// Remove a lock previously set by `cleaning_lock_permanent`. Idempotent.
#[tauri::command]
pub fn cleaning_unlock_permanent(
    state: State<'_, AppState>,
    dataset: String,
) -> Result<serde_json::Value, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    let before = app.cleaning_locked_stages.len();
    app.cleaning_locked_stages.retain(|s| s != &dataset);
    if app.cleaning_locked_stages.len() != before {
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            format!("Verrou retiré de « {} »", dataset),
        );
        crate::utils::session_persist::save(&app);
    }
    Ok(serde_json::json!({ "locked_stages": app.cleaning_locked_stages }))
}

#[tauri::command]
pub fn cleaning_reset_ml(state: State<'_, AppState>) -> Result<(), String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    if app.ml_cleaner.take().is_some() {
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            "Modèle ML déchargé".to_string(),
        );
    }
    Ok(())
}

// =============================================================================
// Ml cleaner status (used by UI to know whether "Apply" is available)
// =============================================================================

#[tauri::command]
pub fn cleaning_ml_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    if let Some(cleaner) = app.ml_cleaner.as_ref() {
        let metrics = cleaner.metrics().cloned();
        Ok(serde_json::json!({
            "loaded": true,
            "target_columns": cleaner.target_columns(),
            "predictor_columns": cleaner.predictor_columns(),
            "metrics": metrics,
        }))
    } else {
        Ok(serde_json::json!({ "loaded": false }))
    }
}

// =============================================================================
// Boot-time snapshot — used by the frontend to restore UI flags after a
// reload (e.g., webview suspended on system sleep) without re-importing
// data the backend still has.
// =============================================================================

/// One-shot status of the entire backend state. The frontend calls this
/// at app startup to know what's already loaded and skip the empty-state
/// flow when the underlying Rust process survived a webview reload.
#[tauri::command]
pub fn get_app_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;

    // Raw data presence — drives `dataLoaded` and the "you have a file
    // imported" UI affordances on the import / table / detection pages.
    let raw = app.raw_data.as_ref();
    let n_rows = raw.map(|d| d.height()).unwrap_or(0);
    let n_cols = raw.map(|d| d.width()).unwrap_or(0);
    let columns: Vec<String> = raw
        .map(|d| d.get_column_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    // The TTD pipeline writes its outputs into `results.*`. If `tm` is
    // populated we can be confident the calculations have been run at
    // least once — the UI uses this to unlock dependent tabs.
    let calculations_complete = app.results.tm.is_some();

    let env_loaded = app.env_data.is_some();
    let ml_loaded = app.ml_cleaner.is_some();
    let (ml_targets, ml_predictors) = match app.ml_cleaner.as_ref() {
        Some(c) => (c.target_columns().to_vec(), c.predictor_columns().to_vec()),
        None => (Vec::new(), Vec::new()),
    };

    Ok(serde_json::json!({
        "data_loaded": raw.is_some(),
        "file_path": app.file_path,
        "sheet_name": app.sheet_name,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns,
        "calculations_complete": calculations_complete,
        "env_loaded": env_loaded,
        "env_file_path": app.env_file_path,
        "ml_loaded": ml_loaded,
        "ml_target_columns": ml_targets,
        "ml_predictor_columns": ml_predictors,
        "cleaning_source_dataset": app.cleaning_source_dataset,
        "cleaning_method_label": app.cleaning_method_label,
        "cleaning_path": app.cleaning_path,
        "cleaning_target_columns": app.cleaning_target_columns,
        // Tells the frontend "this state came from disk, not a fresh
        // import" — used to fire a discreet "Session restaurée" toast
        // on boot rather than re-prompting for files.
        "session_restored": raw.is_some(),
    }))
}
