use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::core::types::LogLevel;
use crate::state::AppState;
use crate::utils::logger;

// ---------------------------------------------------------------------------
// Dataset key → DataFrame dispatch
//
// Centralised so the aggregation, listing, and saved-aggregation flows all
// resolve the same key the same way. Add new sources here only.
// ---------------------------------------------------------------------------

fn resolve_dataset(
    app: &crate::state::AppData,
    dataset: &str,
) -> Option<polars::prelude::DataFrame> {
    use polars::prelude::DataFrame;
    fn cloned(df: Option<&DataFrame>) -> Option<DataFrame> { df.cloned() }
    match dataset {
        "raw" => cloned(app.raw_data.as_ref()),
        "cleaned" => cloned(app.cleaned_data.as_ref()),
        "env" => cloned(app.env_data.as_ref()),
        // TTD pipeline
        "tslope" => cloned(app.results.tslope.as_ref()),
        "baseline" => cloned(app.results.baseline.as_ref()),
        "delta_t" => cloned(app.results.delta_t.as_ref()),
        "t600" => cloned(app.results.t600.as_ref()),
        "tm" => cloned(app.results.tm.as_ref()),
        "stm" => cloned(app.results.stm.as_ref()),
        "tmi" => cloned(app.results.tmi.as_ref()),
        "k" => cloned(app.results.k.as_ref()),
        "sap_flow" => cloned(app.results.sap_flow.as_ref()),
        // TTD+
        "ttdplus_fourier" => cloned(app.results.ttdplus_fourier.as_ref()),
        "ttdplus_refs" => cloned(app.results.ttdplus_refs.as_ref()),
        "ttdplus_sap_flow" => cloned(app.results.ttdplus_sap_flow.as_ref()),
        // Calculs avancés
        "jh" => cloned(app.results.jh.as_ref()),
        "jhp" => cloned(app.results.jhp.as_ref()),
        "qh" => cloned(app.results.qh.as_ref()),
        "qd" => cloned(app.results.qd.as_ref()),
        _ => None,
    }
}

/// List every dataset key that currently holds a DataFrame, with row/col
/// counts. Used by the frontend to populate the source dropdown.
#[tauri::command]
pub async fn list_aggregation_sources(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let keys = [
        "raw", "cleaned", "env",
        "tslope", "baseline", "delta_t", "t600", "tm", "stm", "tmi", "k", "sap_flow",
        "ttdplus_fourier", "ttdplus_refs", "ttdplus_sap_flow",
        "jh", "jhp", "qh", "qd",
    ];
    let mut out: Vec<serde_json::Value> = Vec::new();
    for k in &keys {
        if let Some(df) = resolve_dataset(&app, k) {
            out.push(serde_json::json!({
                "key": k,
                "rows": df.height(),
                "cols": df.width(),
                "columns": df.get_column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            }));
        }
    }
    Ok(serde_json::json!({ "sources": out }))
}

// ---------------------------------------------------------------------------
// aggregate_data — supports an explicit `dataset` argument
// ---------------------------------------------------------------------------

/// `cross_sensor`: when true, the N per-column per-period values are
/// collapsed into ONE number per period using the same operation across
/// sensors. Result has a single column "<Op>_capteurs" instead of N.
/// Default false (per-column output, historical behaviour).
///
/// `custom_seconds`: sub-day granularity for the "Personnalisé" period. Wins
/// over `custom_days` when both are set. Lets the user bucket by N minutes
/// (e.g. 30 → half-hour periods) or N hours.
#[tauri::command]
pub async fn aggregate_data(
    state: State<'_, AppState>,
    dataset: Option<String>,
    columns: Vec<String>,
    period: String,
    operation: String,
    custom_days: Option<i64>,
    save_as: Option<String>,
    cross_sensor: Option<bool>,
    custom_seconds: Option<i64>,
    profile: Option<String>,
    profile_by: Option<String>,
    // Minimum number of non-missing steps a bucket must hold to be reported.
    // A daily sum over 6 valid half-hours is not a daily total, it is a
    // fragment that lands on the same axis as complete days and drags every
    // statistic down — convention C4 of the processing report calls a day
    // valid only at 48/48. `None` keeps the historical behaviour (report
    // whatever is there).
    min_count: Option<usize>,
    // Multiply every aggregated value. Carries the unit conversion the report
    // needs: summing 48 half-hourly values of l dm⁻² h⁻¹ gives l dm⁻² j⁻¹ only
    // after ×0.5, and doing it by hand on every export invites mistakes.
    scale: Option<f64>,
) -> Result<serde_json::Value, String> {
    // Resolve dataset under a short lock then release.
    let (df, dataset_key) = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let key = dataset.unwrap_or_else(|| "sap_flow".to_string());
        // Fall back through sap_flow → cleaned → raw when the requested key
        // is empty, mirroring the historical default.
        let df = resolve_dataset(&app, &key)
            .or_else(|| resolve_dataset(&app, "sap_flow"))
            .or_else(|| resolve_dataset(&app, "cleaned"))
            .or_else(|| resolve_dataset(&app, "raw"))
            .ok_or_else(|| format!("Dataset '{}' indisponible ou pipeline non lancé.", key))?;
        (df, key)
    };

    let columns_clone = columns.clone();
    let period_clone = period.clone();
    let operation_clone = operation.clone();
    let dataset_key_for_inner = dataset_key.clone();
    let cross_sensor = cross_sensor.unwrap_or(false);
    // Normalise profile: empty / "none" / "Global"-without-profile → disabled.
    let profile_opt = profile.as_deref().filter(|p| !p.is_empty() && *p != "none").map(String::from);
    let profile_clone = profile_opt.clone();
    let profile_by_clone = profile_by.clone();
    let result_json = tokio::task::spawn_blocking(move || {
        aggregate_inner(
            &df, &columns_clone, &period_clone, &operation_clone,
            custom_days, custom_seconds, &dataset_key_for_inner, cross_sensor,
            profile_clone.as_deref(), profile_by_clone.as_deref(),
            min_count.unwrap_or(0), scale.unwrap_or(1.0),
        )
    })
    .await
    .map_err(|e| e.to_string())??;

    // Human label for the "period" shown in the saved list / header. In
    // profile mode it reflects the diurnal-profile grouping.
    let period_label = if profile_opt.is_some() {
        let outer = match profile_by.as_deref().unwrap_or("Global") {
            "Journalier" => "jour", "Hebdomadaire" => "semaine",
            "Mensuel" => "mois", "Annuel" => "année", _ => "tout le fichier",
        };
        format!("Profil horaire / {}", outer)
    } else {
        period.clone()
    };

    // Optionally persist the result on disk so the user can list it later.
    let saved = if let Some(name) = save_as {
        let name = name.trim().to_string();
        let name = if name.is_empty() {
            // Auto-name: "<dataset> · <period> · <op>"
            format!("{} · {} · {}", dataset_key, period_label, operation)
        } else {
            name
        };
        // Persist with the ACTUAL output columns from the result (which
        // differ from the request when cross_sensor reduced everything to
        // one column), so the saved-aggregations list reads correctly.
        let actual_cols: Vec<String> = result_json
            .get("columns")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_else(|| columns.clone());
        let saved_meta = persist_aggregation(
            &name, &dataset_key, &period_label, &operation, &actual_cols, &result_json,
            profile_opt.as_deref(), profile_by.as_deref(),
            &columns, cross_sensor,
        ).map_err(|e| e.to_string())?;
        {
            let mut app = state.inner.lock().map_err(|e| e.to_string())?;
            logger::add_log(
                &mut app.logs,
                LogLevel::Success,
                format!("Agrégation '{}' sauvegardée ({} périodes)", saved_meta.name, saved_meta.n_periods),
            );
        }
        Some(saved_meta)
    } else {
        None
    };

    let mut out = result_json;
    if let Some(meta) = saved {
        out["saved"] = serde_json::to_value(meta).unwrap_or(serde_json::Value::Null);
    }
    out["dataset"] = serde_json::Value::String(dataset_key);
    Ok(out)
}

fn aggregate_inner(
    df: &polars::prelude::DataFrame,
    columns: &[String],
    period: &str,
    operation: &str,
    custom_days: Option<i64>,
    custom_seconds: Option<i64>,
    dataset_key: &str,
    cross_sensor: bool,
    profile: Option<&str>,
    profile_by: Option<&str>,
    min_count: usize,
    scale: f64,
) -> Result<serde_json::Value, String> {
    // Find the timestamp column. Qd uses DATE (Polars Date) — every other
    // dataset uses TIMESTAMP (datetime). Both branches end up as strings of
    // the form "YYYY-MM-DD…" before bucketing.
    let ts_col_name = df
        .get_column_names()
        .into_iter()
        .find(|n| {
            let u = n.to_uppercase();
            u == "TIMESTAMP" || u == "DATE"
        })
        .map(|n| n.to_string())
        .ok_or_else(|| format!("Colonne TIMESTAMP/DATE introuvable dans dataset '{}'", dataset_key))?;

    let ts_series = df
        .column(&ts_col_name)
        .map_err(|e| e.to_string())?
        .cast(&polars::prelude::DataType::String)
        .map_err(|e| e.to_string())?;
    let ts_str = ts_series.str().map_err(|e| e.to_string())?;

    let days = custom_days.unwrap_or(7);
    // Sub-day granularity for Custom: `custom_seconds` wins when set (lets
    // the UI offer minute / hour level bucketing). Fallback: derive seconds
    // from custom_days for back-compat with older callers.
    let custom_secs: i64 = custom_seconds.filter(|&n| n > 0).unwrap_or(days * 86_400);

    // Collect timestamp strings once (reused by the profile pivot + bucketing).
    let ts_vec: Vec<&str> = ts_str.into_iter().map(|v| v.unwrap_or("")).collect();

    // Per-column f64 values, aligned by row index.
    let mut col_data: HashMap<String, Vec<f64>> = HashMap::new();
    for col_name in columns {
        if let Ok(col) = df.column(col_name) {
            if let Ok(cast) = col.cast(&polars::prelude::DataType::Float64) {
                if let Ok(f64ca) = cast.f64() {
                    col_data.insert(col_name.clone(), f64ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect());
                }
            }
        }
    }

    // Profile (diurnal-dynamics) mode → pivoted output: rows = hour-of-day,
    // one column (curve) per outer group (week/month/…). Returns early so the
    // Visualisation tab can overlay the curves on a synthetic time-of-day axis.
    if profile.is_some() {
        return aggregate_profile(&ts_vec, &col_data, columns, operation, profile_by, cross_sensor, min_count, scale);
    }

    let group_keys: Vec<String> = ts_vec.iter().map(|&s| {
        match period {
            "Journalier" => s.chars().take(10).collect(),
            "Horaire" => s.chars().take(13).collect(),
            "Mensuel" => s.chars().take(7).collect(),
            "Annuel" => s.chars().take(4).collect(),
            "Hebdomadaire" => week_start_key(s),
            "Custom" => custom_period_key_seconds(s, custom_secs),
            _ => s.chars().take(10).collect(),
        }
    }).collect();

    let all_groups: Vec<String> = group_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut result_rows: Vec<serde_json::Value> = Vec::new();

    for group in &all_groups {
        let mut row = serde_json::json!({ "period": group });

        for col_name in columns {
            if let Some(col_vals) = col_data.get(col_name) {
                let vals: Vec<f64> = group_keys
                    .iter()
                    .zip(col_vals.iter())
                    .filter_map(|(k, &v)| {
                        if k == group && v.is_finite() {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Below the required count the bucket is incomplete: report
                // nothing rather than a partial total that would pass for one.
                let agg = if vals.is_empty() || vals.len() < min_count {
                    serde_json::Value::Null
                } else {
                    let v = match operation {
                        "Somme" => vals.iter().sum::<f64>(),
                        "Minimum" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                        "Maximum" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        _ => vals.iter().sum::<f64>() / vals.len() as f64, // Moyenne
                    } * scale;
                    serde_json::json!((v * 10000.0).round() / 10000.0)
                };
                row[col_name.as_str()] = agg;
            }
        }

        result_rows.push(row);
    }

    // Cross-sensor reduction (optional): collapse the N per-column values
    // of each period into a SINGLE number using the same operation. Result
    // has one f64 column named "<Op>_capteurs" instead of one per sensor.
    if cross_sensor && !columns.is_empty() {
        let reduced_name = match operation {
            "Somme" => "Somme_capteurs",
            "Minimum" => "Min_capteurs",
            "Maximum" => "Max_capteurs",
            _ => "Moyenne_capteurs",
        };
        let reduced_rows: Vec<serde_json::Value> = result_rows.into_iter().map(|row| {
            let mut vals: Vec<f64> = Vec::with_capacity(columns.len());
            for c in columns {
                if let Some(v) = row.get(c.as_str()).and_then(|v| v.as_f64()) {
                    if v.is_finite() { vals.push(v); }
                }
            }
            let reduced = if vals.is_empty() {
                serde_json::Value::Null
            } else {
                let v = match operation {
                    "Somme" => vals.iter().sum::<f64>(),
                    "Minimum" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                    "Maximum" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    _ => vals.iter().sum::<f64>() / vals.len() as f64,
                };
                serde_json::json!((v * 10000.0).round() / 10000.0)
            };
            let mut new_row = serde_json::json!({ "period": row["period"].clone() });
            new_row[reduced_name] = reduced;
            new_row
        }).collect();
        return Ok(serde_json::json!({
            "period": period,
            "operation": operation,
            "columns": [reduced_name],
            "source_columns": columns,
            "cross_sensor": true,
            "profile": profile,
            "profile_by": profile_by,
            "rows": reduced_rows,
            "total": all_groups.len(),
        }));
    }

    Ok(serde_json::json!({
        "period": period,
        "operation": operation,
        "columns": columns,
        "profile": profile,
        "profile_by": profile_by,
        "rows": result_rows,
        "total": all_groups.len(),
    }))
}

/// Diurnal-profile aggregation (pivoted). For each outer group (week/month/…)
/// and each hour-of-day, aggregate the values; output one ROW per hour (x =
/// synthetic "1970-01-01 HH:00:00" so the time-series chart shows a clean 0–23h
/// axis) and one COLUMN (curve) per group — or per group · sensor — so the
/// Visualisation tab can overlay the diurnal cycles.
fn aggregate_profile(
    ts_vec: &[&str],
    col_data: &HashMap<String, Vec<f64>>,
    columns: &[String],
    operation: &str,
    profile_by: Option<&str>,
    cross_sensor: bool,
    min_count: usize,
    scale: f64,
) -> Result<serde_json::Value, String> {
    let by = profile_by.unwrap_or("Global");

    // Accumulate finite values per (group, hour) → sensor → Vec<f64>.
    let mut buckets: HashMap<(String, u32), HashMap<String, Vec<f64>>> = HashMap::new();
    for (i, &s) in ts_vec.iter().enumerate() {
        let hour = match s.get(11..13).and_then(|h| h.parse::<u32>().ok()) {
            Some(h) if h < 24 => h,
            _ => continue, // date-only / unparseable hour → skip
        };
        let group: String = match by {
            "Journalier"   => s.chars().take(10).collect(),
            "Hebdomadaire" => week_start_key(s),
            "Mensuel"      => s.chars().take(7).collect(),
            "Annuel"       => s.chars().take(4).collect(),
            _              => "Tout".to_string(),
        };
        let per_sensor = buckets.entry((group, hour)).or_default();
        for col in columns {
            if let Some(vals) = col_data.get(col) {
                if let Some(&v) = vals.get(i) {
                    if v.is_finite() {
                        per_sensor.entry(col.clone()).or_default().push(v);
                    }
                }
            }
        }
    }

    // Same two rules as the period path: a bucket thinner than `min_count`
    // reports nothing, and `scale` carries the unit conversion.
    let agg = |vals: &[f64]| -> Option<f64> {
        if vals.is_empty() || vals.len() < min_count {
            return None;
        }
        let v = match operation {
            "Somme" => vals.iter().sum::<f64>(),
            "Minimum" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
            "Maximum" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            _ => vals.iter().sum::<f64>() / vals.len() as f64,
        };
        Some(v * scale)
    };
    let reduce = |vals: &[f64]| -> f64 {
        match operation {
            "Somme" => vals.iter().sum::<f64>(),
            "Minimum" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
            "Maximum" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            _ => vals.iter().sum::<f64>() / vals.len() as f64,
        }
    };
    let round = |v: f64| (v * 10000.0).round() / 10000.0;

    let groups: Vec<String> = buckets.keys().map(|(g, _)| g.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    let hours: Vec<u32> = buckets.keys().map(|(_, h)| *h).collect::<BTreeSet<_>>().into_iter().collect();

    let out_columns: Vec<String> = if cross_sensor {
        groups.clone()
    } else {
        let mut c = Vec::new();
        for g in &groups {
            for s in columns { c.push(format!("{} · {}", g, s)); }
        }
        c
    };

    let mut rows: Vec<serde_json::Value> = Vec::with_capacity(hours.len());
    for h in &hours {
        let mut row = serde_json::json!({ "period": format!("1970-01-01 {:02}:00:00", h) });
        for g in &groups {
            let per_sensor = buckets.get(&(g.clone(), *h));
            if cross_sensor {
                let cell = per_sensor.and_then(|ps| {
                    let sensor_vals: Vec<f64> = columns.iter()
                        .filter_map(|s| ps.get(s).and_then(|v| agg(v)))
                        .collect();
                    if sensor_vals.is_empty() { None } else { Some(round(reduce(&sensor_vals))) }
                });
                row[g.as_str()] = match cell { Some(x) => serde_json::json!(x), None => serde_json::Value::Null };
            } else {
                for s in columns {
                    let name = format!("{} · {}", g, s);
                    let cell = per_sensor.and_then(|ps| ps.get(s)).and_then(|v| agg(v)).map(round);
                    row[name.as_str()] = match cell { Some(x) => serde_json::json!(x), None => serde_json::Value::Null };
                }
            }
        }
        rows.push(row);
    }

    Ok(serde_json::json!({
        "period": "Profil",
        "operation": operation,
        "columns": out_columns,
        "source_columns": columns,
        "cross_sensor": cross_sensor,
        "profile": "HourOfDay",
        "profile_by": by,
        "rows": rows,
        "total": hours.len(),
    }))
}

fn week_start_key(s: &str) -> String {
    if s.len() < 10 {
        return s.chars().take(10).collect();
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
        use chrono::Datelike;
        let mon_offset = d.weekday().num_days_from_monday() as i64;
        let monday = d - chrono::Duration::days(mon_offset);
        monday.format("%Y-%m-%d").to_string()
    } else {
        s.chars().take(10).collect()
    }
}

/// Generalised version of `custom_period_key` that buckets a timestamp
/// string into windows of `seconds` length, anchored to the Unix epoch.
/// Handles ISO-ish strings down to second resolution; falls back to the
/// day-based logic when the input is just a date.
///
/// Output format is chosen to match the bucket width: pure date for ≥ 1 day,
/// "YYYY-MM-DD HH:MM" for ≥ 1 minute, full "YYYY-MM-DD HH:MM:SS" otherwise.
fn custom_period_key_seconds(s: &str, seconds: i64) -> String {
    if seconds <= 0 || s.len() < 10 {
        return s.chars().take(10).collect();
    }
    // Try to parse a full datetime first, then fall back to date-only.
    let ndt = chrono::NaiveDateTime::parse_from_str(
        s.get(..19).unwrap_or(s), "%Y-%m-%d %H:%M:%S",
    )
    .or_else(|_| chrono::NaiveDateTime::parse_from_str(
        s.get(..19).unwrap_or(s), "%Y-%m-%dT%H:%M:%S",
    ))
    .or_else(|_| {
        chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d")
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap_or_default())
    });
    let dt = match ndt { Ok(d) => d, Err(_) => return s.chars().take(10).collect() };
    let epoch_date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let epoch = epoch_date.and_hms_opt(0, 0, 0).unwrap();
    let total_secs = (dt - epoch).num_seconds();
    let bucket_start_secs = (total_secs.div_euclid(seconds)) * seconds;
    let bucket_dt = epoch + chrono::Duration::seconds(bucket_start_secs);
    // Pick a display format that reflects the bucket's resolution.
    if seconds % 86_400 == 0 {
        bucket_dt.format("%Y-%m-%d").to_string()
    } else if seconds % 60 == 0 {
        bucket_dt.format("%Y-%m-%d %H:%M").to_string()
    } else {
        bucket_dt.format("%Y-%m-%d %H:%M:%S").to_string()
    }
}

#[allow(dead_code)]
fn custom_period_key(s: &str, days: i64) -> String {
    if s.len() < 10 {
        return s.chars().take(10).collect();
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let day_num = (d - epoch).num_days();
        let group_start = (day_num / days) * days;
        (epoch + chrono::Duration::days(group_start))
            .format("%Y-%m-%d")
            .to_string()
    } else {
        s.chars().take(10).collect()
    }
}

// ---------------------------------------------------------------------------
// Saved aggregations — disk-backed, listable, deletable
//
// Layout: <APPDATA>/ttdprocess/aggregations/<uuid>/{meta.json, result.json}
// Result is stored as JSON (small, already in the JSON shape the frontend
// expects). Switching to parquet would require deeper Polars integration on
// the frontend boundary — not worth it for typical aggregation sizes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedAggregationMeta {
    pub id: String,
    pub name: String,
    pub dataset: String,
    pub period: String,
    pub operation: String,
    pub columns: Vec<String>,
    pub n_periods: usize,
    pub created_at: String,
    // Profile (diurnal-dynamics) mode — None for a plain period aggregation.
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub profile_by: Option<String>,
    /// The columns the user SELECTED as input (before a `cross_sensor` collapse).
    /// `columns` above holds the ACTUAL output columns (post-collapse) for the
    /// list display / viz source; `input_columns` is what re-hydrates the form
    /// so the exact same calculation can be re-run. Empty on legacy saves.
    #[serde(default)]
    pub input_columns: Vec<String>,
    #[serde(default)]
    pub cross_sensor: bool,
}

pub(crate) fn aggregations_root() -> PathBuf {
    if let Some(appdata) = std::env::var_os("APPDATA") {
        return PathBuf::from(appdata).join("ttdprocess").join("aggregations");
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("aggregations")
}

fn persist_aggregation(
    name: &str,
    dataset: &str,
    period: &str,
    operation: &str,
    columns: &[String],
    result_json: &serde_json::Value,
    profile: Option<&str>,
    profile_by: Option<&str>,
    input_columns: &[String],
    cross_sensor: bool,
) -> anyhow::Result<SavedAggregationMeta> {
    let id = uuid::Uuid::new_v4().to_string();
    let dir = aggregations_root().join(&id);
    fs::create_dir_all(&dir)?;

    fs::write(
        dir.join("result.json"),
        serde_json::to_string(result_json)?,
    )?;

    let n_periods = result_json.get("total")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let meta = SavedAggregationMeta {
        id,
        name: name.to_string(),
        dataset: dataset.to_string(),
        period: period.to_string(),
        operation: operation.to_string(),
        columns: columns.to_vec(),
        n_periods,
        created_at: chrono::Utc::now().to_rfc3339(),
        profile: profile.map(String::from),
        profile_by: profile_by.map(String::from),
        input_columns: input_columns.to_vec(),
        cross_sensor,
    };
    fs::write(
        dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(meta)
}

/// Rename a saved aggregation in place. Updates only the `name` field of
/// `meta.json` — id / dataset / period / operation / columns / created_at
/// stay untouched so any downstream reference (scenario bundle, viz source)
/// remains valid.
#[tauri::command]
pub async fn rename_aggregation(aggregation_id: String, new_name: String) -> Result<SavedAggregationMeta, String> {
    let new_name = new_name.trim().to_string();
    if new_name.is_empty() {
        return Err("Le nom ne peut pas être vide.".to_string());
    }
    let dir = aggregations_root().join(&aggregation_id);
    let meta_path = dir.join("meta.json");
    if !meta_path.exists() {
        return Err(format!("Agrégation introuvable : {}", aggregation_id));
    }
    let mut meta: SavedAggregationMeta = serde_json::from_str(
        &fs::read_to_string(&meta_path).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    meta.name = new_name;
    fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    Ok(meta)
}

#[tauri::command]
pub async fn list_aggregations() -> Result<Vec<SavedAggregationMeta>, String> {
    list_aggregations_inner()
}

/// Same listing, callable from other Rust code. The command above is `async`
/// so Tauri runs it off the main thread (a synchronous command blocks the GTK
/// event loop, which costs us the Wayland connection); callers inside the crate
/// need the plain function.
pub fn list_aggregations_inner() -> Result<Vec<SavedAggregationMeta>, String> {
    let root = aggregations_root();
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    let entries = fs::read_dir(&root).map_err(|e| e.to_string())?;
    for entry in entries.flatten() {
        let p = entry.path().join("meta.json");
        if !p.exists() { continue; }
        if let Ok(s) = fs::read_to_string(&p) {
            if let Ok(m) = serde_json::from_str::<SavedAggregationMeta>(&s) {
                out.push(m);
            }
        }
    }
    // Newest first.
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

#[tauri::command]
pub async fn load_aggregation(aggregation_id: String) -> Result<serde_json::Value, String> {
    let dir = aggregations_root().join(&aggregation_id);
    let meta_path = dir.join("meta.json");
    let result_path = dir.join("result.json");
    if !meta_path.exists() || !result_path.exists() {
        return Err(format!("Agrégation introuvable : {}", aggregation_id));
    }
    let meta: SavedAggregationMeta = serde_json::from_str(
        &fs::read_to_string(&meta_path).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    let result: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&result_path).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "meta": meta, "result": result }))
}

/// Load a saved aggregation from disk and convert its rows into a DataFrame
/// for export. Returns `(name, df)` where the first column is "period"
/// (string) followed by one numeric column per aggregated series. Used by the
/// Export tab so aggregations can be written alongside the datasets.
pub(crate) fn aggregation_as_df(id: &str) -> anyhow::Result<(String, polars::prelude::DataFrame)> {
    use anyhow::Context;
    use polars::prelude::*;

    let dir = aggregations_root().join(id);
    let meta_path = dir.join("meta.json");
    let result_path = dir.join("result.json");
    if !meta_path.exists() || !result_path.exists() {
        anyhow::bail!("agrégation introuvable : {}", id);
    }
    let meta: SavedAggregationMeta = serde_json::from_str(&fs::read_to_string(&meta_path)?)?;
    let result: serde_json::Value = serde_json::from_str(&fs::read_to_string(&result_path)?)?;

    let cols: Vec<String> = result
        .get("columns")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let empty = Vec::new();
    let rows = result.get("rows").and_then(|v| v.as_array()).unwrap_or(&empty);

    // First column: the period / hour-of-day label.
    let periods: Vec<String> = rows
        .iter()
        .map(|r| match r.get("period") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        })
        .collect();
    let mut series: Vec<Series> = vec![Series::new("period", periods)];
    for col in &cols {
        let vals: Vec<Option<f64>> = rows
            .iter()
            .map(|r| r.get(col.as_str()).and_then(|v| v.as_f64()))
            .collect();
        series.push(Series::new(col.as_str(), vals));
    }
    let df = DataFrame::new(series).context("agrégation → DataFrame")?;
    Ok((meta.name, df))
}

#[tauri::command]
pub async fn delete_aggregation(
    state: State<'_, AppState>,
    aggregation_id: String,
) -> Result<(), String> {
    let dir = aggregations_root().join(&aggregation_id);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Info,
        format!("Agrégation supprimée : {}", aggregation_id),
    );
    Ok(())
}

#[cfg(test)]
mod bucket_rules_tests {
    use super::*;
    use polars::prelude::*;

    /// One day at a half-hourly step, `vals` in order from 00:00.
    fn frame(vals: Vec<Option<f64>>) -> DataFrame {
        let base = 1_704_067_200_000_000i64; // 2024-01-01T00:00:00Z, microseconds
        let ts: Vec<i64> = (0..vals.len() as i64).map(|i| base + i * 30 * 60_000_000).collect();
        let ts_s = Series::new("TIMESTAMP".into(), ts)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        DataFrame::new(vec![ts_s, Series::new("Fd_1".into(), vals)]).unwrap()
    }

    fn day_sum(df: &DataFrame, min_count: usize, scale: f64) -> serde_json::Value {
        let cols = vec!["Fd_1".to_string()];
        aggregate_inner(df, &cols, "Journalier", "Somme", None, None, "t", false, None, None, min_count, scale)
            .unwrap()
    }

    fn first_value(v: &serde_json::Value) -> serde_json::Value {
        v["rows"][0]["Fd_1"].clone()
    }

    #[test]
    fn without_a_minimum_a_partial_day_still_reports_its_partial_total() {
        let df = frame(vec![Some(1.0), Some(2.0), None, None]);
        assert_eq!(first_value(&day_sum(&df, 0, 1.0)), serde_json::json!(3.0));
    }

    #[test]
    fn a_day_below_the_minimum_reports_nothing_rather_than_a_fragment() {
        // Three valid steps where four are required: the total would look like
        // a real daily figure on the same axis as complete days.
        let df = frame(vec![Some(1.0), Some(2.0), Some(3.0), None]);
        assert_eq!(first_value(&day_sum(&df, 4, 1.0)), serde_json::Value::Null);
    }

    #[test]
    fn a_complete_day_passes_the_minimum() {
        let df = frame(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        assert_eq!(first_value(&day_sum(&df, 4, 1.0)), serde_json::json!(10.0));
    }

    #[test]
    fn scale_carries_the_unit_conversion() {
        // Σ(Fd × 0.5) — half-hourly l dm⁻² h⁻¹ summed into l dm⁻² j⁻¹.
        let df = frame(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        assert_eq!(first_value(&day_sum(&df, 0, 0.5)), serde_json::json!(5.0));
    }

    #[test]
    fn the_two_rules_compose() {
        let df = frame(vec![Some(1.0), Some(2.0), Some(3.0), None]);
        assert_eq!(first_value(&day_sum(&df, 4, 0.5)), serde_json::Value::Null);
        assert_eq!(first_value(&day_sum(&df, 3, 0.5)), serde_json::json!(3.0));
    }
}
