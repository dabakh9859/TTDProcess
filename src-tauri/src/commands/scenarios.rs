//! Scenarios: save a complete pipeline run (parameters + ALL result
//! DataFrames) so the user can re-load it later in one click — including the
//! pre-computed results — and immediately visualize them.
//!
//! Layout on disk (under %APPDATA%/ttdprocess/scenarios/):
//!   <uuid>/
//!     scenario.json         metadata (params, cleaning info, ResultsSummary)
//!     raw_data.parquet      (if loaded)
//!     cleaned_data.parquet  (if cleaning was applied)
//!     env_data.parquet      (if env data was loaded)
//!     tslope.parquet … sap_flow.parquet   (whichever pipeline outputs exist)

use std::path::{Path, PathBuf};
use std::fs;

use polars::prelude::*;
use tauri::State;

use crate::core::types::{LogLevel, ResultsSummary, Scenario};
use crate::state::AppState;
use crate::utils::logger;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Per-user scenarios directory. Lives next to `session_persist`'s data
/// under `%APPDATA%/ttdprocess/scenarios/`. Falls back to a folder next
/// to the executable if APPDATA isn't set (shouldn't happen on Windows).
fn scenarios_root() -> PathBuf {
    if let Some(appdata) = std::env::var_os("APPDATA") {
        return PathBuf::from(appdata).join("ttdprocess").join("scenarios");
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("scenarios")
}

fn scenario_dir(id: &str) -> PathBuf {
    scenarios_root().join(id)
}

// ---------------------------------------------------------------------------
// Parquet helpers (duplicated from session_persist — kept simple/self-contained)
// ---------------------------------------------------------------------------

fn write_parquet(df: &DataFrame, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut df_clone = df.clone();
    ParquetWriter::new(file).finish(&mut df_clone)?;
    Ok(())
}

fn read_parquet(path: &Path) -> anyhow::Result<DataFrame> {
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    Ok(df)
}

fn try_write(df: &Option<DataFrame>, path: &Path) {
    if let Some(df) = df {
        if let Err(e) = write_parquet(df, path) {
            eprintln!("scenarios: failed to write {}: {}", path.display(), e);
        }
    }
}

fn try_read(path: &Path) -> Option<DataFrame> {
    if !path.exists() {
        return None;
    }
    match read_parquet(path) {
        Ok(df) => Some(df),
        Err(e) => {
            eprintln!("scenarios: failed to read {}: {}", path.display(), e);
            None
        }
    }
}

/// Snapshot the currently-saved aggregation IDs (best-effort). Used so a
/// scenario records WHICH aggregations were on disk when it was saved — the
/// aggregations themselves live in their own root and are not duplicated
/// here (delete-an-aggregation stays the source of truth).
#[allow(dead_code)]
fn list_aggregation_ids_safely() -> Vec<String> {
    crate::commands::aggregation::list_aggregations()
        .map(|v| v.into_iter().map(|m| m.id).collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Snapshot summary from the live state
// ---------------------------------------------------------------------------

fn compute_summary(app: &crate::state::AppData) -> ResultsSummary {
    // total_rows = whichever active source has the largest height (raw → cleaned
    // → sap_flow). num_columns same. Date range from the first dataset that has
    // a TIMESTAMP column. mean_sap_flow = mean over Fd_* columns.
    let pick = app
        .results
        .sap_flow
        .as_ref()
        .or(app.cleaned_data.as_ref())
        .or(app.raw_data.as_ref());
    let total_rows = pick.map(|df| df.height()).unwrap_or(0);
    let num_columns = pick.map(|df| df.width()).unwrap_or(0);

    let date_range = pick
        .and_then(|df| df.column("TIMESTAMP").ok())
        .and_then(|col| {
            crate::core::timestamp_utils::ts_col_to_datetimes(col).ok()
        })
        .and_then(|dts| {
            let min = dts.iter().min().copied()?;
            let max = dts.iter().max().copied()?;
            Some((min.to_rfc3339(), max.to_rfc3339()))
        })
        .unwrap_or_else(|| (String::new(), String::new()));

    let mean_sap_flow = app
        .results
        .sap_flow
        .as_ref()
        .map(|df| {
            let mut sum = 0.0_f64;
            let mut n = 0_u64;
            for c in df.get_column_names() {
                if c.starts_with("Fd_") {
                    if let Ok(s) = df.column(c).and_then(|s| s.f64().cloned()) {
                        for v in s.into_iter().flatten() {
                            if v.is_finite() {
                                sum += v;
                                n += 1;
                            }
                        }
                    }
                }
            }
            if n > 0 { sum / n as f64 } else { 0.0 }
        })
        .unwrap_or(0.0);

    ResultsSummary {
        total_rows,
        num_columns,
        date_range,
        mean_sap_flow,
    }
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn save_scenario(
    state: State<'_, AppState>,
    name: String,
    description: String,
) -> Result<serde_json::Value, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;

    let data_source = app
        .file_path
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    // Reproducibility bundle: capture the current cleaning config (path +
    // method + predictor list + training metrics) so reloading the scenario
    // restores the UI setup too.
    let (cleaning_path, cleaning_method, ml_metrics, predictor_columns) =
        if let Some(cleaner) = app.ml_cleaner.as_ref() {
            use crate::core::types::CleaningPath;
            (
                Some(CleaningPath::AIBased),
                Some(cleaner.method().clone()),
                cleaner.metrics().cloned(),
                cleaner.predictor_columns().to_vec(),
            )
        } else {
            (None, None, None, Vec::new())
        };

    let id = uuid::Uuid::new_v4().to_string();
    let dir = scenario_dir(&id);
    fs::create_dir_all(&dir).map_err(|e| format!("create scenario dir: {}", e))?;

    // Persist all DataFrames the user may want to revisit. Each is best-effort;
    // a missing slot just means that part of the pipeline didn't run.
    try_write(&app.raw_data,     &dir.join("raw_data.parquet"));
    try_write(&app.cleaned_data, &dir.join("cleaned_data.parquet"));
    try_write(&app.env_data,     &dir.join("env_data.parquet"));
    try_write(&app.results.tslope,            &dir.join("tslope.parquet"));
    try_write(&app.results.baseline,          &dir.join("baseline.parquet"));
    try_write(&app.results.delta_t,           &dir.join("delta_t.parquet"));
    try_write(&app.results.t600,              &dir.join("t600.parquet"));
    try_write(&app.results.tm,                &dir.join("tm.parquet"));
    try_write(&app.results.stm,               &dir.join("stm.parquet"));
    try_write(&app.results.tmi,               &dir.join("tmi.parquet"));
    try_write(&app.results.k,                 &dir.join("k.parquet"));
    try_write(&app.results.sap_flow,          &dir.join("sap_flow.parquet"));
    try_write(&app.results.ttdplus_fourier,   &dir.join("ttdplus_fourier.parquet"));
    try_write(&app.results.ttdplus_refs,      &dir.join("ttdplus_refs.parquet"));
    try_write(&app.results.ttdplus_sap_flow,  &dir.join("ttdplus_sap_flow.parquet"));
    // RegressionDiurne diagnostic tables
    try_write(&app.results.rd_regression,     &dir.join("rd_regression.parquet"));
    try_write(&app.results.rd_result,         &dir.join("rd_result.parquet"));
    // Calculs avancés — Jh → Jhp → Qh → Qd
    try_write(&app.results.jh,                &dir.join("jh.parquet"));
    try_write(&app.results.jhp,               &dir.join("jhp.parquet"));
    try_write(&app.results.qh,                &dir.join("qh.parquet"));
    try_write(&app.results.qd,                &dir.join("qd.parquet"));

    // Cleaning pre-snapshots (before/after comparisons surfaced on the Gap
    // Filling tab). Copied as parquet files so the scenario carries the
    // full before/after material with it.
    let snap_target = dir.join("cleaning_pre_snapshots");
    let _ = fs::create_dir_all(&snap_target);
    for (key, df) in &app.cleaning_pre_snapshots {
        let path = snap_target.join(format!("{}.parquet", key));
        let mut cloned = df.clone();
        let _ = std::fs::File::create(&path)
            .map_err(anyhow::Error::from)
            .and_then(|f| ParquetWriter::new(f).finish(&mut cloned).map_err(anyhow::Error::from));
    }

    // Bundle every saved aggregation INTO the scenario directory so the
    // scenario is self-contained. Even if the user deletes an aggregation
    // later, reloading the scenario re-creates it.
    let agg_metas = crate::commands::aggregation::list_aggregations().unwrap_or_default();
    let agg_target = dir.join("aggregations");
    let _ = fs::create_dir_all(&agg_target);
    let mut bundled_agg_ids: Vec<String> = Vec::with_capacity(agg_metas.len());
    for meta in &agg_metas {
        let src_dir = crate::commands::aggregation::aggregations_root().join(&meta.id);
        if !src_dir.exists() { continue; }
        let dst_dir = agg_target.join(&meta.id);
        if fs::create_dir_all(&dst_dir).is_err() { continue; }
        let ok_meta = fs::copy(src_dir.join("meta.json"), dst_dir.join("meta.json")).is_ok();
        let ok_res = fs::copy(src_dir.join("result.json"), dst_dir.join("result.json")).is_ok();
        if ok_meta && ok_res {
            bundled_agg_ids.push(meta.id.clone());
        }
    }

    // Diurnal diagnostics: HashMap<(String, String), DiurnalDiagnostic>.
    // JSON requires string keys, so flatten the tuple key into Vec<(a, b, diag)>.
    let diurnal_diag_serialized = app.results.diurnal_diagnostics.as_ref().map(|m| {
        m.iter()
            .map(|((a, b), d)| (a.clone(), b.clone(), d.clone()))
            .collect::<Vec<_>>()
    });

    // The scenario's complete bundle. Anything not covered by the
    // top-level Scenario struct goes here so old scenarios still
    // deserialise (serde(default)) and new fields don't break compat.
    let extras = serde_json::json!({
        "advanced_groups": app.advanced_groups,
        "aggregation_ids": bundled_agg_ids,
        "sheet_name": app.sheet_name,
        "env_file_path": app.env_file_path,
        "env_sheet_name": app.env_sheet_name,
        "project_config": app.config,
        "ttdplus_params": app.ttdplus_params,
        "cleaning_target_columns": app.cleaning_target_columns,
        "vpd_par_stats": app.results.vpd_par_stats,
        "diurnal_diagnostics": diurnal_diag_serialized,
    });
    if let Ok(s) = serde_json::to_string_pretty(&extras) {
        let _ = fs::write(dir.join("extras.json"), s);
    }

    let summary = compute_summary(&app);

    let scenario = Scenario {
        id: id.clone(),
        name: name.clone(),
        description,
        created_at: chrono::Utc::now().to_rfc3339(),
        data_source,
        tm_method: app.sap_flow_params.tm_method.clone(),
        sap_flow_params: app.sap_flow_params.clone(),
        results_summary: summary,
        cleaning_path,
        cleaning_stage: None,
        cleaning_method,
        cleaning_report: None,
        ml_metrics,
        model_path: None,
        predictor_columns,
    };

    // Write the metadata JSON last so a partially-written scenario can be
    // detected (missing scenario.json → ignored by list).
    let json_path = dir.join("scenario.json");
    let json = serde_json::to_string_pretty(&scenario).map_err(|e| e.to_string())?;
    fs::write(&json_path, json).map_err(|e| format!("write scenario.json: {}", e))?;

    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Scénario '{}' enregistré (données + paramètres)", name),
    );

    serde_json::to_value(&scenario).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn load_scenario(
    state: State<'_, AppState>,
    scenario_id: String,
) -> Result<serde_json::Value, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;

    let dir = scenario_dir(&scenario_id);
    let json_path = dir.join("scenario.json");
    if !json_path.exists() {
        return Err(format!("Scénario introuvable : {}", json_path.display()));
    }
    let content = fs::read_to_string(&json_path).map_err(|e| format!("read scenario.json: {}", e))?;
    let scenario: Scenario = serde_json::from_str(&content).map_err(|e| format!("parse scenario.json: {}", e))?;

    // Restore parameters into state.
    app.sap_flow_params = scenario.sap_flow_params.clone();
    // Diagnostics (diurnal_diagnostics, vpd_par_stats) are restored further
    // down from extras.json — don't blank them here.

    // Restore data: every slot that has a parquet file overrides the live state.
    app.raw_data     = try_read(&dir.join("raw_data.parquet")).or(app.raw_data.take());
    app.cleaned_data = try_read(&dir.join("cleaned_data.parquet")).or(app.cleaned_data.take());
    app.env_data     = try_read(&dir.join("env_data.parquet")).or(app.env_data.take());
    app.results.tslope             = try_read(&dir.join("tslope.parquet")).or(app.results.tslope.take());
    app.results.baseline           = try_read(&dir.join("baseline.parquet")).or(app.results.baseline.take());
    app.results.delta_t            = try_read(&dir.join("delta_t.parquet")).or(app.results.delta_t.take());
    app.results.t600               = try_read(&dir.join("t600.parquet")).or(app.results.t600.take());
    app.results.tm                 = try_read(&dir.join("tm.parquet")).or(app.results.tm.take());
    app.results.stm                = try_read(&dir.join("stm.parquet")).or(app.results.stm.take());
    app.results.tmi                = try_read(&dir.join("tmi.parquet")).or(app.results.tmi.take());
    app.results.k                  = try_read(&dir.join("k.parquet")).or(app.results.k.take());
    app.results.sap_flow           = try_read(&dir.join("sap_flow.parquet")).or(app.results.sap_flow.take());
    app.results.ttdplus_fourier    = try_read(&dir.join("ttdplus_fourier.parquet")).or(app.results.ttdplus_fourier.take());
    app.results.ttdplus_refs       = try_read(&dir.join("ttdplus_refs.parquet")).or(app.results.ttdplus_refs.take());
    app.results.ttdplus_sap_flow   = try_read(&dir.join("ttdplus_sap_flow.parquet")).or(app.results.ttdplus_sap_flow.take());
    // RegressionDiurne diagnostic tables
    app.results.rd_regression      = try_read(&dir.join("rd_regression.parquet")).or(app.results.rd_regression.take());
    app.results.rd_result          = try_read(&dir.join("rd_result.parquet")).or(app.results.rd_result.take());
    // Calculs avancés
    app.results.jh                 = try_read(&dir.join("jh.parquet")).or(app.results.jh.take());
    app.results.jhp                = try_read(&dir.join("jhp.parquet")).or(app.results.jhp.take());
    app.results.qh                 = try_read(&dir.join("qh.parquet")).or(app.results.qh.take());
    app.results.qd                 = try_read(&dir.join("qd.parquet")).or(app.results.qd.take());

    // Restore the rest of the snapshot from extras.json. Every field is
    // optional / best-effort so old scenarios (which only carry a subset of
    // the fields, or none of them) still load without losing previous
    // settings.
    let extras_path = dir.join("extras.json");
    if extras_path.exists() {
        if let Ok(s) = fs::read_to_string(&extras_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(groups) = v.get("advanced_groups") {
                    if let Ok(parsed) = serde_json::from_value::<Vec<crate::core::types::AdvancedGroup>>(groups.clone()) {
                        app.advanced_groups = parsed;
                    }
                }
                if let Some(sheet) = v.get("sheet_name").and_then(|x| x.as_str()) {
                    app.sheet_name = Some(sheet.to_string());
                }
                if let Some(env_path) = v.get("env_file_path").and_then(|x| x.as_str()) {
                    app.env_file_path = Some(env_path.to_string());
                }
                if let Some(env_sheet) = v.get("env_sheet_name").and_then(|x| x.as_str()) {
                    app.env_sheet_name = Some(env_sheet.to_string());
                }
                if let Some(cfg) = v.get("project_config") {
                    if let Ok(parsed) = serde_json::from_value::<crate::core::types::ProjectConfig>(cfg.clone()) {
                        app.config = parsed;
                    }
                }
                if let Some(tp) = v.get("ttdplus_params") {
                    if let Ok(parsed) = serde_json::from_value::<crate::core::types::TtdPlusParams>(tp.clone()) {
                        app.ttdplus_params = parsed;
                    }
                }
                if let Some(cols) = v.get("cleaning_target_columns") {
                    if let Ok(parsed) = serde_json::from_value::<Vec<String>>(cols.clone()) {
                        app.cleaning_target_columns = parsed;
                    }
                }
                if let Some(vps) = v.get("vpd_par_stats") {
                    if !vps.is_null() {
                        if let Ok(parsed) = serde_json::from_value::<crate::core::types::VpdParStats>(vps.clone()) {
                            app.results.vpd_par_stats = Some(parsed);
                        }
                    }
                }
                // Diurnal diagnostics: serialised as Vec<(date, col, diag)>,
                // unflatten back to the HashMap form the rest of the code
                // expects.
                if let Some(diag_v) = v.get("diurnal_diagnostics") {
                    if !diag_v.is_null() {
                        if let Ok(flat) = serde_json::from_value::<Vec<(String, String, crate::core::types::DiurnalDiagnostic)>>(diag_v.clone()) {
                            let mut m = std::collections::HashMap::with_capacity(flat.len());
                            for (a, b, d) in flat { m.insert((a, b), d); }
                            app.results.diurnal_diagnostics = Some(m);
                        }
                    }
                }
            }
        }
    }

    // Restore the cleaning pre-snapshots (before/after for each touched
    // dataset) — overrides whatever was in state.
    let snap_src = dir.join("cleaning_pre_snapshots");
    if snap_src.exists() {
        if let Ok(entries) = fs::read_dir(&snap_src) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) != Some("parquet") { continue; }
                let key = match p.file_stem().and_then(|s| s.to_str()) { Some(k) => k.to_string(), None => continue };
                if let Some(df) = try_read(&p) {
                    app.cleaning_pre_snapshots.insert(key, df);
                }
            }
        }
    }

    // Restore the saved aggregations bundled with this scenario — merge into
    // the global aggregations root (same-id dirs overwrite, no duplicates).
    let agg_src = dir.join("aggregations");
    if agg_src.exists() {
        let global_root = crate::commands::aggregation::aggregations_root();
        let _ = fs::create_dir_all(&global_root);
        if let Ok(entries) = fs::read_dir(&agg_src) {
            for entry in entries.flatten() {
                let p = entry.path();
                if !p.is_dir() { continue; }
                let id = match p.file_name().and_then(|s| s.to_str()) { Some(k) => k.to_string(), None => continue };
                let dst = global_root.join(&id);
                let _ = fs::create_dir_all(&dst);
                let _ = fs::copy(p.join("meta.json"), dst.join("meta.json"));
                let _ = fs::copy(p.join("result.json"), dst.join("result.json"));
            }
        }
    }

    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Scénario '{}' importé (données + résultats)", scenario.name),
    );

    serde_json::to_value(&scenario).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_scenarios(_state: State<'_, AppState>) -> Result<Vec<Scenario>, String> {
    let root = scenarios_root();
    let mut scenarios = Vec::new();
    if !root.exists() {
        return Ok(scenarios);
    }
    let entries = match fs::read_dir(&root) {
        Ok(it) => it,
        Err(e) => return Err(format!("read scenarios dir: {}", e)),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let json_path = path.join("scenario.json");
        if !json_path.exists() {
            // Backward-compat: older flat layout used <uuid>.json at the root.
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(scenario) = serde_json::from_str::<Scenario>(&content) {
                        scenarios.push(scenario);
                    }
                }
            }
            continue;
        }
        if let Ok(content) = fs::read_to_string(&json_path) {
            if let Ok(scenario) = serde_json::from_str::<Scenario>(&content) {
                scenarios.push(scenario);
            }
        }
    }
    scenarios.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(scenarios)
}

#[tauri::command]
pub fn delete_scenario(
    state: State<'_, AppState>,
    scenario_id: String,
) -> Result<String, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    let dir = scenario_dir(&scenario_id);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| format!("remove scenario dir: {}", e))?;
    } else {
        // Legacy flat layout
        let legacy = scenarios_root().join(format!("{}.json", scenario_id));
        if legacy.exists() {
            let _ = fs::remove_file(legacy);
        }
    }
    logger::add_log(
        &mut app.logs,
        LogLevel::Info,
        format!("Scénario supprimé : {}", scenario_id),
    );
    Ok("Scenario deleted".to_string())
}

#[tauri::command]
pub fn list_scenario_datasets(scenario_id: String) -> Result<Vec<serde_json::Value>, String> {
    let dir = scenario_dir(&scenario_id);
    if !dir.exists() {
        return Err(format!("Scénario introuvable : {}", scenario_id));
    }
    let known: &[(&str, &str)] = &[
        ("t600", "T600"),
        ("delta_t", "Delta-T"),
        ("tslope", "Tslope"),
        ("tm", "T0"),
        ("stm", "sTm"),
        ("tmi", "Tmi"),
        ("k", "K"),
        ("sap_flow", "Flux de sève"),
        ("raw_data", "Données brutes"),
        ("cleaned_data", "Données nettoyées"),
    ];
    let mut out = Vec::new();
    for &(key, label) in known {
        let pq = dir.join(format!("{}.parquet", key));
        if pq.exists() {
            out.push(serde_json::json!({
                "key": key,
                "label": label,
                "path": pq.to_string_lossy(),
            }));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Visualisation overlays — load a scenario WITHOUT replacing the live state,
// expose its DataFrames as extra sources on the Viz tab.
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
pub struct ScenarioOverlaySummary {
    pub id: String,
    pub name: String,
    pub created_at: String,
    /// Which datasets the overlay actually carries (a subset of the standard
    /// dataset keys). Lets the frontend know how many sources to expect.
    pub available_keys: Vec<String>,
}

fn summarize_overlay(o: &crate::core::types::ScenarioOverlay) -> ScenarioOverlaySummary {
    let keys = [
        "raw","cleaned","env","tslope","baseline","delta_t","t600","tm","stm",
        "tmi","k","sap_flow","ttdplus_fourier","ttdplus_refs","ttdplus_sap_flow",
        "rd_regression","rd_result","jh","jhp","qh","qd",
    ];
    let available_keys = keys.iter()
        .filter(|k| o.dataset(k).is_some())
        .map(|k| k.to_string())
        .collect();
    ScenarioOverlaySummary {
        id: o.id.clone(),
        name: o.name.clone(),
        created_at: o.created_at.clone(),
        available_keys,
    }
}

/// Load a scenario's parquet files into an in-memory overlay so the
/// Visualisation tab can list them as extra sources. Does NOT touch the
/// live raw/cleaned/results state — overlays sit alongside, never replace.
#[tauri::command]
pub fn viz_overlay_load(
    state: State<'_, AppState>,
    scenario_id: String,
) -> Result<ScenarioOverlaySummary, String> {
    let dir = scenario_dir(&scenario_id);
    let json_path = dir.join("scenario.json");
    if !json_path.exists() {
        return Err(format!("Scénario introuvable : {}", json_path.display()));
    }
    let content = fs::read_to_string(&json_path).map_err(|e| format!("read scenario.json: {}", e))?;
    let scenario: Scenario = serde_json::from_str(&content).map_err(|e| format!("parse scenario.json: {}", e))?;

    let overlay = crate::core::types::ScenarioOverlay {
        id: scenario.id.clone(),
        name: scenario.name.clone(),
        created_at: scenario.created_at.clone(),
        raw_data:         try_read(&dir.join("raw_data.parquet")),
        cleaned_data:     try_read(&dir.join("cleaned_data.parquet")),
        env_data:         try_read(&dir.join("env_data.parquet")),
        tslope:           try_read(&dir.join("tslope.parquet")),
        baseline:         try_read(&dir.join("baseline.parquet")),
        delta_t:          try_read(&dir.join("delta_t.parquet")),
        t600:             try_read(&dir.join("t600.parquet")),
        tm:               try_read(&dir.join("tm.parquet")),
        stm:              try_read(&dir.join("stm.parquet")),
        tmi:              try_read(&dir.join("tmi.parquet")),
        k:                try_read(&dir.join("k.parquet")),
        sap_flow:         try_read(&dir.join("sap_flow.parquet")),
        ttdplus_fourier:  try_read(&dir.join("ttdplus_fourier.parquet")),
        ttdplus_refs:     try_read(&dir.join("ttdplus_refs.parquet")),
        ttdplus_sap_flow: try_read(&dir.join("ttdplus_sap_flow.parquet")),
        rd_regression:    try_read(&dir.join("rd_regression.parquet")),
        rd_result:        try_read(&dir.join("rd_result.parquet")),
        jh:               try_read(&dir.join("jh.parquet")),
        jhp:              try_read(&dir.join("jhp.parquet")),
        qh:               try_read(&dir.join("qh.parquet")),
        qd:               try_read(&dir.join("qd.parquet")),
    };
    let summary = summarize_overlay(&overlay);

    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    app.loaded_scenario_overlays.insert(scenario.id.clone(), overlay);
    logger::add_log(
        &mut app.logs,
        LogLevel::Info,
        format!("Overlay scénario chargé : {} ({} datasets)", scenario.name, summary.available_keys.len()),
    );
    Ok(summary)
}

#[tauri::command]
pub fn viz_overlay_remove(
    state: State<'_, AppState>,
    scenario_id: String,
) -> Result<(), String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    if let Some(o) = app.loaded_scenario_overlays.remove(&scenario_id) {
        logger::add_log(
            &mut app.logs,
            LogLevel::Info,
            format!("Overlay scénario retiré : {}", o.name),
        );
    }
    Ok(())
}

/// Lists every overlay currently loaded. Sorted by scenario name for a
/// stable UI order.
#[tauri::command]
pub fn viz_overlay_list(
    state: State<'_, AppState>,
) -> Result<Vec<ScenarioOverlaySummary>, String> {
    let app = state.inner.lock().map_err(|e| e.to_string())?;
    let mut out: Vec<ScenarioOverlaySummary> = app.loaded_scenario_overlays
        .values()
        .map(summarize_overlay)
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}
