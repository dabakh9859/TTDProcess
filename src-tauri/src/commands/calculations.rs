use serde::Deserialize;
use tauri::State;

use crate::core::calculations;
use crate::core::timestamp_utils;
use crate::core::ttdplus;
use crate::core::types::{CalculationResults, LogLevel, T0SmoothConfig, TmMethod, TtdPlusParams, VpdParConfig};
use crate::state::AppState;
use crate::utils::logger;

#[derive(Debug, Deserialize)]
pub struct SapFlowParamsInput {
    pub alpha: f64,
    pub beta: f64,
    pub tm_method: String,
    pub nb_max_points: Option<usize>,
    /// Granier rolling-baseline window expressed in DAYS (typical values
    /// 5–10, default 7). Replaces the legacy `window_hours` parameter.
    pub window_days: Option<usize>,
    /// Nocturnal window for the per-day ΔT_max extraction in the rolling
    /// method. Defaults: 20h start, 8h end (wraps midnight).
    pub night_start_hour: Option<u32>,
    pub night_end_hour: Option<u32>,
    pub min_points: Option<usize>,
    /// Full VpdPar config (required when tm_method is "VpdPar" / "DataEnv" /
    /// "DonneesEnv"). Frontend-side validation should run first; backend
    /// performs a second validation via `VpdParConfig::validate()` at Tm time.
    pub vpd_par_config: Option<VpdParConfig>,
    /// ETo column name for RegressionDiurne method
    pub etp_column: Option<String>,
    /// Optional manual selection range on X = ETo^(1/β). When either bound
    /// is set (or hour_min/hour_max below), only points inside the range
    /// feed the regression. With every bound None, ALL diurnal points
    /// (6h–20h window) feed the regression.
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    /// Optional manual diurnal window in clock hours. Default 6h–20h.
    pub hour_min: Option<u32>,
    pub hour_max: Option<u32>,
    /// Enable Pasqualotto-style two-pass robust fit (lower envelope re-fit
    /// in the diurnal context). Default false.
    pub double_regression: Option<bool>,
    /// Smoothing config for the final T0 — all methods.
    pub t0_smooth: Option<T0SmoothConfig>,
}

/// Update sap flow parameters in the backend state before running the pipeline.
#[tauri::command]
pub fn set_sap_flow_params(
    state: State<'_, AppState>,
    params: SapFlowParamsInput,
) -> Result<(), String> {
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    data.sap_flow_params.alpha = params.alpha;
    data.sap_flow_params.beta = params.beta;
    data.sap_flow_params.tm_method = match params.tm_method.as_str() {
        "PreAube" => TmMethod::PreAube {
            nb_max_points: params.nb_max_points.unwrap_or(3),
        },
        "FenetreGlissante" => TmMethod::FenetreGlissante {
            window_days: params.window_days.unwrap_or(7).clamp(1, 30),
            night_start_hour: params.night_start_hour.unwrap_or(20).min(23),
            night_end_hour: params.night_end_hour.unwrap_or(8).min(23),
        },
        "DoubleRegression" => TmMethod::DoubleRegression {
            min_points: params.min_points.unwrap_or(3),
        },
        // Accept the old alias "DonneesEnv" as well as new "VpdPar"/"DataEnv"
        // so the frontend can send whichever string it wants.
        "DonneesEnv" | "VpdPar" | "DataEnv" => {
            let config = params.vpd_par_config.unwrap_or_default();
            // Validate up-front so the UI gets fast feedback.
            config.validate()?;
            TmMethod::VpdPar { config }
        }
        "RegressionDiurne" => TmMethod::RegressionDiurne {
            alpha: params.alpha,
            beta: params.beta,
            etp_column: params.etp_column.unwrap_or_else(|| "ETo".to_string()),
            x_min: params.x_min,
            x_max: params.x_max,
            hour_min: params.hour_min,
            hour_max: params.hour_max,
            double_regression: params.double_regression.unwrap_or(false),
        },
        _ => TmMethod::PreAube { nb_max_points: 3 },
    };
    data.sap_flow_params.t0_smooth = params.t0_smooth.unwrap_or_default();
    Ok(())
}

/// Run the full 9-step calculation pipeline (async, runs on blocking thread pool).
///
/// A full run rebuilds every derived slot from raw, so any cleaning the user
/// "locked" (Rendre permanent) on a derived stage would be wiped. When such a
/// lock exists and `force` is not `true`, the run is REFUSED with a
/// machine-parseable `LOCKED_STAGES:<csv>` error so the frontend can confirm
/// before re-invoking with `force = true`.
#[tauri::command]
pub async fn run_pipeline(
    state: State<'_, AppState>,
    force: Option<bool>,
) -> Result<serde_json::Value, String> {
    // 0. Refuse to blow away locked cleanings unless explicitly forced.
    {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        if !force.unwrap_or(false) && !data.cleaning_locked_stages.is_empty() {
            return Err(format!(
                "LOCKED_STAGES:{}",
                data.cleaning_locked_stages.join(",")
            ));
        }
    }

    // 1. Clone data out of the lock — release it BEFORE heavy work
    let (input_df, env_df, pattern, tm_method, alpha, beta, t0_smooth) = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let raw_df = data
            .raw_data
            .as_ref()
            .ok_or_else(|| "No data loaded. Import data first.".to_string())?
            .clone();
        let input = data.cleaned_data.as_ref().unwrap_or(&raw_df).clone();
        // env_data is cloned only for methods that actually need it — VpdPar
        // (VPD/PAR conditions to flag no-flow periods) and RegressionDiurne
        // (ETo column for the diurnal regression). Other methods get `None`
        // to avoid a pointless DataFrame clone.
        let env = match &data.sap_flow_params.tm_method {
            TmMethod::VpdPar { .. } | TmMethod::RegressionDiurne { .. } => data.env_data.clone(),
            _ => None,
        };
        (
            input,
            env,
            data.config.pattern.clone(),
            data.sap_flow_params.tm_method.clone(),
            data.sap_flow_params.alpha,
            data.sap_flow_params.beta,
            data.sap_flow_params.t0_smooth.clone(),
        )
    };

    // Log start (separate brief lock)
    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        logger::add_log(
            &mut data.logs,
            LogLevel::Info,
            "Starting 9-step TTD pipeline...".to_string(),
        );
    }

    // 2. Heavy work on a blocking thread — main thread stays free for UI
    let (results, completed, errors, step_logs) = tokio::task::spawn_blocking(move || {
        let mut results = CalculationResults::default();
        let mut completed: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let mut step_logs: Vec<(LogLevel, String)> = Vec::new();

        macro_rules! run_step {
            ($name:expr, $expr:expr, $target:expr) => {
                match $expr {
                    Ok(df) => {
                        $target = Some(df);
                        completed.push($name.to_string());
                        step_logs.push((LogLevel::Info, format!("Step {} OK", $name)));
                    }
                    Err(e) => {
                        errors.push(format!("{}: {}", $name, e));
                        step_logs.push((LogLevel::Warning, format!("Step {} failed: {}", $name, e)));
                    }
                }
            };
        }

        // Pre-step — drop rows with a null/blank TIMESTAMP. Empty export rows
        // have no time; ts_col_to_datetimes drops them when building the time
        // index, which misaligns every step that maps a compacted datetime
        // index back onto the full frame (T600 :00/:30 filter, Tm grouping) →
        // zeroed/garbage outputs. Removing them keeps row index == time index.
        let input_df = match timestamp_utils::drop_null_timestamp_rows(&input_df) {
            Ok((df, n)) => {
                if n > 0 {
                    step_logs.push((LogLevel::Info, format!(
                        "Pré-traitement: {} lignes sans horodatage retirées", n
                    )));
                }
                df
            }
            Err(e) => {
                step_logs.push((LogLevel::Warning, format!(
                    "Filtrage des horodatages nuls ignoré ({}): pipeline continue", e
                )));
                input_df
            }
        };

        // Pre-step — drop intra-cycle extras: legacy +10 s (2019), duplicate
        // timestamps (0 s, since ~2024-06), and backwards timestamps (< 0 s).
        let input_df = match timestamp_utils::drop_intra_cycle_extras(&input_df) {
            Ok((df, n)) => {
                if n > 0 {
                    step_logs.push((LogLevel::Info, format!(
                        "Pré-traitement: {} lignes parasites retirées (doublons, inversions, extras legacy)",
                        n
                    )));
                }
                df
            }
            Err(e) => {
                step_logs.push((LogLevel::Warning, format!(
                    "Pré-traitement ignoré ({}): pipeline continue sur les données brutes", e
                )));
                input_df
            }
        };

        // Pre-step — infer the heating-cycle pattern from the actual data and
        // override the configured one if they don't match. See
        // timestamp_utils::infer_heating_pattern. Misalignment between the
        // configured 11-delta pattern and a 7-delta file produces zigzag T600
        // because baseline accumulation uses the wrong delta on alternating
        // rows.
        let effective_pattern = match timestamp_utils::infer_heating_pattern(&input_df) {
            Ok(Some(inferred)) if inferred != pattern => {
                step_logs.push((LogLevel::Warning, format!(
                    "Pattern de chauffe auto-détecté: {:?} (différent de la config {:?}) — utilisation du pattern détecté",
                    inferred, pattern
                )));
                inferred
            }
            Ok(Some(_)) => {
                step_logs.push((LogLevel::Info,
                    "Pattern de chauffe auto-détecté: identique à la config".to_string()));
                pattern.clone()
            }
            Ok(None) => {
                step_logs.push((LogLevel::Info,
                    "Pattern de chauffe non détectable (données trop courtes ou irrégulières) — pattern config utilisé".to_string()));
                pattern.clone()
            }
            Err(e) => {
                step_logs.push((LogLevel::Warning, format!(
                    "Détection du pattern échouée ({}): pattern config utilisé", e
                )));
                pattern.clone()
            }
        };

        // Step 1 – Tslope
        run_step!("Tslope", calculations::calculate_tslope(&input_df, &effective_pattern), results.tslope);

        // Step 2 – Baseline
        if let Some(ref tslope) = results.tslope {
            run_step!("Baseline", calculations::calculate_baseline(&input_df, tslope, &effective_pattern), results.baseline);
        }

        // Step 3 – DeltaT
        if let Some(ref baseline) = results.baseline {
            run_step!("DeltaT", calculations::calculate_delta_t(&input_df, baseline), results.delta_t);
        }

        // Step 4 – T600
        if let Some(ref delta_t) = results.delta_t {
            run_step!("T600", calculations::create_t600(delta_t), results.t600);
        }

        // Step 5 – Tm (also captures RegressionDiurne/VpdPar diagnostics)
        if let Some(ref t600) = results.t600 {
            match calculations::calculate_tm(t600, &tm_method, env_df.as_ref()) {
                Ok(out) => {
                    if let Some(stats) = &out.vpd_par_stats {
                        let total = stats.n_total.max(1) as f64;
                        step_logs.push((LogLevel::Info, format!(
                            "VpdPar diagnostic: total={} | nuit-horloge={} ({:.0}%) | rayonnement OK={} ({:.0}%) | nuit+rayonnement={} ({:.0}%) | VPD OK={} ({:.0}%) | env-passé={} ({:.0}%) | nuits Valid={}, Interp={}, NoValid={}",
                            stats.n_total,
                            stats.n_in_clock_window, 100.0 * stats.n_in_clock_window as f64 / total,
                            stats.n_rad_ok, 100.0 * stats.n_rad_ok as f64 / total,
                            stats.n_night_flag, 100.0 * stats.n_night_flag as f64 / total,
                            stats.n_vpd_ok, 100.0 * stats.n_vpd_ok as f64 / total,
                            stats.n_env_passed, 100.0 * stats.n_env_passed as f64 / total,
                            stats.n_valid_nights, stats.n_interpolated_nights, stats.n_no_valid_nights,
                        )));
                    }
                    // Optional smoothing of the final T0 (all methods, several modalities).
                    results.tm = Some(calculations::smooth_tm(&out.df, &t0_smooth));
                    results.diurnal_diagnostics = out.diurnal_diagnostics;
                    results.vpd_par_stats = out.vpd_par_stats;
                    if let Some(steps) = out.diurnal_steps {
                        results.rd_regression = Some(steps.regression);
                        results.rd_result = Some(steps.result);
                    }
                    completed.push("T0".to_string());
                    step_logs.push((LogLevel::Info, "Step T0 OK".to_string()));
                }
                Err(e) => {
                    errors.push(format!("T0: {}", e));
                    step_logs.push((LogLevel::Warning, format!("Step T0 failed: {}", e)));
                }
            }
        }

        // Step 6 – sTm
        if let (Some(ref tm), Some(ref t600)) = (&results.tm, &results.t600) {
            run_step!("sT0", calculations::calculate_stm(tm, t600), results.stm);
        }

        // Step 7 – Tmi
        if let (Some(ref tm), Some(ref stm), Some(ref t600)) =
            (&results.tm, &results.stm, &results.t600)
        {
            run_step!("T0i", calculations::calculate_tmi(tm, stm, t600), results.tmi);
        }

        // Step 8 – K
        if let (Some(ref tmi), Some(ref t600)) = (&results.tmi, &results.t600) {
            run_step!("K", calculations::calculate_k(tmi, t600), results.k);
        }

        // Step 9 – Sap Flow
        if let Some(ref k) = results.k {
            run_step!("SapFlow", calculations::calculate_sapflow(k, alpha, beta), results.sap_flow);
        }

        (results, completed, errors, step_logs)
    })
    .await
    .map_err(|e| e.to_string())?;

    // 3. Store results back in state (brief lock)
    let msg = format!(
        "Pipeline complete: {} steps done, {} errors",
        completed.len(),
        errors.len()
    );

    // Surface any VpdPar diagnostic line in the response so the frontend can
    // route it to its own Journal. (Rust's `data.logs` is NOT synced to the
    // Zustand store the UI reads, so logger::add_log alone is invisible.)
    let vpd_par_diagnostic: Option<String> = step_logs
        .iter()
        .find(|(_, m)| m.starts_with("VpdPar diagnostic:"))
        .map(|(_, m)| m.clone());
    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        // Preserve TTD+ results when overwriting
        let ttdplus_fourier = data.results.ttdplus_fourier.take();
        let ttdplus_refs = data.results.ttdplus_refs.take();
        let ttdplus_sap_flow = data.results.ttdplus_sap_flow.take();
        data.results = results;
        data.results.ttdplus_fourier = ttdplus_fourier;
        data.results.ttdplus_refs = ttdplus_refs;
        data.results.ttdplus_sap_flow = ttdplus_sap_flow;

        // A full run rebuilds every derived slot from raw/cleaned_data, so a
        // cleaning that had been applied to a DERIVED stage (T600, Tslope, …)
        // no longer exists. Drop its provenance, otherwise `get_cleaning_info`
        // keeps reporting it and the Calculs banner goes on offering
        // "recompute from t600 — the cleaning is preserved" over a T600 that
        // was just rebuilt from scratch.
        //
        // Cleanings applied to raw/cleaned are NOT affected: `cleaned_data` is
        // this pipeline's input, so that work is still very much in effect.
        let derived_cleaning = data
            .cleaning_source_dataset
            .as_deref()
            .is_some_and(|s| !matches!(s, "raw" | "cleaned"));
        if derived_cleaning {
            let lost = data.cleaning_source_dataset.take().unwrap_or_default();
            data.cleaning_method_label = None;
            data.cleaning_path = None;
            data.cleaning_target_columns.clear();
            data.cleaning_pre_snapshots.remove(&lost);
            logger::add_log(
                &mut data.logs,
                LogLevel::Warning,
                format!(
                    "Le nettoyage appliqué à « {} » a été écrasé par le recalcul complet du pipeline.",
                    lost
                ),
            );
        }

        // We only reach here when the run was allowed (no locks, or the user
        // forced through). Either way every derived stage was just rebuilt, so
        // any lock is now stale — drop them all.
        if !data.cleaning_locked_stages.is_empty() {
            let dropped = std::mem::take(&mut data.cleaning_locked_stages);
            logger::add_log(
                &mut data.logs,
                LogLevel::Warning,
                format!(
                    "Recalcul complet forcé : verrou(s) levé(s) sur {} (nettoyage écrasé).",
                    dropped.join(", ")
                ),
            );
        }

        // Replay step logs
        for (level, msg) in step_logs {
            logger::add_log(&mut data.logs, level, msg);
        }
        logger::add_log(
            &mut data.logs,
            if errors.is_empty() { LogLevel::Success } else { LogLevel::Warning },
            msg.clone(),
        );
        crate::utils::session_persist::save(&data);
    }

    Ok(serde_json::json!({
        "message": msg,
        "completed_steps": completed,
        "errors": errors,
        "vpd_par_diagnostic": vpd_par_diagnostic,
    }))
}

/// Recompute the DOWNSTREAM TTD steps after a calculated dataset was cleaned
/// in place (e.g. T600). Unlike `run_pipeline`, this does NOT recompute the
/// cleaned stage from raw — it keeps the cleaned value and only refreshes the
/// steps that depend on it (… → Tm → sTm → Tmi → K → Sap Flow), so the sap
/// flow reflects the cleaning. `stage` is the cleaned slot key.
#[tauri::command]
pub async fn recompute_from_stage(
    state: State<'_, AppState>,
    stage: String,
    force: Option<bool>,
) -> Result<serde_json::Value, String> {
    const STAGES: [&str; 9] = ["tslope", "baseline", "delta_t", "t600", "tm", "stm", "tmi", "k", "sap_flow"];
    let from_idx = STAGES.iter().position(|s| *s == stage)
        .ok_or_else(|| format!("Étape non recalculable en flux de sève : '{}'", stage))?;
    if from_idx >= 8 {
        return Err("Le flux de sève est la dernière étape — rien à recalculer en aval.".to_string());
    }

    // This keeps `stage` itself but REBUILDS every downstream stage — so a lock
    // on a downstream stage would be silently wiped. Same LOCKED_STAGES contract
    // as run_pipeline: refuse unless forced. (The recomputed-from stage is
    // preserved, so a lock there is never a problem.)
    {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        if !force.unwrap_or(false) {
            let downstream = &STAGES[from_idx + 1..];
            let hit: Vec<String> = data.cleaning_locked_stages.iter()
                .filter(|s| downstream.contains(&s.as_str()))
                .cloned()
                .collect();
            if !hit.is_empty() {
                return Err(format!("LOCKED_STAGES:{}", hit.join(",")));
            }
        }
    }

    let (input_df, env_df, pattern, tm_method, alpha, beta, mut res) = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let raw = data.raw_data.as_ref().ok_or("Aucune donnée chargée.")?.clone();
        let input = data.cleaned_data.as_ref().unwrap_or(&raw).clone();
        let env = match &data.sap_flow_params.tm_method {
            TmMethod::VpdPar { .. } | TmMethod::RegressionDiurne { .. } => data.env_data.clone(),
            _ => None,
        };
        (
            input, env,
            data.config.pattern.clone(),
            data.sap_flow_params.tm_method.clone(),
            data.sap_flow_params.alpha,
            data.sap_flow_params.beta,
            data.results.clone(),
        )
    };

    let (res, completed, errors) = tokio::task::spawn_blocking(move || {
        let mut completed: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        if from_idx < 1 {
            if let Some(tslope) = res.tslope.clone() {
                match calculations::calculate_baseline(&input_df, &tslope, &pattern) {
                    Ok(df) => { res.baseline = Some(df); completed.push("Baseline".into()); }
                    Err(e) => errors.push(format!("Baseline: {}", e)),
                }
            }
        }
        if from_idx < 2 {
            if let Some(baseline) = res.baseline.clone() {
                match calculations::calculate_delta_t(&input_df, &baseline) {
                    Ok(df) => { res.delta_t = Some(df); completed.push("DeltaT".into()); }
                    Err(e) => errors.push(format!("DeltaT: {}", e)),
                }
            }
        }
        if from_idx < 3 {
            if let Some(delta_t) = res.delta_t.clone() {
                match calculations::create_t600(&delta_t) {
                    Ok(df) => { res.t600 = Some(df); completed.push("T600".into()); }
                    Err(e) => errors.push(format!("T600: {}", e)),
                }
            }
        }
        if from_idx < 4 {
            if let Some(t600) = res.t600.clone() {
                match calculations::calculate_tm(&t600, &tm_method, env_df.as_ref()) {
                    Ok(out) => {
                        res.tm = Some(out.df);
                        res.diurnal_diagnostics = out.diurnal_diagnostics;
                        res.vpd_par_stats = out.vpd_par_stats;
                        if let Some(steps) = out.diurnal_steps {
                            res.rd_regression = Some(steps.regression);
                            res.rd_result = Some(steps.result);
                        }
                        completed.push("T0".into());
                    }
                    Err(e) => errors.push(format!("T0: {}", e)),
                }
            }
        }
        if from_idx < 5 {
            if let (Some(tm), Some(t600)) = (res.tm.clone(), res.t600.clone()) {
                match calculations::calculate_stm(&tm, &t600) {
                    Ok(df) => { res.stm = Some(df); completed.push("sT0".into()); }
                    Err(e) => errors.push(format!("sT0: {}", e)),
                }
            }
        }
        if from_idx < 6 {
            if let (Some(tm), Some(stm), Some(t600)) = (res.tm.clone(), res.stm.clone(), res.t600.clone()) {
                match calculations::calculate_tmi(&tm, &stm, &t600) {
                    Ok(df) => { res.tmi = Some(df); completed.push("T0i".into()); }
                    Err(e) => errors.push(format!("T0i: {}", e)),
                }
            }
        }
        if from_idx < 7 {
            if let (Some(tmi), Some(t600)) = (res.tmi.clone(), res.t600.clone()) {
                match calculations::calculate_k(&tmi, &t600) {
                    Ok(df) => { res.k = Some(df); completed.push("K".into()); }
                    Err(e) => errors.push(format!("K: {}", e)),
                }
            }
        }
        if from_idx < 8 {
            if let Some(k) = res.k.clone() {
                match calculations::calculate_sapflow(&k, alpha, beta) {
                    Ok(df) => { res.sap_flow = Some(df); completed.push("SapFlow".into()); }
                    Err(e) => errors.push(format!("SapFlow: {}", e)),
                }
            }
        }

        (res, completed, errors)
    })
    .await
    .map_err(|e| e.to_string())?;

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.results = res;
        // Every downstream stage was just rebuilt — a lock on one of them is now
        // stale (we only got here past the guard because force=true or there was
        // no downstream lock). The recomputed-from stage itself is preserved, so
        // its lock survives.
        if !data.cleaning_locked_stages.is_empty() {
            let from = STAGES.iter().position(|s| *s == stage).unwrap_or(0);
            let downstream = &STAGES[from + 1..];
            let before = data.cleaning_locked_stages.len();
            data.cleaning_locked_stages.retain(|s| !downstream.contains(&s.as_str()));
            if data.cleaning_locked_stages.len() != before {
                logger::add_log(
                    &mut data.logs,
                    LogLevel::Warning,
                    format!("Recalcul partiel forcé depuis « {} » : verrou(s) aval levé(s).", stage),
                );
            }
        }
        logger::add_log(
            &mut data.logs,
            if errors.is_empty() { LogLevel::Success } else { LogLevel::Warning },
            format!("Recalcul du flux de sève depuis '{}' : {} étape(s), {} erreur(s)", stage, completed.len(), errors.len()),
        );
        crate::utils::session_persist::save(&data);
    }

    Ok(serde_json::json!({
        "from": stage,
        "completed_steps": completed,
        "errors": errors,
    }))
}

// =============================================================================
// TTD+ Commands
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct TtdPlusParamsInput {
    pub period: f64,
    pub num_samples: usize,
    pub dt: f64,
    pub distance: f64,
    pub ref_hour: Option<u32>,
}

#[tauri::command]
pub fn set_ttdplus_params(
    state: State<'_, AppState>,
    params: TtdPlusParamsInput,
) -> Result<(), String> {
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    data.ttdplus_params = TtdPlusParams {
        period: params.period,
        num_samples: params.num_samples,
        dt: params.dt,
        distance: params.distance,
        ref_hour: params.ref_hour.unwrap_or(6),
    };
    Ok(())
}

#[tauri::command]
pub async fn run_ttdplus_pipeline(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Clone data out of lock
    let (input_df, params) = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        let raw = data.raw_data.as_ref()
            .ok_or("No data loaded. Import data first.")?;
        let df = data.cleaned_data.as_ref().unwrap_or(raw).clone();
        let p = data.ttdplus_params.clone();
        (df, p)
    };

    // Run heavy computation on blocking thread
    let result = tokio::task::spawn_blocking(move || {
        ttdplus::run_ttdplus_pipeline(&input_df, &params)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let (fourier_df, refs_df, sap_flow_df) = result;
    let num_cycles = sap_flow_df.height();
    let num_cols = sap_flow_df.width() - 1; // minus timestamp

    // Store results back in state
    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.results.ttdplus_fourier = Some(fourier_df);
        data.results.ttdplus_refs = Some(refs_df);
        data.results.ttdplus_sap_flow = Some(sap_flow_df);

        logger::add_log(
            &mut data.logs,
            LogLevel::Success,
            format!("TTD+ pipeline complete: {} cycles, {} columns", num_cycles, num_cols),
        );
        crate::utils::session_persist::save(&data);
    }

    Ok(serde_json::json!({
        "message": format!("TTD+ pipeline complete: {} cycles, {} colonnes", num_cycles, num_cols),
        "completed_steps": ["Fourier", "References", "SapFlow"],
        "errors": [],
        "num_cycles": num_cycles,
    }))
}

/// List the (date, T600 column) pairs for which a RegressionDiurne diagnostic
/// was computed during the last pipeline run. Used by the CalculsPage to
/// populate the date/column pickers above the inspection charts.
#[tauri::command]
pub fn list_diurnal_regression_diagnostics(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;
    let map = match data.results.diurnal_diagnostics.as_ref() {
        Some(m) => m,
        None => return Ok(serde_json::json!({ "dates": [], "columns": [] })),
    };
    let mut dates: Vec<String> = map.keys().map(|(d, _)| d.clone()).collect();
    let mut columns: Vec<String> = map.keys().map(|(_, c)| c.clone()).collect();
    dates.sort(); dates.dedup();
    columns.sort(); columns.dedup();
    Ok(serde_json::json!({ "dates": dates, "columns": columns }))
}

/// Fetch one diagnostic entry (one night × one T600 column).
#[tauri::command]
pub fn get_diurnal_regression_diagnostic(
    state: State<'_, AppState>,
    date: String,
    column: String,
) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;
    let map = data.results.diurnal_diagnostics.as_ref()
        .ok_or_else(|| "No diurnal diagnostics available — run the pipeline with RegressionDiurne first.".to_string())?;
    let diag = map.get(&(date.clone(), column.clone()))
        .ok_or_else(|| format!("No diagnostic for ({}, {}).", date, column))?;
    serde_json::to_value(diag).map_err(|e| e.to_string())
}

// =============================================================================
// Calculs avancés — Jh → Jhp → Qh → Qd
// =============================================================================

/// Returns the sensors available to populate group definitions on the
/// Calculs avancés tab. Each entry is parsed from a `Fd_<tree>-<position>-<index>`
/// column name in `sap_flow`. Sensors that don't follow the three-segment
/// convention come back with `tree = ""` and `position = ""`.
///
/// Also returns the persisted groups so the frontend can re-render the user's
/// previous configuration on app reload.
#[tauri::command]
pub fn list_advanced_sensors(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;
    let sap_flow = data.results.sap_flow.as_ref()
        .ok_or_else(|| "Flux de sève non calculé — lance d'abord le pipeline TTD.".to_string())?;

    let mut sensors: Vec<serde_json::Value> = Vec::new();
    for col in sap_flow.get_column_names().iter().filter(|c| c.starts_with("Fd_")) {
        let raw = &col[3..]; // strip "Fd_"
        let parts: Vec<&str> = raw.splitn(3, '-').collect();
        let (tree, position, index) = match parts.as_slice() {
            [t, p, i] => (t.to_string(), p.to_string(), i.to_string()),
            _ => (String::new(), String::new(), raw.to_string()),
        };
        sensors.push(serde_json::json!({
            "sensor_id": raw,
            "tree": tree,
            "position": position,
            "index": index,
        }));
    }

    Ok(serde_json::json!({
        "sensors": sensors,
        "station": data.file_path.as_ref().and_then(|p| {
            std::path::Path::new(p).file_stem().map(|s| s.to_string_lossy().to_string())
        }),
        "groups": data.advanced_groups,
    }))
}

#[derive(Debug, Deserialize)]
pub struct AdvancedGroupInput {
    pub name: String,
    pub sensors: Vec<String>,
    pub k_radial: f64,
    pub a_sapwood_dm2: f64,
}

/// Runs the full Jh → Jhp → Qh → Qd chain on the current `sap_flow` for the
/// user-defined groups. Stores the four resulting DataFrames in
/// `CalculationResults` and persists the group definitions in
/// `AppData::advanced_groups` so they survive a restart.
#[tauri::command]
pub fn compute_advanced_chain(
    state: State<'_, AppState>,
    groups: Vec<AdvancedGroupInput>,
) -> Result<serde_json::Value, String> {
    if groups.is_empty() {
        return Err("Aucun groupe défini.".to_string());
    }
    // Validate unique names — DataFrame columns must be unique.
    {
        let mut seen = std::collections::HashSet::new();
        for g in &groups {
            if !seen.insert(g.name.clone()) {
                return Err(format!("Nom de groupe dupliqué : '{}'. Les noms doivent être uniques.", g.name));
            }
            if g.name.trim().is_empty() {
                return Err("Un groupe a un nom vide.".to_string());
            }
            if g.name.eq_ignore_ascii_case("TIMESTAMP") || g.name.eq_ignore_ascii_case("DATE") || g.name.eq_ignore_ascii_case("n_rows_in_day") {
                return Err(format!("Nom de groupe réservé : '{}'.", g.name));
            }
        }
    }

    let advanced_groups: Vec<crate::core::types::AdvancedGroup> = groups.into_iter()
        .map(|g| crate::core::types::AdvancedGroup {
            name: g.name,
            sensors: g.sensors,
            k_radial: g.k_radial,
            a_sapwood_dm2: g.a_sapwood_dm2,
        })
        .collect();

    // Snapshot sap_flow under a brief lock then release.
    let sap_flow = {
        let data = state.inner.lock().map_err(|e| e.to_string())?;
        data.results.sap_flow.as_ref()
            .ok_or_else(|| "Flux de sève non calculé — lance d'abord le pipeline TTD.".to_string())?
            .clone()
    };

    let jh = calculations::compute_jh(&sap_flow, &advanced_groups).map_err(|e| e.to_string())?;
    let jhp = calculations::compute_jhp(&jh, &advanced_groups).map_err(|e| e.to_string())?;
    let qh = calculations::compute_qh(&jhp, &advanced_groups).map_err(|e| e.to_string())?;
    let qd = calculations::compute_qd(&qh).map_err(|e| e.to_string())?;

    let n_rows_jh = jh.height();
    let n_rows_qd = qd.height();
    let n_groups = advanced_groups.len();

    {
        let mut data = state.inner.lock().map_err(|e| e.to_string())?;
        data.results.jh = Some(jh);
        data.results.jhp = Some(jhp);
        data.results.qh = Some(qh);
        data.results.qd = Some(qd);
        data.advanced_groups = advanced_groups;
        logger::add_log(
            &mut data.logs,
            LogLevel::Success,
            format!(
                "Calculs avancés OK : {} groupes × {} pas → Jh / Jhp / Qh ({} lignes), Qd ({} jours)",
                n_groups, n_rows_jh, n_rows_jh, n_rows_qd
            ),
        );
        crate::utils::session_persist::save(&data);
    }

    Ok(serde_json::json!({
        "n_groups": n_groups,
        "n_rows": n_rows_jh,
        "n_days": n_rows_qd,
    }))
}
