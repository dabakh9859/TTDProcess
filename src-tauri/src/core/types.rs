use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Project Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub min_temp: f64,
    pub max_temp: f64,
    pub temp_columns: Vec<String>,
    pub pattern: Vec<i64>,
    pub auto_correct: bool,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            min_temp: -10.0,
            max_temp: 60.0,
            temp_columns: Vec::new(),
            pattern: vec![30, 30, 60, 180, 300, 30, 30, 60, 180, 300, 600],
            auto_correct: true,
        }
    }
}

// =============================================================================
// Sap Flow Parameters
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SapFlowParams {
    pub alpha: f64,
    pub beta: f64,
    pub tm_method: TmMethod,
    pub night_start: u32,
    pub night_end: u32,
    pub nb_max_points: usize,
    /// Smoothing applied to the final T0 series, for ALL T0 methods.
    #[serde(default)]
    pub t0_smooth: T0SmoothConfig,
}

/// T0 curve smoothing — several modalities, each with its own parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T0SmoothConfig {
    /// "none" | "moving_avg" | "median" | "spline" | "ewma".
    #[serde(default)]
    pub method: String,
    /// Window in nights (moving_avg / median).
    #[serde(default = "default_one_usize")]
    pub window: usize,
    /// Smoothing strength λ for the spline (Whittaker) — higher = smoother.
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    /// Reactivity α for the exponential smoother (0–1, higher = less smooth).
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

fn default_one_usize() -> usize { 1 }
fn default_lambda() -> f64 { 10.0 }
fn default_alpha() -> f64 { 0.3 }

impl Default for T0SmoothConfig {
    fn default() -> Self {
        Self { method: "none".into(), window: 1, lambda: 10.0, alpha: 0.3 }
    }
}

impl Default for SapFlowParams {
    fn default() -> Self {
        Self {
            alpha: 12.95,
            beta: 1.0,
            tm_method: TmMethod::default(),
            night_start: 20,
            night_end: 8,
            nb_max_points: 1,
            t0_smooth: T0SmoothConfig::default(),
        }
    }
}

// =============================================================================
// Tm Method variants
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TmMethod {
    PreAube { nb_max_points: usize },
    /// Granier 1985 / Lu 2004 / Pasqualotto 2019 rolling-baseline method.
    /// Two passes:
    ///   1. For each calendar day D, compute ΔT_max over the nocturnal
    ///      window [night_start_hour..night_end_hour) — the daily maximum
    ///      "no-flow proxy".
    ///   2. For each day D, set Tm(D) = max{ΔT_max(D−n) … ΔT_max(D+n)}
    ///      across a rolling window of `window_days` (centred on D).
    /// Tm(D) is then propagated to every T600 timestamp belonging to day D.
    FenetreGlissante {
        window_days: usize,
        night_start_hour: u32,
        night_end_hour: u32,
    },
    DoubleRegression { min_points: usize },
    /// Oishi et al. 2008/2016 method: ΔTmax identified only when three
    /// cumulative "no-flow" conditions are met (night + low radiation,
    /// low VPD, thermal stability over a sliding window).
    VpdPar { config: VpdParConfig },
    /// Regaldo & Ritter 2007 method: Diurnal regression based on ETo.
    /// Robust regression between 1/T600 and ETo^(1/β), where the intercept
    /// gives 1/T600_max. Uses diurnal data only to avoid nighttime sap flow bias.
    RegressionDiurne {
        alpha: f64,
        beta: f64,
        etp_column: String,
        /// Optional manual selection range on X = ETo^(1/β). When EITHER
        /// bound is set (or hour_min/hour_max below), only points inside
        /// these ranges feed the regression. With every bound None, ALL
        /// diurnal points (default 6h–20h window) feed the regression.
        /// #[serde(default)] so pre-existing scenarios (saved before these
        /// fields existed) deserialize cleanly — load_scenarios() used to
        /// silently drop any failing file.
        #[serde(default)]
        x_min: Option<f64>,
        #[serde(default)]
        x_max: Option<f64>,
        /// Optional manual diurnal window in clock hours [0..24). Narrows the
        /// default 6h–20h window — useful when the user wants to keep ONLY
        /// the morning stomatal-opening points (~6h–9h), without also picking
        /// up the late-afternoon low-ETo points that share the same X value.
        #[serde(default)]
        hour_min: Option<u32>,
        #[serde(default)]
        hour_max: Option<u32>,
        /// When true, run a Pasqualotto-style two-pass robust fit: first
        /// Theil-Sen on the selected points, then keep only the LOWER
        /// envelope (Y ≤ slope1·X + intercept1) — the physical analogue of
        /// the nocturnal upper envelope, since here Y = 1/T600 and small Y
        /// means large T600 (i.e. close to ΔTmax). Re-fit Theil-Sen on the
        /// lower envelope, and T0 = 1 / (new intercept).
        #[serde(default)]
        double_regression: bool,
    },
}

impl Default for TmMethod {
    fn default() -> Self {
        TmMethod::PreAube { nb_max_points: 3 }
    }
}

// =============================================================================
// VPD/PAR Method configuration (Oishi 2008/2016, Rabbel 2016, Ward 2017)
// =============================================================================

/// Configuration for the VpdPar Tm method.
///
/// References:
/// - Oishi, Oren & Stoy 2008, 2016 — original VPD/PAR + stability criteria
/// - Rabbel et al. 2016 — critique of the nocturnal-zero-flow assumption
/// - Ward et al. 2017 (TRACC) — linear interpolation for missing nights
/// - Wang et al. 2025 — synthesis of no-flow conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpdParConfig {
    /// PAR threshold in µmol·m⁻²·s⁻¹ (ignored if `use_par` is false).
    /// Default 10.0, range [0, 50].
    pub par_threshold: f64,
    /// Global radiation threshold in W·m⁻² (used if `use_par` is false).
    /// Default 5.0, range [0, 20].
    pub solar_threshold: f64,
    /// Maximum VPD to accept a point as no-flow. Expressed in **hPa**
    /// (the unit most flux towers export). Default 1.0 hPa
    /// (= 0.1 kPa, Oishi 2008 reference). Range [0.05, 5.0] covers
    /// both kPa- and hPa-stored columns since the algorithm just
    /// compares to the raw column value.
    pub vpd_threshold: f64,
    /// Maximum coefficient of variation of ΔT over the stability window
    /// (std/mean). Default 0.01 (= 1%), range [0.005, 0.05].
    pub cv_threshold: f64,
    /// Width of the stability window in hours. Default 2.0, range [1.0, 6.0].
    pub stability_window_h: f64,
    /// Minimum number of consecutive valid points required inside a stability
    /// window. Default 4 (= 2h at 30-min steps), range [2, 12].
    pub min_consecutive_pts: usize,
    /// Start hour of the nocturnal window (inclusive). Default 20, range [18, 23].
    pub night_start_hour: u32,
    /// End hour of the nocturnal window (exclusive). Default 6, range [3, 8].
    pub night_end_hour: u32,
    /// Legacy "use PAR vs Rg" toggle. Kept on the type for backward-compat
    /// deserialization of older scenarios — the algorithm now reads
    /// `par_column` and `solar_column` independently and applies BOTH
    /// filters when both are set. Ignored at runtime.
    #[serde(default = "default_true_use_par")]
    pub use_par: bool,
    /// Name of the VPD column inside env_data. Required.
    pub vpd_column: String,
    /// Legacy single-radiation-column field. Still deserialized for older
    /// scenarios. The algorithm prefers `par_column` / `solar_column`; if
    /// those are empty and this is set, it gets routed to one of them
    /// based on `use_par` (back-compat path).
    #[serde(default)]
    pub radiation_column: String,
    /// PAR column name. When non-empty, rows must satisfy
    /// `PAR < par_threshold` to be counted as no-flow. Optional — may be
    /// used alongside `solar_column` for a cumulative AND filter.
    #[serde(default)]
    pub par_column: String,
    /// Global-radiation (Rg) column name. When non-empty, rows must
    /// satisfy `Rg < solar_threshold`. Optional, can coexist with PAR.
    #[serde(default)]
    pub solar_column: String,
    /// Name of the timestamp column inside env_data. Required.
    pub timestamp_column: String,
    /// How to fill nights where no stable-window candidate exists. Default
    /// "Spline" (smooth cubic through the valid nights). Falls back to the
    /// legacy "Linear" behaviour for pre-existing scenarios that don't carry
    /// this field. See `NightInterpolationMethod` for the available options.
    #[serde(default)]
    pub interpolation_method: NightInterpolationMethod,
    /// Smoothing window (in nights) applied to the valid nights before the
    /// cubic spline ("Courbe lisse"). 1 = interpolate through every point;
    /// higher = smoother. Only affects `NightInterpolationMethod::Spline`.
    #[serde(default = "default_spline_smooth_window")]
    pub spline_smooth_window: usize,
}

fn default_spline_smooth_window() -> usize { 1 }

/// Strategy for filling T0 values on nights where the env-stable-window
/// search yielded nothing. Drives the second pass of `calculate_tm_vpd_par`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NightInterpolationMethod {
    /// Linear interpolation between the nearest prev + next valid nights,
    /// constant-hold at the series edges. Original TRACC behaviour.
    Linear,
    /// Nearest-valid neighbour (no interpolation, just snap to the closest
    /// valid night in time).
    Nearest,
    /// Last-observation-carried-forward — repeat the most recent valid value
    /// until the next valid one appears.
    Hold,
    /// Natural cubic spline through every valid night. Smoother than linear
    /// but can overshoot on long gaps with curvature in the data.
    Spline,
    /// Global OLS regression `T0 ~ day_index` fitted on all valid nights,
    /// evaluated at every missing night. Captures slow sensor drift.
    LinearRegression,
    /// Pasqualotto-style two-pass robust fit. Pass 1 = OLS on all valid
    /// nights; pass 2 = keep ONLY points above the line (upper envelope —
    /// the no-flow ceiling) and refit; evaluate the second line at every
    /// missing night.
    DoubleRegression,
}

impl Default for NightInterpolationMethod {
    fn default() -> Self { NightInterpolationMethod::Spline }
}

fn default_true_use_par() -> bool { true }

impl Default for VpdParConfig {
    fn default() -> Self {
        Self {
            par_threshold: 10.0,
            solar_threshold: 5.0,
            vpd_threshold: 5.0,
            cv_threshold: 0.05,
            stability_window_h: 2.0,
            min_consecutive_pts: 4,
            night_start_hour: 20,
            night_end_hour: 6,
            use_par: true,
            vpd_column: String::new(),
            radiation_column: String::new(),
            par_column: String::new(),
            solar_column: String::new(),
            timestamp_column: String::new(),
            interpolation_method: NightInterpolationMethod::default(),
            spline_smooth_window: 1,
        }
    }
}

impl VpdParConfig {
    /// Validate all threshold ranges. Columns are validated later at runtime
    /// when we actually try to read them from env_data.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=50.0).contains(&self.par_threshold) {
            return Err("par_threshold doit être entre 0 et 50 µmol·m⁻²·s⁻¹".into());
        }
        if !(0.0..=20.0).contains(&self.solar_threshold) {
            return Err("solar_threshold doit être entre 0 et 20 W·m⁻²".into());
        }
        // VPD column expected in hPa. Range up to 20 hPa covers dry/Sahel
        // climates where nocturnal VPD can routinely sit at 3–10 hPa.
        if !(0.1..=20.0).contains(&self.vpd_threshold) {
            return Err("vpd_threshold doit être entre 0.1 et 20 hPa".into());
        }
        if !(0.005..=0.20).contains(&self.cv_threshold) {
            return Err("cv_threshold doit être entre 0.005 et 0.20".into());
        }
        if !(1.0..=6.0).contains(&self.stability_window_h) {
            return Err("stability_window_h doit être entre 1 et 6 heures".into());
        }
        if !(2..=12).contains(&self.min_consecutive_pts) {
            return Err("min_consecutive_pts doit être entre 2 et 12 points".into());
        }
        if !(18..=23).contains(&self.night_start_hour) {
            return Err("night_start_hour doit être entre 18 et 23".into());
        }
        if !(3..=8).contains(&self.night_end_hour) {
            return Err("night_end_hour doit être entre 3 et 8".into());
        }
        if self.vpd_column.is_empty() {
            return Err("Colonne VPD non renseignée".into());
        }
        // At least one radiation column must be set (PAR or Rg). Both can
        // be set simultaneously — the algorithm then ANDs both filters.
        // The legacy `radiation_column` is also accepted to load older
        // configs (the algorithm routes it via `use_par`).
        let has_radiation = !self.par_column.is_empty()
            || !self.solar_column.is_empty()
            || !self.radiation_column.is_empty();
        if !has_radiation {
            return Err("Sélectionne au moins une colonne de rayonnement (PAR ou Rg)".into());
        }
        if self.timestamp_column.is_empty() {
            return Err("Colonne timestamp non renseignée".into());
        }
        Ok(())
    }
}

// =============================================================================
// TTD+ Parameters
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtdPlusParams {
    /// Period of heating/cooling cycle in seconds
    pub period: f64,
    /// Number of temperature samples per cycle (must be even)
    pub num_samples: usize,
    /// Time interval between readings in seconds
    pub dt: f64,
    /// Distance between probe and heat source in meters
    pub distance: f64,
    /// Hour of day for daily reference calculation (default 6)
    pub ref_hour: u32,
}

impl Default for TtdPlusParams {
    fn default() -> Self {
        Self {
            period: 300.0,
            num_samples: 10,
            dt: 30.0,
            distance: 0.01,
            ref_hour: 6,
        }
    }
}

// =============================================================================
// Calculation Results (9-step TTD pipeline + TTD+ pipeline)
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct CalculationResults {
    // TTD classic pipeline
    pub tslope: Option<DataFrame>,
    pub baseline: Option<DataFrame>,
    pub delta_t: Option<DataFrame>,
    pub t600: Option<DataFrame>,
    pub tm: Option<DataFrame>,
    pub stm: Option<DataFrame>,
    pub tmi: Option<DataFrame>,
    pub k: Option<DataFrame>,
    pub sap_flow: Option<DataFrame>,
    // TTD+ pipeline
    pub ttdplus_fourier: Option<DataFrame>,
    pub ttdplus_refs: Option<DataFrame>,
    pub ttdplus_sap_flow: Option<DataFrame>,
    /// Diagnostic data emitted by the RegressionDiurne Tm method, indexed by
    /// (night_date, T600_column). Lets the CalculsPage render the two
    /// inspection charts (daily-cycle comparison + regression scatter) for
    /// any night the user picks. Populated only when the user runs the
    /// pipeline with TmMethod::RegressionDiurne.
    pub diurnal_diagnostics: Option<std::collections::HashMap<(String, String), DiurnalDiagnostic>>,
    /// Per-filter pass-rate stats emitted by the VpdPar Tm method. Populated
    /// only when the user runs the pipeline with TmMethod::VpdPar.
    pub vpd_par_stats: Option<VpdParStats>,
    /// Step-by-step verification tables for the RegressionDiurne T0
    /// determination. Each table mirrors one stage of the algorithm so a
    /// reviewer can re-check it by hand in the Tableau tab. Populated only
    /// when the pipeline runs with TmMethod::RegressionDiurne; None otherwise.
    pub rd_regression: Option<DataFrame>, // Step 4: Theil-Sen slope/intercept/r²
    pub rd_result: Option<DataFrame>,     // Step 5: T0 = 1/intercept (None when the night was skipped)

    // ──────────────────────────────────────────────────────────────────────
    // Calculs avancés — Jh → Jhp → Qh → Qd  (per-group aggregation)
    //
    // Built from `sap_flow` (Fd_<sensor>) by the "Flux total" chain on the
    // Calculs avancés tab. Each column is a USER-DEFINED GROUP (one tree, or
    // a custom pool of multiple trees), not a single sensor.
    //   - jh   = mean(Fd) across the group's selected sensors   (L/dm²/h)
    //   - jhp  = jh × k_radial                                   (L/dm²/h)
    //   - qh   = jhp × A_sapwood_dm2                             (L/h)
    //   - qd   = ∑(qh) × Δt/3600 over one calendar day            (L/day)
    // Group definitions live in `AppData::advanced_groups` so the chain is
    // reproducible across sessions.
    // ──────────────────────────────────────────────────────────────────────
    pub jh: Option<DataFrame>,   // TIMESTAMP + <group_name>… in L/dm²/h
    pub jhp: Option<DataFrame>,  // TIMESTAMP + <group_name>… in L/dm²/h
    pub qh: Option<DataFrame>,   // TIMESTAMP + <group_name>… in L/h
    pub qd: Option<DataFrame>,   // DATE + <group_name>… in L/day
}

/// One scenario loaded as an *overlay* on the Visualisation tab. Holds the
/// scenario's DataFrames in memory so the viz can list them as additional
/// sources alongside the live state, without overwriting the user's current
/// pipeline. Several overlays can be loaded simultaneously to compare runs.
#[derive(Debug, Clone)]
pub struct ScenarioOverlay {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub raw_data: Option<DataFrame>,
    pub cleaned_data: Option<DataFrame>,
    pub env_data: Option<DataFrame>,
    pub tslope: Option<DataFrame>,
    pub baseline: Option<DataFrame>,
    pub delta_t: Option<DataFrame>,
    pub t600: Option<DataFrame>,
    pub tm: Option<DataFrame>,
    pub stm: Option<DataFrame>,
    pub tmi: Option<DataFrame>,
    pub k: Option<DataFrame>,
    pub sap_flow: Option<DataFrame>,
    pub ttdplus_fourier: Option<DataFrame>,
    pub ttdplus_refs: Option<DataFrame>,
    pub ttdplus_sap_flow: Option<DataFrame>,
    pub rd_regression: Option<DataFrame>,
    pub rd_result: Option<DataFrame>,
    pub jh: Option<DataFrame>,
    pub jhp: Option<DataFrame>,
    pub qh: Option<DataFrame>,
    pub qd: Option<DataFrame>,
}

impl ScenarioOverlay {
    /// Lookup a DataFrame by its dataset key (raw / t600 / fd / jh / …).
    /// Returns None for unknown keys or empty slots.
    pub fn dataset(&self, key: &str) -> Option<&DataFrame> {
        match key {
            "raw" => self.raw_data.as_ref(),
            "cleaned" => self.cleaned_data.as_ref(),
            "env" => self.env_data.as_ref(),
            "tslope" => self.tslope.as_ref(),
            "baseline" => self.baseline.as_ref(),
            "delta_t" => self.delta_t.as_ref(),
            "t600" => self.t600.as_ref(),
            "tm" => self.tm.as_ref(),
            "stm" => self.stm.as_ref(),
            "tmi" => self.tmi.as_ref(),
            "k" => self.k.as_ref(),
            "sap_flow" => self.sap_flow.as_ref(),
            "ttdplus_fourier" => self.ttdplus_fourier.as_ref(),
            "ttdplus_refs" => self.ttdplus_refs.as_ref(),
            "ttdplus_sap_flow" => self.ttdplus_sap_flow.as_ref(),
            "rd_regression" => self.rd_regression.as_ref(),
            "rd_result" => self.rd_result.as_ref(),
            "jh" => self.jh.as_ref(),
            "jhp" => self.jhp.as_ref(),
            "qh" => self.qh.as_ref(),
            "qd" => self.qd.as_ref(),
            _ => None,
        }
    }
}

/// Per-group definition for the Calculs avancés chain. A group is either a
/// single tree (auto-detected from the sensor-name prefix) or a custom pool
/// (user-composed). It carries the sensors that feed the Jh average, plus
/// the two scalar coefficients applied downstream (radial weighting + sapwood
/// area).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedGroup {
    /// Display name + DataFrame column name (e.g. "1a", "pool5"). Must be
    /// unique within the groups list.
    pub name: String,
    /// Sensor IDs (Fd column name minus the `Fd_` prefix) that get averaged
    /// into the group's Jh. Typically the TA-position sensors of one tree.
    pub sensors: Vec<String>,
    /// Radial weighting factor (dimensionless, ~0.5). Applied on Jh → Jhp.
    pub k_radial: f64,
    /// Sapwood cross-section area in dm². Applied on Jhp → Qh.
    pub a_sapwood_dm2: f64,
}

/// One diurnal-cycle's regression diagnostic for one (date, T600 column) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiurnalDiagnostic {
    /// Date the night belongs to (YYYY-MM-DD, the "previous day" for the
    /// 20h→8h window).
    pub date: String,
    /// T600 column this diagnostic is about (e.g. "T600_1a-TA-1").
    pub column: String,
    /// Diurnal points (6h ≤ hour < 20h) used for the regression.
    pub points: Vec<DiurnalPoint>,
    /// Same indices as `points`, masking which ones survived the tolerance
    /// filter and were actually fed to the robust regression.
    pub selected: Vec<bool>,
    /// β exponent used to transform ETo. Mirrored back so the frontend can
    /// label the X axis correctly ("ETo^(1/β)").
    pub beta: f64,
    /// Theil-Sen slope & intercept of 1/T600 = slope·ETo^(1/β) + intercept.
    pub slope: f64,
    pub intercept: f64,
    /// Coefficient of determination of the fit on the selected points.
    pub r2: f64,
    /// T600_max = 1 / intercept (the value reported as Tm for that night).
    pub t_max: Option<f64>,
    /// Max T600 measured during the nocturnal window (20h → 8h). Kept as a
    /// side-by-side informational comparison to `t_max` (large gap = strong
    /// diurnal signal) — not used as a T0 fallback.
    pub t_night_max: Option<f64>,
    /// Diagnostic counters that explain *why* `points` ended up the size it
    /// did. Useful for nights where the regression had to be skipped (T0 =
    /// None, e.g. ETo file doesn't cover the same period as T600).
    pub n_in_window: usize,    // samples falling in the 6h-20h band, this column
    pub n_t600_valid: usize,   // of those, with finite T600 > 0
    pub n_etp_matched: usize,  // of those, with an ETo value (any level: sec/hour/day)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiurnalPoint {
    /// ISO-8601 UTC timestamp of the original 5-min sample.
    pub timestamp: String,
    /// Hour of day in 24h notation, with minute fraction (e.g. 14.5 for 14:30).
    pub hour: f64,
    pub etp: f64,
    pub t600: f64,
}

/// Diagnostic counters emitted by the VpdPar Tm method. Helps the user
/// understand WHICH filter (radiation / VPD / stability) is the bottleneck
/// when every night ends up tagged "NoValidNight".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VpdParStats {
    pub n_total: usize,
    pub n_in_clock_window: usize,  // hour in [night_start..night_end]
    pub n_rad_ok: usize,           // PAR/Rg below threshold (cumulative AND)
    pub n_night_flag: usize,       // clock_window AND rad_ok
    pub n_vpd_ok: usize,           // VPD below threshold
    pub n_env_passed: usize,       // night_flag AND vpd_ok (joint env pass)
    pub n_valid_nights: usize,     // nights where regression actually returned a Valid Tm
    pub n_interpolated_nights: usize,
    pub n_no_valid_nights: usize,
}

// =============================================================================
// Logging
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
}

// =============================================================================
// Scenarios
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub data_source: String,
    pub tm_method: TmMethod,
    pub sap_flow_params: SapFlowParams,
    pub results_summary: ResultsSummary,
    /// Cleaning path chosen (Classical vs AIBased). Optional for legacy
    /// scenarios created before the two-path architecture existed.
    #[serde(default)]
    pub cleaning_path: Option<CleaningPath>,
    /// Where in the pipeline cleaning was applied.
    #[serde(default)]
    pub cleaning_stage: Option<CleaningStage>,
    /// Full cleaning method + config (replaces the loose `String` that some
    /// older code paths used).
    #[serde(default)]
    pub cleaning_method: Option<CleaningMethod>,
    /// Aggregated cleaning statistics (detection + replacement counts).
    #[serde(default)]
    pub cleaning_report: Option<CleaningReport>,
    /// ML metrics — populated only when `cleaning_path == AIBased`.
    #[serde(default)]
    pub ml_metrics: Option<CleaningTrainingMetrics>,
    /// On-disk path to the serialized model bundle (for Voie B reload).
    #[serde(default)]
    pub model_path: Option<String>,
    /// Columns used as predictors (Voie B) — stored for reproducibility.
    #[serde(default)]
    pub predictor_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsSummary {
    pub total_rows: usize,
    pub num_columns: usize,
    pub date_range: (String, String),
    pub mean_sap_flow: f64,
}

impl Default for ResultsSummary {
    fn default() -> Self {
        Self {
            total_rows: 0,
            num_columns: 0,
            date_range: (String::new(), String::new()),
            mean_sap_flow: 0.0,
        }
    }
}

// =============================================================================
// ML types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LSTM,
    RandomForest,
    LinearRegression,
    GradientBoosting,
    Prophet,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingPreset {
    Rapide,
    Standard,
    Precis,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    pub model_type: ModelType,
    pub preset: TrainingPreset,
    pub input_columns: Vec<String>,
    pub target_column: String,
    pub train_start_idx: usize,
    pub train_end_idx: usize,
    pub test_split: f64,
    pub custom_params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedModelInfo {
    pub model_id: String,
    pub model_type: ModelType,
    pub target_column: String,
    pub metrics: TrainingMetrics,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub mape: f64,
}

// =============================================================================
// Data Cleaning types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionState {
    pub method: String,
    pub dataset: String,
    pub outlier_indices: HashMap<String, Vec<usize>>,
    pub validated: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    IQR { multiplier: f64 },
    ZScore { threshold: f64 },
    MAD { threshold: f64 },
    IsolationForest { contamination: f64 },
    RollingZScore { window_size: usize, threshold: f64 },
    LSTM { model_id: String },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillMethod {
    Linear,
    MovingAverage { window_size: usize },
    MLModel { model_id: String },
}

// =============================================================================
// Aggregation types
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationPeriod {
    Journalier,
    Hebdomadaire,
    Mensuel,
    Annuel,
    Custom { days: usize },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationOp {
    Moyenne,
    Somme,
    Minimum,
    Maximum,
}

// =============================================================================
// Column statistics (for import page)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub name: String,
    pub dtype: String,
    pub count: usize,
    pub null_count: usize,
    pub null_percentage: f64,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub median: Option<f64>,
    /// Number of outliers detected via IQR method
    pub outlier_count: usize,
    /// Whether the column was detected as a timestamp/datetime
    pub is_timestamp: bool,
    /// Number of timestamp ordering violations (non-monotonic)
    pub timestamp_gaps: usize,
    /// Overall quality score 0–100
    pub quality_score: f64,
}

// =============================================================================
// Import options (used internally)
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub file_path: String,
    pub sheet_name: Option<String>,
    pub header_row: usize,
    pub data_start_row: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            sheet_name: None,
            header_row: 0,
            data_start_row: 1,
        }
    }
}

// =============================================================================
// Timestamp validation
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TimestampValidation {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub anomalies_count: usize,
    pub total_rows: usize,
}

// =============================================================================
// Data Cleaning v2 — Two-path architecture (Classical vs AI-based)
// =============================================================================
// Spec reference: "Spécification technique complète — Module Data Cleaning",
// sections 2 (Two paths), 5 (Types), 6 (Classical), 7 (AI-based).
// Scientific context: Wang & Renninger 2025 (SapFlower, MATLAB) → reimplemented
// and extended here for the TTD (Do & Rocheteau 2002) method instead of CTD.

/// Which cleaning path the user has selected. Enforced exclusively at runtime.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CleaningPath {
    /// Deterministic statistical cleaning (no ML, no env data required).
    Classical,
    /// Supervised ML/DL cleaning using environmental predictors.
    AIBased,
}

impl CleaningPath {
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Classical => "Nettoyage classique",
            Self::AIBased => "Nettoyage par IA",
        }
    }
}

/// Where in the TTD pipeline the cleaning is applied.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CleaningStage {
    /// Before Tslope, on SF_* and TC_* raw sensor columns.
    RawData,
    /// After the T600 filter step (recommended default — regular 30-min grid,
    /// scientifically meaningful, aligned with env data for joins).
    T600,
    /// Run both passes in sequence: raw cleaning → pipeline → T600 cleaning.
    Both,
}

impl Default for CleaningStage {
    fn default() -> Self {
        Self::T600
    }
}

/// How outlier values / gaps are replaced after detection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplaceStrategy {
    /// Replace with NaN and leave the subsequent pipeline / gap-filling step
    /// to handle the missing values.
    MarkNaN,
    /// Linearly interpolate between nearest finite neighbours.
    LinearInterpolate,
    /// Propagate the last finite value forward.
    ForwardFill,
    /// Replace with a centred rolling mean of `window_size` points.
    RollingMean(usize),
}

impl Default for ReplaceStrategy {
    fn default() -> Self {
        Self::LinearInterpolate
    }
}

// ----------------------------------------------------------------------------
// Path A configs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IqrConfig {
    pub k: f64,
    pub target_columns: Vec<String>,
    pub replace_strategy: ReplaceStrategy,
}

impl Default for IqrConfig {
    fn default() -> Self {
        Self {
            k: 1.5,
            target_columns: Vec::new(),
            replace_strategy: ReplaceStrategy::default(),
        }
    }
}

impl IqrConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=5.0).contains(&self.k) {
            anyhow::bail!("IQR.k doit être entre 0.5 et 5.0 (reçu {})", self.k);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZScoreConfig {
    pub threshold: f64,
    pub target_columns: Vec<String>,
    pub replace_strategy: ReplaceStrategy,
}

impl Default for ZScoreConfig {
    fn default() -> Self {
        Self {
            threshold: 3.0,
            target_columns: Vec::new(),
            replace_strategy: ReplaceStrategy::default(),
        }
    }
}

impl ZScoreConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1.0..=6.0).contains(&self.threshold) {
            anyhow::bail!("ZScore.threshold doit être entre 1 et 6 (reçu {})", self.threshold);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MadConfig {
    pub threshold: f64,
    pub target_columns: Vec<String>,
    pub replace_strategy: ReplaceStrategy,
}

impl Default for MadConfig {
    fn default() -> Self {
        Self {
            threshold: 3.5,
            target_columns: Vec::new(),
            replace_strategy: ReplaceStrategy::default(),
        }
    }
}

impl MadConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1.0..=8.0).contains(&self.threshold) {
            anyhow::bail!("MAD.threshold doit être entre 1 et 8 (reçu {})", self.threshold);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollingWindowConfig {
    pub window_hours: f64,
    pub high_var_threshold: f64,
    pub low_var_threshold: f64,
    pub target_columns: Vec<String>,
}

impl Default for RollingWindowConfig {
    fn default() -> Self {
        Self {
            window_hours: 24.0,
            high_var_threshold: 0.1,
            low_var_threshold: 0.01,
            target_columns: Vec::new(),
        }
    }
}

impl RollingWindowConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1.0..=168.0).contains(&self.window_hours) {
            anyhow::bail!("window_hours doit être entre 1 et 168 (reçu {})", self.window_hours);
        }
        if self.low_var_threshold < 0.0 {
            anyhow::bail!("low_var_threshold doit être ≥ 0 (reçu {})", self.low_var_threshold);
        }
        if self.high_var_threshold <= self.low_var_threshold {
            anyhow::bail!(
                "high_var_threshold ({}) doit être > low_var_threshold ({})",
                self.high_var_threshold,
                self.low_var_threshold
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReverseDetectionConfig {
    pub num_neighbors: usize,
    pub ratio: f64,
    pub target_columns: Vec<String>,
}

impl Default for ReverseDetectionConfig {
    fn default() -> Self {
        Self {
            num_neighbors: 6,
            ratio: 0.75,
            target_columns: Vec::new(),
        }
    }
}

impl ReverseDetectionConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1..=48).contains(&self.num_neighbors) {
            anyhow::bail!("num_neighbors doit être entre 1 et 48 (reçu {})", self.num_neighbors);
        }
        if !(0.1..=1.0).contains(&self.ratio) {
            anyhow::bail!("ratio doit être entre 0.1 et 1.0 (reçu {})", self.ratio);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IsolationForestConfig {
    pub contamination: f64,
    pub n_estimators: usize,
    pub max_samples: Option<usize>,
    pub random_seed: u64,
    pub target_columns: Vec<String>,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self {
            contamination: 0.1,
            n_estimators: 100,
            max_samples: None,
            random_seed: 42,
            target_columns: Vec::new(),
        }
    }
}

impl IsolationForestConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.001..=0.5).contains(&self.contamination) {
            anyhow::bail!(
                "contamination doit être entre 0.001 et 0.5 (reçu {})",
                self.contamination
            );
        }
        if !(10..=1000).contains(&self.n_estimators) {
            anyhow::bail!(
                "n_estimators doit être entre 10 et 1000 (reçu {})",
                self.n_estimators
            );
        }
        if let Some(n) = self.max_samples {
            if !(16..=10_000).contains(&n) {
                anyhow::bail!("max_samples doit être entre 16 et 10000 (reçu {})", n);
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Path B configs (ML / DL models)
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinearConfig {
    pub target_columns: Vec<String>,
    pub predictor_columns: Vec<String>,
    /// Held-out columns the model trains on (for richer metrics) but whose
    /// values are NEVER overwritten by `clean()`. Used when the lab wants to
    /// cross-validate on specific sensors without altering them.
    #[serde(default)]
    pub test_columns: Vec<String>,
    /// Seasonal / diurnal features appended to X at build time.
    #[serde(default)]
    pub temporal_features: TemporalFeatures,
    /// Fraction of the dataset (sorted by time) used for training the model.
    /// Range is expressed as [start, end] in [0.0, 1.0]. Default = full span.
    #[serde(default)] pub train_range_start: f64,
    #[serde(default = "default_one")] pub train_range_end: f64,
    /// Multi-zone training selection. When non-empty, takes precedence over
    /// `train_range_start/end` — the cleaner trains on the union of all the
    /// listed [start, end] ranges (each in [0, 1]). Used for excluding
    /// "bad" intervals (sensor faults, irrigation peaks) without truncating
    /// the whole window.
    #[serde(default)] pub train_ranges: Vec<(f64, f64)>,
    /// Multi-zone training selection by absolute TIMESTAMP (millis since
    /// epoch, UTC). Takes precedence over `train_ranges` and the legacy
    /// pct pair. Preferred for UI selections.
    #[serde(default)] pub train_time_ranges: Vec<(i64, i64)>,
    pub train_validation_split: f64,
    pub feature_scaling: bool,
    pub log_transform_vpd: bool,
    pub random_seed: u64,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            predictor_columns: Vec::new(),
            test_columns: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.75,
            feature_scaling: true,
            log_transform_vpd: true,
            random_seed: 42,
        }
    }
}

impl LinearConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=0.95).contains(&self.train_validation_split) {
            anyhow::bail!(
                "train_validation_split doit être entre 0.5 et 0.95 (reçu {})",
                self.train_validation_split
            );
        }
        if self.target_columns.is_empty() {
            anyhow::bail!("Au moins une colonne cible requise");
        }
        if self.predictor_columns.is_empty() {
            anyhow::bail!("Au moins un prédicteur requis");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KernelType {
    Linear,
    RBF,
    Matern32,
    Matern52,
    Polynomial,
}

// ----------------------------------------------------------------------------
// Temporal features — shared across all ML cleaning methods
// ----------------------------------------------------------------------------
//
// Rationale: sap flow follows a strong diurnal + seasonal pattern. Giving the
// model explicit time-of-day / day-of-year features helps it generalize
// without trying to infer the cycle from scratch.
//
// We support two encodings per dimension:
//   - one-hot     : N binary columns (24 for hour, 366 for doy, 12 for month).
//                   Exact but feature-heavy — best for trees.
//   - cyclical    : 2 columns (sin, cos) encoding the angle. Preserves
//                   periodicity (hour 23 is close to hour 0) and keeps the
//                   feature count tiny — best for linear / RNN models.
//
// Defaults: cyclical_hour + cyclical_doy ON (the two most informative rhythms
// for TTD sap flow, per Oishi 2016 and Wang 2025). Everything else OFF.

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TemporalFeatures {
    /// 24 one-hot columns for hour-of-day (0–23).
    #[serde(default)]
    pub one_hot_hour: bool,
    /// 2 columns: sin(2π·h/24), cos(2π·h/24).
    #[serde(default = "default_true")]
    pub cyclical_hour: bool,
    /// 366 one-hot columns for day-of-year.
    #[serde(default)]
    pub one_hot_doy: bool,
    /// 2 columns: sin(2π·d/366), cos(2π·d/366) — captures seasonality.
    #[serde(default = "default_true")]
    pub cyclical_doy: bool,
    /// 12 one-hot columns for month.
    #[serde(default)]
    pub one_hot_month: bool,
    /// 2 columns: sin(2π·m/12), cos(2π·m/12).
    #[serde(default)]
    pub cyclical_month: bool,
    /// 2 one-hot columns for 30-min step inside the hour (0 vs 30).
    #[serde(default)]
    pub one_hot_minute: bool,
    /// 1 column: days since the first timestamp — lets the model capture
    /// secular drift (sensor ageing, bark growth).
    #[serde(default)]
    pub days_since_start: bool,
}

fn default_true() -> bool { true }
fn default_one() -> f64 { 1.0 }

impl Default for TemporalFeatures {
    fn default() -> Self {
        Self {
            one_hot_hour: false,
            cyclical_hour: true,
            one_hot_doy: false,
            cyclical_doy: true,
            one_hot_month: false,
            cyclical_month: false,
            one_hot_minute: false,
            days_since_start: false,
        }
    }
}

impl TemporalFeatures {
    /// Number of extra columns this configuration adds to X.
    pub fn extra_cols(&self) -> usize {
        (if self.one_hot_hour { 24 } else { 0 })
            + (if self.cyclical_hour { 2 } else { 0 })
            + (if self.one_hot_doy { 366 } else { 0 })
            + (if self.cyclical_doy { 2 } else { 0 })
            + (if self.one_hot_month { 12 } else { 0 })
            + (if self.cyclical_month { 2 } else { 0 })
            + (if self.one_hot_minute { 2 } else { 0 })
            + (if self.days_since_start { 1 } else { 0 })
    }
}

impl Default for KernelType {
    fn default() -> Self {
        Self::RBF
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MLConfig {
    pub target_columns: Vec<String>,
    pub predictor_columns: Vec<String>,
    /// See `LinearConfig::test_columns`.
    #[serde(default)]
    pub test_columns: Vec<String>,
    #[serde(default)]
    pub temporal_features: TemporalFeatures,
    /// Fraction of the dataset (sorted by time) used for training the model.
    /// Range is expressed as [start, end] in [0.0, 1.0]. Default = full span.
    #[serde(default)] pub train_range_start: f64,
    #[serde(default = "default_one")] pub train_range_end: f64,
    /// Multi-zone training selection. See `LinearConfig::train_ranges`.
    #[serde(default)] pub train_ranges: Vec<(f64, f64)>,
    /// Multi-zone training selection by absolute TIMESTAMP (millis since
    /// epoch, UTC). When non-empty, takes precedence over `train_ranges`
    /// and `train_range_start/end`. Preferred for UI selections — avoids
    /// the row-index pct ambiguity when previewRows is a slice of a larger
    /// dataset.
    #[serde(default)] pub train_time_ranges: Vec<(i64, i64)>,
    pub train_validation_split: f64,
    pub feature_scaling: bool,
    pub random_seed: u64,
    pub n_estimators: Option<usize>,
    pub max_depth: Option<usize>,
    pub kernel: Option<KernelType>,
    pub svr_c: Option<f64>,
    pub svr_epsilon: Option<f64>,
    /// Number of principal components kept by the PCA (ACP) method. None / 0 =
    /// use all features (no reduction). Ignored by the other methods.
    #[serde(default)]
    pub n_components: Option<usize>,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            predictor_columns: Vec::new(),
            test_columns: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.75,
            feature_scaling: true,
            random_seed: 42,
            n_estimators: None,
            max_depth: None,
            kernel: None,
            svr_c: None,
            svr_epsilon: None,
            n_components: None,
        }
    }
}

impl MLConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=0.95).contains(&self.train_validation_split) {
            anyhow::bail!("train_validation_split hors plage [0.5, 0.95]");
        }
        if self.target_columns.is_empty() {
            anyhow::bail!("Au moins une colonne cible requise");
        }
        if self.predictor_columns.is_empty() {
            anyhow::bail!("Au moins un prédicteur requis");
        }
        if let Some(n) = self.n_estimators {
            if !(10..=500).contains(&n) {
                anyhow::bail!("n_estimators doit être entre 10 et 500");
            }
        }
        if let Some(d) = self.max_depth {
            if !(1..=50).contains(&d) {
                anyhow::bail!("max_depth doit être entre 1 et 50");
            }
        }
        if let Some(c) = self.svr_c {
            if !(0.01..=1000.0).contains(&c) {
                anyhow::bail!("svr_c doit être entre 0.01 et 1000");
            }
        }
        if let Some(eps) = self.svr_epsilon {
            if !(0.0..=1.0).contains(&eps) {
                anyhow::bail!("svr_epsilon doit être entre 0 et 1");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ARXConfig {
    pub target_columns: Vec<String>,
    pub predictor_columns: Vec<String>,
    /// See `LinearConfig::test_columns`.
    #[serde(default)]
    pub test_columns: Vec<String>,
    #[serde(default)]
    pub temporal_features: TemporalFeatures,
    /// Fraction of the dataset (sorted by time) used for training the model.
    /// Range is expressed as [start, end] in [0.0, 1.0]. Default = full span.
    #[serde(default)] pub train_range_start: f64,
    #[serde(default = "default_one")] pub train_range_end: f64,
    /// Multi-zone training selection. See `LinearConfig::train_ranges`.
    #[serde(default)] pub train_ranges: Vec<(f64, f64)>,
    /// Multi-zone training selection by absolute TIMESTAMP (millis since
    /// epoch, UTC). When non-empty, takes precedence over `train_ranges`
    /// and `train_range_start/end`. Preferred for UI selections — avoids
    /// the row-index pct ambiguity when previewRows is a slice of a larger
    /// dataset.
    #[serde(default)] pub train_time_ranges: Vec<(i64, i64)>,
    pub train_validation_split: f64,
    pub ar_order: usize,
    pub exog_order: usize,
    pub random_seed: u64,
}

impl Default for ARXConfig {
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            predictor_columns: Vec::new(),
            test_columns: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.75,
            ar_order: 4,
            exog_order: 2,
            random_seed: 42,
        }
    }
}

impl ARXConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=0.95).contains(&self.train_validation_split) {
            anyhow::bail!("train_validation_split hors plage [0.5, 0.95]");
        }
        if !(1..=24).contains(&self.ar_order) {
            anyhow::bail!("ar_order doit être entre 1 et 24");
        }
        if !(0..=12).contains(&self.exog_order) {
            anyhow::bail!("exog_order doit être entre 0 et 12");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ARMAXConfig {
    pub target_columns: Vec<String>,
    pub predictor_columns: Vec<String>,
    /// See `LinearConfig::test_columns`.
    #[serde(default)]
    pub test_columns: Vec<String>,
    #[serde(default)]
    pub temporal_features: TemporalFeatures,
    /// Fraction of the dataset (sorted by time) used for training the model.
    /// Range is expressed as [start, end] in [0.0, 1.0]. Default = full span.
    #[serde(default)] pub train_range_start: f64,
    #[serde(default = "default_one")] pub train_range_end: f64,
    /// Multi-zone training selection. See `LinearConfig::train_ranges`.
    #[serde(default)] pub train_ranges: Vec<(f64, f64)>,
    /// Multi-zone training selection by absolute TIMESTAMP (millis since
    /// epoch, UTC). When non-empty, takes precedence over `train_ranges`
    /// and `train_range_start/end`. Preferred for UI selections — avoids
    /// the row-index pct ambiguity when previewRows is a slice of a larger
    /// dataset.
    #[serde(default)] pub train_time_ranges: Vec<(i64, i64)>,
    pub train_validation_split: f64,
    pub ar_order: usize,
    pub exog_order: usize,
    pub ma_order: usize,
    pub random_seed: u64,
}

impl Default for ARMAXConfig {
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            predictor_columns: Vec::new(),
            test_columns: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.75,
            ar_order: 4,
            exog_order: 2,
            ma_order: 2,
            random_seed: 42,
        }
    }
}

impl ARMAXConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=0.95).contains(&self.train_validation_split) {
            anyhow::bail!("train_validation_split hors plage [0.5, 0.95]");
        }
        if !(1..=24).contains(&self.ar_order) {
            anyhow::bail!("ar_order doit être entre 1 et 24");
        }
        if !(0..=12).contains(&self.exog_order) {
            anyhow::bail!("exog_order doit être entre 0 et 12");
        }
        if !(0..=12).contains(&self.ma_order) {
            anyhow::bail!("ma_order doit être entre 0 et 12");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGDMomentum,
    RMSProp,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adam
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DLConfig {
    pub target_columns: Vec<String>,
    pub predictor_columns: Vec<String>,
    /// See `LinearConfig::test_columns`.
    #[serde(default)]
    pub test_columns: Vec<String>,
    /// Training zone as [start, end] in [0, 1]. Default full span.
    #[serde(default)] pub train_range_start: f64,
    #[serde(default = "default_one")] pub train_range_end: f64,
    /// Multi-zone training selection. See `LinearConfig::train_ranges`.
    #[serde(default)] pub train_ranges: Vec<(f64, f64)>,
    /// Multi-zone training selection by absolute TIMESTAMP (millis since
    /// epoch, UTC). When non-empty, takes precedence over `train_ranges`
    /// and `train_range_start/end`. Preferred for UI selections — avoids
    /// the row-index pct ambiguity when previewRows is a slice of a larger
    /// dataset.
    #[serde(default)] pub train_time_ranges: Vec<(i64, i64)>,
    pub train_validation_split: f64,
    pub feature_scaling: bool,
    pub random_seed: u64,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub hidden_units: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub gradient_clip: Option<f64>,
    pub optimizer: OptimizerType,
    pub sequence_length: usize,
    /// Shared temporal feature flags (replaces the old inline one_hot_*).
    #[serde(default)]
    pub temporal_features: TemporalFeatures,
}

impl Default for DLConfig {
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            predictor_columns: Vec::new(),
            test_columns: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            train_range_start: 0.0,
            train_range_end: 1.0,
            train_ranges: Vec::new(),
            train_time_ranges: Vec::new(),
            train_validation_split: 0.75,
            feature_scaling: true,
            random_seed: 42,
            epochs: 200,
            batch_size: 32,
            learning_rate: 1e-3,
            hidden_units: 30,
            num_layers: 1,
            dropout: 0.0,
            gradient_clip: Some(1.0),
            optimizer: OptimizerType::default(),
            sequence_length: 48,
        }
    }
}

impl DLConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.5..=0.95).contains(&self.train_validation_split) {
            anyhow::bail!("train_validation_split hors plage [0.5, 0.95]");
        }
        if !(1..=10_000).contains(&self.epochs) {
            anyhow::bail!("epochs doit être entre 1 et 10 000");
        }
        if !(1..=512).contains(&self.batch_size) {
            anyhow::bail!("batch_size doit être entre 1 et 512");
        }
        if !(1e-6..=1.0).contains(&self.learning_rate) {
            anyhow::bail!("learning_rate doit être entre 1e-6 et 1");
        }
        if !(1..=1024).contains(&self.hidden_units) {
            anyhow::bail!("hidden_units doit être entre 1 et 1024");
        }
        if !(1..=8).contains(&self.num_layers) {
            anyhow::bail!("num_layers doit être entre 1 et 8");
        }
        if !(0.0..=0.8).contains(&self.dropout) {
            anyhow::bail!("dropout doit être entre 0.0 et 0.8");
        }
        if let Some(gc) = self.gradient_clip {
            if !(0.1..=10.0).contains(&gc) {
                anyhow::bail!("gradient_clip doit être entre 0.1 et 10");
            }
        }
        if !(4..=512).contains(&self.sequence_length) {
            anyhow::bail!("sequence_length doit être entre 4 et 512");
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Unified CleaningMethod enum
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CleaningMethod {
    // Path A — classical
    AbsoluteBounds { min: f64, max: f64, target_columns: Vec<String> },
    Iqr(IqrConfig),
    ZScore(ZScoreConfig),
    Mad(MadConfig),
    RollingWindow(RollingWindowConfig),
    ReverseDetection(ReverseDetectionConfig),
    IsolationForest(IsolationForestConfig),

    // Path B — AI
    SimpleLinear(LinearConfig),
    MultipleLinear(LinearConfig),
    RandomForest(MLConfig),
    Svr(MLConfig),
    Gpr(MLConfig),
    /// Principal Component Regression (ACP / PCA).
    Pca(MLConfig),
    Arx(ARXConfig),
    Armax(ARMAXConfig),
    Lstm(DLConfig),
    BiLstm(DLConfig),
    Gru(DLConfig),

    /// Ordered composition of classical methods. AI methods are not allowed
    /// inside a pipeline to keep train/validate semantics unambiguous.
    ClassicalPipeline(Vec<CleaningMethod>),
}

impl CleaningMethod {
    pub fn path(&self) -> CleaningPath {
        match self {
            Self::AbsoluteBounds { .. }
            | Self::Iqr(_)
            | Self::ZScore(_)
            | Self::Mad(_)
            | Self::RollingWindow(_)
            | Self::ReverseDetection(_)
            | Self::IsolationForest(_)
            | Self::ClassicalPipeline(_) => CleaningPath::Classical,
            Self::SimpleLinear(_)
            | Self::MultipleLinear(_)
            | Self::RandomForest(_)
            | Self::Svr(_)
            | Self::Gpr(_)
            | Self::Pca(_)
            | Self::Arx(_)
            | Self::Armax(_)
            | Self::Lstm(_)
            | Self::BiLstm(_)
            | Self::Gru(_) => CleaningPath::AIBased,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AbsoluteBounds { .. } => "AbsoluteBounds",
            Self::Iqr(_) => "IQR",
            Self::ZScore(_) => "ZScore",
            Self::Mad(_) => "MAD",
            Self::RollingWindow(_) => "RollingWindow",
            Self::ReverseDetection(_) => "ReverseDetection",
            Self::IsolationForest(_) => "IsolationForest",
            Self::SimpleLinear(_) => "SimpleLinear",
            Self::MultipleLinear(_) => "MultipleLinear",
            Self::RandomForest(_) => "RandomForest",
            Self::Svr(_) => "SVR",
            Self::Gpr(_) => "GPR",
            Self::Pca(_) => "PCA",
            Self::Arx(_) => "ARX",
            Self::Armax(_) => "ARMAX",
            Self::Lstm(_) => "LSTM",
            Self::BiLstm(_) => "BiLSTM",
            Self::Gru(_) => "GRU",
            Self::ClassicalPipeline(_) => "ClassicalPipeline",
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::AbsoluteBounds { min, max, .. } => {
                if !(min < max) {
                    anyhow::bail!("AbsoluteBounds: min ({}) doit être < max ({})", min, max);
                }
                Ok(())
            }
            Self::Iqr(c) => c.validate(),
            Self::ZScore(c) => c.validate(),
            Self::Mad(c) => c.validate(),
            Self::RollingWindow(c) => c.validate(),
            Self::ReverseDetection(c) => c.validate(),
            Self::IsolationForest(c) => c.validate(),
            Self::SimpleLinear(c) | Self::MultipleLinear(c) => c.validate(),
            Self::RandomForest(c) | Self::Svr(c) | Self::Gpr(c) | Self::Pca(c) => c.validate(),
            Self::Arx(c) => c.validate(),
            Self::Armax(c) => c.validate(),
            Self::Lstm(c) | Self::BiLstm(c) | Self::Gru(c) => c.validate(),
            Self::ClassicalPipeline(methods) => {
                if methods.is_empty() {
                    anyhow::bail!("ClassicalPipeline vide");
                }
                for m in methods {
                    if m.path() != CleaningPath::Classical {
                        anyhow::bail!(
                            "ClassicalPipeline ne peut contenir que des méthodes classiques (trouvé {})",
                            m.name()
                        );
                    }
                    if matches!(m, Self::ClassicalPipeline(_)) {
                        anyhow::bail!("ClassicalPipeline ne peut être imbriquée");
                    }
                    m.validate()?;
                }
                Ok(())
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Cleaning reports & training metrics (prefixed to avoid collision with the
// existing `ColumnStats` for import quality and `TrainingMetrics` for ML).
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CleaningColumnReport {
    pub n_raw: usize,
    pub n_nan_raw: usize,
    pub n_outliers: usize,
    pub n_replaced: usize,
    pub n_nan_final: usize,
    pub mean_raw: f64,
    pub mean_cleaned: f64,
    pub std_raw: f64,
    pub std_cleaned: f64,
    pub min_raw: f64,
    pub max_raw: f64,
    pub min_cleaned: f64,
    pub max_cleaned: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CleaningReport {
    pub n_total: usize,
    pub n_outliers_detected: usize,
    pub n_outliers_replaced: usize,
    pub n_gaps_filled: usize,
    pub pct_data_modified: f64,
    pub per_column: HashMap<String, CleaningColumnReport>,
    pub method_chain: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PerTargetMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
    /// RMSE as percentage of mean(observed) — publication target: ≤ 10 %.
    pub rmse_pct_of_mean: f64,
}

/// Richer metrics bundle for cleaning Voie B (AIC/BIC/timings).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CleaningTrainingMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub training_time_s: f64,
    pub prediction_time_s: f64,
    pub n_train_points: usize,
    pub n_val_points: usize,
    pub n_parameters: usize,
    pub per_target: HashMap<String, PerTargetMetrics>,
}
