use std::collections::HashMap;
use std::sync::Mutex;

use polars::prelude::*;

use crate::core::types::*;
use crate::ml::engine::ModelCache;

pub struct AppState {
    pub inner: Mutex<AppData>,
}

/// A SAITS gap-fill result computed but NOT yet written to its dataset slot.
/// The Gap Filling page previews it and the user decides (cleaning_commit_
/// completion / cleaning_discard_completion) whether to apply it to the
/// principal dataset.
pub struct PendingCompletion {
    pub dataset_key: String,
    pub df: DataFrame,
    pub per_column: Vec<(String, usize)>,
    pub n_total: usize,
}

/// A file opened for visualisation only (see `AppData::viz_files`).
pub struct VizFile {
    /// Human label shown in the source picker — the file name.
    pub label: String,
    /// Full path, surfaced as the source's `origin` line.
    pub path: String,
    pub df: DataFrame,
}

pub struct AppData {
    pub raw_data: Option<DataFrame>,
    pub cleaned_data: Option<DataFrame>,
    pub results: CalculationResults,
    pub config: ProjectConfig,
    pub sap_flow_params: SapFlowParams,
    pub ttdplus_params: TtdPlusParams,
    pub logs: Vec<LogEntry>,
    #[allow(dead_code)]
    pub scenarios: Vec<Scenario>,
    pub columns: Vec<String>,
    pub file_path: Option<String>,
    pub sheet_name: Option<String>,
    pub trained_models: HashMap<String, TrainedModelInfo>,
    pub model_cache: HashMap<String, ModelCache>,
    pub detection_results: Option<DetectionState>,
    /// Environmental data (VPD, PAR, …) loaded separately from the main
    /// raw_data. Used by the Tm "DataEnv" method to identify no-flow periods
    /// and by the Voie B ML cleaner as predictor source.
    pub env_data: Option<DataFrame>,
    pub env_file_path: Option<String>,
    pub env_sheet_name: Option<String>,
    /// Trained ML-based cleaner (Voie B). Set by `cleaning_train_ml`,
    /// consumed (then reinserted) by `cleaning_apply_ml`.
    pub ml_cleaner: Option<crate::core::ml_cleaning::MLCleaner>,
    /// Provenance of `cleaned_data`: which source dataset it came from
    /// (e.g. "raw", "t600", "tslope") + which path produced it ("classical",
    /// "ml", "gapfill_classical"). Surfaced to the UI via `get_cleaning_info`
    /// so the Gap Filling page can show "the cleaning was done on T600".
    pub cleaning_source_dataset: Option<String>,
    pub cleaning_method_label: Option<String>,
    pub cleaning_path: Option<String>,
    /// Columns that were touched by the most recent cleaning operation.
    /// Lets the UI show clickable chips so the user can re-select them in
    /// the gap-fill picker without having to remember the names.
    pub cleaning_target_columns: Vec<String>,
    /// Derived stages ("t600", "tm", …) the user explicitly LOCKED via
    /// "Rendre permanent" after a manual/algo cleaning. A full `run_pipeline`
    /// rebuilds every derived slot from raw and would silently wipe that work,
    /// so it refuses to overwrite a locked stage unless called with
    /// `force = true`. Unlike `cleaning_source_dataset` (a single, volatile
    /// provenance pointer) this set is intentional, additive, and persisted
    /// across sessions. Cleared by a forced full run, a reset, or a new import.
    pub cleaning_locked_stages: Vec<String>,
    /// One-shot pre-cleaning snapshot per dataset key. Captured the FIRST
    /// time we touch a slot so the UI can render a "before / after"
    /// comparison even after the slot has been overwritten in-place.
    /// Cleared by `cleaning_reset_to_raw`.
    pub cleaning_pre_snapshots: HashMap<String, DataFrame>,
    /// SAITS gap-fill computed but awaiting the user's apply/discard choice.
    pub pending_completion: Option<PendingCompletion>,
    /// User-defined groups for the Calculs avancés Jh→Jhp→Qh→Qd chain.
    /// Each entry pins together a group name, its sensors, and the two
    /// scalar coefficients (k_radial, A_sapwood_dm2). Persisted across
    /// sessions so the user doesn't re-compose pools every restart.
    pub advanced_groups: Vec<AdvancedGroup>,
    /// Ad-hoc files opened purely to LOOK at them on the Visualisation tab —
    /// a previously exported T600 / sap-flow CSV, another station's export, …
    /// Keyed by a generated id; exposed by `viz_list_sources` as `file:<id>`.
    ///
    /// Deliberately separate from `raw_data`: the normal import replaces the
    /// working data and wipes every pipeline result, which is far too
    /// destructive for "I just want to see this file next to mine". In-memory
    /// only, like scenario overlays — the user re-opens what they need.
    pub viz_files: HashMap<String, VizFile>,
    /// Scenario "overlays" loaded for side-by-side comparison on the
    /// Visualisation tab. Each entry exposes its DataFrames as additional
    /// sources via `viz_list_sources` (keyed `overlay:<id>:<dataset>`).
    /// In-memory only — the user re-toggles overlays at each session.
    pub loaded_scenario_overlays: HashMap<String, ScenarioOverlay>,
}

impl Default for AppData {
    fn default() -> Self {
        Self {
            raw_data: None,
            cleaned_data: None,
            results: CalculationResults::default(),
            config: ProjectConfig::default(),
            sap_flow_params: SapFlowParams::default(),
            ttdplus_params: TtdPlusParams::default(),
            logs: Vec::new(),
            scenarios: Vec::new(),
            columns: Vec::new(),
            file_path: None,
            sheet_name: None,
            trained_models: HashMap::new(),
            model_cache: HashMap::new(),
            detection_results: None,
            env_data: None,
            env_file_path: None,
            env_sheet_name: None,
            ml_cleaner: None,
            cleaning_source_dataset: None,
            cleaning_method_label: None,
            cleaning_path: None,
            cleaning_target_columns: Vec::new(),
            cleaning_locked_stages: Vec::new(),
            cleaning_pre_snapshots: HashMap::new(),
            pending_completion: None,
            advanced_groups: Vec::new(),
            viz_files: HashMap::new(),
            loaded_scenario_overlays: HashMap::new(),
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        // Re-hydrate from disk if a previous session was persisted
        // (Windows sleep / cargo restart / unexpected exit). Falls back
        // to AppData::default() when nothing is on disk.
        Self {
            inner: Mutex::new(crate::utils::session_persist::load()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
