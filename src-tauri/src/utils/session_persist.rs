//! On-disk session persistence.
//!
//! Tauri keeps `AppData` (DataFrames, env data, pipeline outputs, cleaning
//! provenance, …) in a `Mutex<…>` that lives in process memory. When
//! Windows suspends the dev server during sleep — or kills the process
//! outright on resume — that state evaporates and the user is forced to
//! re-import every file. This module mirrors the important parts of
//! `AppData` to disk after each state-changing command, and re-loads them
//! on startup so the session survives unexpected restarts.
//!
//! Layout under `<app_data_dir>/ttdprocess/session/`:
//!   raw_data.parquet              Raw imported data (Polars)
//!   cleaned_data.parquet          Post-cleaning version
//!   env_data.parquet              Env / VPD / PAR file
//!   results/{slot}.parquet        Pipeline outputs (tslope, baseline, …)
//!   cleaning_pre_snapshots/{key}.parquet   Before-snapshots for the UI
//!   meta.json                     Metadata: file paths, sap-flow params,
//!                                  cleaning provenance, etc. (all the
//!                                  small fields that aren't a DataFrame)
//!
//! Trade-offs:
//!   - We do NOT persist the trained ML model (`Box<dyn Predictor>` is not
//!     serializable without ad-hoc work per model). The user re-trains.
//!   - We do NOT persist `logs` (append-only, low value cross-session).
//!   - Saves are best-effort: if the disk write fails, the in-memory state
//!     is still authoritative — we just lose the persistence guarantee
//!     for that change. Errors are logged but don't propagate to the user.

use std::fs;
use std::path::{Path, PathBuf};

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::types::{ProjectConfig, SapFlowParams, TtdPlusParams};
use crate::state::AppData;

const SESSION_DIRNAME: &str = "session";

/// Resolve the session root directory under the OS user-data dir. Falls
/// back to a `./session/` next to the executable if anything else fails —
/// the worst case is non-portable (per-machine) but never crashes.
pub fn session_dir() -> PathBuf {
    if let Some(dirs) = dirs_app_data() {
        return dirs.join("ttdprocess").join(SESSION_DIRNAME);
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(SESSION_DIRNAME)
}

/// `dirs::data_dir()` without pulling in the full crate. Windows: `%APPDATA%`.
fn dirs_app_data() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMeta {
    pub file_path: Option<String>,
    pub sheet_name: Option<String>,
    pub env_file_path: Option<String>,
    pub env_sheet_name: Option<String>,
    pub cleaning_source_dataset: Option<String>,
    pub cleaning_method_label: Option<String>,
    pub cleaning_path: Option<String>,
    #[serde(default)]
    pub cleaning_target_columns: Vec<String>,
    /// Stages the user pinned via "Rendre permanent" (see AppData). Persisted
    /// so a lock survives a Windows-sleep restart or a cargo rebuild.
    #[serde(default)]
    pub cleaning_locked_stages: Vec<String>,
    pub sap_flow_params: SapFlowParams,
    pub ttdplus_params: TtdPlusParams,
    pub config: ProjectConfig,
    /// User-defined groups for the Calculs avancés Jh→Jhp→Qh→Qd chain.
    #[serde(default)]
    pub advanced_groups: Vec<crate::core::types::AdvancedGroup>,
}

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

/// Persist the entire app state. Best-effort: any individual error is
/// logged via `eprintln!` (visible in the dev console) but doesn't stop
/// the others — a half-saved session is still better than nothing.
pub fn save(data: &AppData) {
    let root = session_dir();
    if let Err(e) = fs::create_dir_all(&root) {
        eprintln!("session_persist: cannot create {:?}: {}", root, e);
        return;
    }

    let try_write_df = |slot: &Option<DataFrame>, name: &str| {
        if let Some(df) = slot {
            let path = root.join(name);
            if let Err(e) = write_parquet(df, &path) {
                eprintln!("session_persist: failed to write {}: {}", name, e);
            }
        } else {
            // Slot is None — remove any stale parquet from a previous run
            // so a load doesn't resurrect cleared data.
            let _ = fs::remove_file(root.join(name));
        }
    };

    try_write_df(&data.raw_data, "raw_data.parquet");
    try_write_df(&data.cleaned_data, "cleaned_data.parquet");
    try_write_df(&data.env_data, "env_data.parquet");

    // results.* — each gets its own file under results/.
    let res_dir = root.join("results");
    if let Err(e) = fs::create_dir_all(&res_dir) {
        eprintln!("session_persist: cannot create {:?}: {}", res_dir, e);
    }
    let r = &data.results;
    let result_slots: [(&Option<DataFrame>, &str); 17] = [
        (&r.tslope, "tslope.parquet"),
        (&r.baseline, "baseline.parquet"),
        (&r.delta_t, "delta_t.parquet"),
        (&r.t600, "t600.parquet"),
        (&r.tm, "tm.parquet"),
        (&r.stm, "stm.parquet"),
        (&r.tmi, "tmi.parquet"),
        (&r.k, "k.parquet"),
        (&r.sap_flow, "sap_flow.parquet"),
        (&r.ttdplus_fourier, "ttdplus_fourier.parquet"),
        (&r.ttdplus_refs, "ttdplus_refs.parquet"),
        (&r.rd_regression, "rd_regression.parquet"),
        (&r.rd_result, "rd_result.parquet"),
        (&r.jh, "jh.parquet"),
        (&r.jhp, "jhp.parquet"),
        (&r.qh, "qh.parquet"),
        (&r.qd, "qd.parquet"),
    ];
    for (slot, name) in &result_slots {
        let path = res_dir.join(name);
        if let Some(df) = slot {
            if let Err(e) = write_parquet(df, &path) {
                eprintln!("session_persist: failed to write {}: {}", name, e);
            }
        } else {
            let _ = fs::remove_file(&path);
        }
    }
    // ttdplus_sap_flow has the same shape as the others; written separately
    // because the array literal above is fixed-size.
    let path = res_dir.join("ttdplus_sap_flow.parquet");
    if let Some(df) = &r.ttdplus_sap_flow {
        let _ = write_parquet(df, &path);
    } else {
        let _ = fs::remove_file(&path);
    }

    // Pre-cleaning snapshots — keyed by dataset name so the Gap Filling
    // tab can show before/after.
    let snap_dir = root.join("cleaning_pre_snapshots");
    if let Err(e) = fs::create_dir_all(&snap_dir) {
        eprintln!("session_persist: cannot create {:?}: {}", snap_dir, e);
    }
    // Wipe stale files first — keys may have changed since last save.
    if let Ok(entries) = fs::read_dir(&snap_dir) {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
        }
    }
    for (key, df) in &data.cleaning_pre_snapshots {
        let path = snap_dir.join(format!("{}.parquet", key));
        if let Err(e) = write_parquet(df, &path) {
            eprintln!("session_persist: failed to write snapshot {}: {}", key, e);
        }
    }

    // Meta.json — small structured fields.
    let meta = SessionMeta {
        file_path: data.file_path.clone(),
        sheet_name: data.sheet_name.clone(),
        env_file_path: data.env_file_path.clone(),
        env_sheet_name: data.env_sheet_name.clone(),
        cleaning_source_dataset: data.cleaning_source_dataset.clone(),
        cleaning_method_label: data.cleaning_method_label.clone(),
        cleaning_path: data.cleaning_path.clone(),
        cleaning_target_columns: data.cleaning_target_columns.clone(),
        cleaning_locked_stages: data.cleaning_locked_stages.clone(),
        sap_flow_params: data.sap_flow_params.clone(),
        ttdplus_params: data.ttdplus_params.clone(),
        config: data.config.clone(),
        advanced_groups: data.advanced_groups.clone(),
    };
    match serde_json::to_string_pretty(&meta) {
        Ok(s) => {
            if let Err(e) = fs::write(root.join("meta.json"), s) {
                eprintln!("session_persist: cannot write meta.json: {}", e);
            }
        }
        Err(e) => eprintln!("session_persist: cannot serialize meta: {}", e),
    }
}

/// Try to restore everything from disk. Returns a fresh-default `AppData`
/// if the session dir doesn't exist or is empty. Per-file errors are
/// logged and skipped — we'd rather load a partial session than refuse
/// the whole thing because one snapshot was corrupted.
pub fn load() -> AppData {
    let root = session_dir();
    if !root.exists() {
        return AppData::default();
    }

    let mut data = AppData::default();

    let try_read_df = |name: &str| -> Option<DataFrame> {
        let path = root.join(name);
        if !path.exists() { return None; }
        match read_parquet(&path) {
            Ok(df) => Some(df),
            Err(e) => {
                eprintln!("session_persist: failed to read {}: {}", name, e);
                None
            }
        }
    };

    data.raw_data = try_read_df("raw_data.parquet");
    data.cleaned_data = try_read_df("cleaned_data.parquet");
    data.env_data = try_read_df("env_data.parquet");

    let res_dir = root.join("results");
    let try_read_res = |name: &str| -> Option<DataFrame> {
        let path = res_dir.join(name);
        if !path.exists() { return None; }
        match read_parquet(&path) {
            Ok(df) => Some(df),
            Err(e) => {
                eprintln!("session_persist: failed to read results/{}: {}", name, e);
                None
            }
        }
    };
    data.results.tslope = try_read_res("tslope.parquet");
    data.results.baseline = try_read_res("baseline.parquet");
    data.results.delta_t = try_read_res("delta_t.parquet");
    data.results.t600 = try_read_res("t600.parquet");
    data.results.tm = try_read_res("tm.parquet");
    data.results.stm = try_read_res("stm.parquet");
    data.results.tmi = try_read_res("tmi.parquet");
    data.results.k = try_read_res("k.parquet");
    data.results.sap_flow = try_read_res("sap_flow.parquet");
    data.results.ttdplus_fourier = try_read_res("ttdplus_fourier.parquet");
    data.results.ttdplus_refs = try_read_res("ttdplus_refs.parquet");
    data.results.ttdplus_sap_flow = try_read_res("ttdplus_sap_flow.parquet");
    data.results.rd_regression = try_read_res("rd_regression.parquet");
    data.results.rd_result = try_read_res("rd_result.parquet");
    data.results.jh = try_read_res("jh.parquet");
    data.results.jhp = try_read_res("jhp.parquet");
    data.results.qh = try_read_res("qh.parquet");
    data.results.qd = try_read_res("qd.parquet");

    let snap_dir = root.join("cleaning_pre_snapshots");
    if snap_dir.exists() {
        if let Ok(entries) = fs::read_dir(&snap_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) != Some("parquet") { continue; }
                let key = p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string());
                if let Some(key) = key {
                    if let Ok(df) = read_parquet(&p) {
                        data.cleaning_pre_snapshots.insert(key, df);
                    }
                }
            }
        }
    }

    let meta_path = root.join("meta.json");
    if meta_path.exists() {
        match fs::read_to_string(&meta_path) {
            Ok(s) => match serde_json::from_str::<SessionMeta>(&s) {
                Ok(m) => {
                    data.file_path = m.file_path;
                    data.sheet_name = m.sheet_name;
                    data.env_file_path = m.env_file_path;
                    data.env_sheet_name = m.env_sheet_name;
                    data.cleaning_source_dataset = m.cleaning_source_dataset;
                    data.cleaning_method_label = m.cleaning_method_label;
                    data.cleaning_path = m.cleaning_path;
                    data.cleaning_target_columns = m.cleaning_target_columns;
                    data.cleaning_locked_stages = m.cleaning_locked_stages;
                    data.sap_flow_params = m.sap_flow_params;
                    data.ttdplus_params = m.ttdplus_params;
                    data.config = m.config;
                    data.advanced_groups = m.advanced_groups;
                }
                Err(e) => eprintln!("session_persist: invalid meta.json: {}", e),
            },
            Err(e) => eprintln!("session_persist: cannot read meta.json: {}", e),
        }
    }

    data
}

/// Wipe the entire session directory. Best-effort.
pub fn clear() {
    let root = session_dir();
    if root.exists() {
        if let Err(e) = fs::remove_dir_all(&root) {
            eprintln!("session_persist: cannot clear session dir: {}", e);
        }
    }
}

/// Returns true if the session dir contains at least the raw_data.parquet
/// — used at boot to decide whether to fire a "session restored" toast.
#[allow(dead_code)]
pub fn has_session() -> bool {
    session_dir().join("raw_data.parquet").exists()
}
