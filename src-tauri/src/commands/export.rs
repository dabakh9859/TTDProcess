use tauri::State;

use crate::core::data_loader;
use crate::core::types::LogLevel;
use crate::state::AppState;
use crate::utils::logger;

fn dataset_df<'a>(
    app: &'a crate::state::AppData,
    dataset: &str,
) -> Option<&'a polars::prelude::DataFrame> {
    match dataset {
        "raw" => app.raw_data.as_ref(),
        "cleaned" => app.cleaned_data.as_ref(),
        "tslope" => app.results.tslope.as_ref(),
        "baseline" => app.results.baseline.as_ref(),
        "delta_t" => app.results.delta_t.as_ref(),
        "t600" => app.results.t600.as_ref(),
        "tm" => app.results.tm.as_ref(),
        "stm" => app.results.stm.as_ref(),
        "tmi" => app.results.tmi.as_ref(),
        "k" => app.results.k.as_ref(),
        "sap_flow" => app.results.sap_flow.as_ref(),
        "ttdplus_fourier" => app.results.ttdplus_fourier.as_ref(),
        "ttdplus_refs" => app.results.ttdplus_refs.as_ref(),
        "ttdplus_sap_flow" => app.results.ttdplus_sap_flow.as_ref(),
        "rd_regression" => app.results.rd_regression.as_ref(),
        "rd_result" => app.results.rd_result.as_ref(),
        "jh" => app.results.jh.as_ref(),
        "jhp" => app.results.jhp.as_ref(),
        "qh" => app.results.qh.as_ref(),
        "qd" => app.results.qd.as_ref(),
        _ => None,
    }
}

/// Export several datasets AND/OR saved aggregations into ONE .xlsx file, one
/// sheet each. Datasets are in-memory DataFrames; aggregations are read from
/// disk and converted to DataFrames. Sheet names are de-duplicated.
#[tauri::command]
pub async fn export_data_multi(
    state: State<'_, AppState>,
    datasets: Vec<String>,
    aggregation_ids: Option<Vec<String>>,
    path: String,
) -> Result<String, String> {
    let aggregation_ids = aggregation_ids.unwrap_or_default();
    if datasets.is_empty() && aggregation_ids.is_empty() {
        return Err("Aucun dataset sélectionné.".to_string());
    }

    // Build owned (name, DataFrame) pairs so in-memory datasets and disk-backed
    // aggregations can share one workbook. Cloning the dataset frames is cheap —
    // Polars columns are behind an Arc, so this only bumps refcounts.
    let mut owned: Vec<(String, polars::prelude::DataFrame)> = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let mut owned = Vec::with_capacity(datasets.len() + aggregation_ids.len());
        for ds in &datasets {
            let df = dataset_df(&app, ds)
                .ok_or_else(|| format!("Dataset '{}' indisponible", ds))?;
            owned.push((ds.clone(), df.clone()));
        }
        owned
    };
    for id in &aggregation_ids {
        let (name, df) = crate::commands::aggregation::aggregation_as_df(id)
            .map_err(|e| format!("Agrégation '{}' : {}", id, e))?;
        owned.push((name, df));
    }

    // De-duplicate sheet names (two aggregations can share a name).
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let sheets: Vec<(String, polars::prelude::DataFrame)> = owned
        .into_iter()
        .map(|(name, df)| {
            let count = seen.entry(name.clone()).or_insert(0);
            let unique = if *count == 0 { name.clone() } else { format!("{} ({})", name, *count + 1) };
            *count += 1;
            (unique, df)
        })
        .collect();

    let n = sheets.len();
    let out = path.clone();
    // Off the main thread: see the note in `export_data`.
    tauri::async_runtime::spawn_blocking(move || {
        let borrowed: Vec<(String, &polars::prelude::DataFrame)> =
            sheets.iter().map(|(name, df)| (name.clone(), df)).collect();
        data_loader::export_to_excel_multi(&borrowed, &out).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Exporté {} feuille(s) dans {}", n, path),
    );
    Ok(format!("Exporté {} feuille(s) dans {}", n, path))
}

/// Export several datasets and/or aggregations as ONE FILE EACH into `dir`.
///
/// The Excel path puts everything in a single workbook (`export_data_multi`),
/// but CSV and DAT have no notion of sheets: the UI used to open one save
/// dialog per dataset, which is unusable past two or three. Here the user picks
/// a directory once and every selection is written into it.
///
/// `format` is the file EXTENSION ("csv", "dat", "xlsx"). DAT is written as
/// CSV content — that is what the datalogger format is — matching what the
/// single-dataset export already did.
#[tauri::command]
pub async fn export_data_multi_files(
    state: State<'_, AppState>,
    datasets: Vec<String>,
    aggregation_ids: Option<Vec<String>>,
    dir: String,
    format: String,
) -> Result<serde_json::Value, String> {
    let aggregation_ids = aggregation_ids.unwrap_or_default();
    if datasets.is_empty() && aggregation_ids.is_empty() {
        return Err("Aucun dataset sélectionné.".to_string());
    }
    let ext = match format.as_str() {
        "csv" | "dat" | "xlsx" => format.clone(),
        other => return Err(format!("Format non supporté : {}", other)),
    };

    let dir_path = std::path::PathBuf::from(&dir);
    if !dir_path.is_dir() {
        return Err(format!("Dossier introuvable : {}", dir));
    }

    // Collect (label, frame) pairs, then drop the lock — writing can take
    // minutes and must hold neither the mutex nor the main thread.
    let mut owned: Vec<(String, polars::prelude::DataFrame)> = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        let mut owned = Vec::with_capacity(datasets.len() + aggregation_ids.len());
        for ds in &datasets {
            let df = dataset_df(&app, ds)
                .ok_or_else(|| format!("Dataset '{}' indisponible", ds))?;
            owned.push((ds.clone(), df.clone()));
        }
        owned
    };
    for id in &aggregation_ids {
        let (name, df) = crate::commands::aggregation::aggregation_as_df(id)
            .map_err(|e| format!("Agrégation '{}' : {}", id, e))?;
        owned.push((name, df));
    }

    // One file per entry, names de-duplicated so two aggregations sharing a
    // label don't overwrite each other.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let planned: Vec<(std::path::PathBuf, polars::prelude::DataFrame)> = owned
        .drain(..)
        .map(|(name, df)| {
            let safe: String = name
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect();
            let count = seen.entry(safe.clone()).or_insert(0);
            let stem = if *count == 0 { safe.clone() } else { format!("{}_{}", safe, *count + 1) };
            *count += 1;
            (dir_path.join(format!("{}_export.{}", stem, ext)), df)
        })
        .collect();

    let ext_for_write = ext.clone();
    let written = tauri::async_runtime::spawn_blocking(move || {
        let mut written: Vec<String> = Vec::with_capacity(planned.len());
        for (path, df) in &planned {
            let p = path.to_string_lossy().to_string();
            if ext_for_write == "xlsx" {
                let sheet = path.file_stem().map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "data".to_string());
                data_loader::export_to_excel(df, &p, &sheet).map_err(|e| e.to_string())?;
            } else {
                data_loader::export_to_csv(df, &p).map_err(|e| e.to_string())?;
            }
            written.push(p);
        }
        Ok::<Vec<String>, String>(written)
    })
    .await
    .map_err(|e| e.to_string())??;

    let n = written.len();
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("{} fichier(s) {} exporté(s) dans {}", n, ext.to_uppercase(), dir),
    );
    Ok(serde_json::json!({ "count": n, "files": written, "dir": dir }))
}

/// Export a single saved aggregation to CSV or XLSX.
#[tauri::command]
pub async fn export_aggregation(
    state: State<'_, AppState>,
    aggregation_id: String,
    format: String,
    path: String,
) -> Result<String, String> {
    let (out, fmt, id) = (path.clone(), format.clone(), aggregation_id.clone());
    // Off the main thread: see the note in `export_data`.
    let name = tauri::async_runtime::spawn_blocking(move || {
        let (name, df) = crate::commands::aggregation::aggregation_as_df(&id)
            .map_err(|e| e.to_string())?;
        match fmt.as_str() {
            "csv" => data_loader::export_to_csv(&df, &out).map_err(|e| e.to_string())?,
            "xlsx" => data_loader::export_to_excel(&df, &out, &name).map_err(|e| e.to_string())?,
            _ => return Err(format!("Format non supporté : {}", fmt)),
        }
        Ok::<String, String>(name)
    })
    .await
    .map_err(|e| e.to_string())??;

    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Agrégation exportée : {} → {} ({})", name, path, format),
    );
    Ok(format!("Exporté vers {}", path))
}

/// Writing a workbook can take minutes on a large dataset. A synchronous
/// `#[tauri::command]` runs inline on the main thread, which stops the GTK
/// event loop: the app then no longer services its Wayland socket and the
/// compositor disconnects it — the process dies with "Lost connection to
/// Wayland compositor". Hence `async` + `spawn_blocking`, and copying the
/// frame out of the state mutex so the rest of the app keeps working while
/// the file is written.
#[tauri::command]
pub async fn export_data(
    state: State<'_, AppState>,
    dataset: String,
    format: String,
    path: String,
) -> Result<String, String> {
    let df = {
        let app = state.inner.lock().map_err(|e| e.to_string())?;
        dataset_df(&app, &dataset)
            .ok_or_else(|| format!("Dataset '{}' is not available", dataset))?
            .clone()
    };

    let (out, fmt, sheet) = (path.clone(), format.clone(), dataset.clone());
    tauri::async_runtime::spawn_blocking(move || match fmt.as_str() {
        "csv" => data_loader::export_to_csv(&df, &out).map_err(|e| e.to_string()),
        "xlsx" => data_loader::export_to_excel(&df, &out, &sheet).map_err(|e| e.to_string()),
        _ => Err(format!("Unsupported export format: {}", fmt)),
    })
    .await
    .map_err(|e| e.to_string())??;

    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Exported '{}' to {} ({})", dataset, path, format),
    );

    Ok(format!("Exported to {}", path))
}
