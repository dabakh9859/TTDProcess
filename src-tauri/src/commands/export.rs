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
pub fn export_data_multi(
    state: State<'_, AppState>,
    datasets: Vec<String>,
    aggregation_ids: Option<Vec<String>>,
    path: String,
) -> Result<String, String> {
    let aggregation_ids = aggregation_ids.unwrap_or_default();
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    if datasets.is_empty() && aggregation_ids.is_empty() {
        return Err("Aucun dataset sélectionné.".to_string());
    }

    // Build owned (name, DataFrame) pairs so in-memory datasets and disk-backed
    // aggregations can share one workbook. Cloning the dataset frames is fine —
    // export is not a hot path.
    let mut owned: Vec<(String, polars::prelude::DataFrame)> =
        Vec::with_capacity(datasets.len() + aggregation_ids.len());
    for ds in &datasets {
        let df = dataset_df(&app, ds)
            .ok_or_else(|| format!("Dataset '{}' indisponible", ds))?;
        owned.push((ds.clone(), df.clone()));
    }
    for id in &aggregation_ids {
        let (name, df) = crate::commands::aggregation::aggregation_as_df(id)
            .map_err(|e| format!("Agrégation '{}' : {}", id, e))?;
        owned.push((name, df));
    }

    // De-duplicate sheet names (two aggregations can share a name).
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let sheets: Vec<(String, &polars::prelude::DataFrame)> = owned
        .iter()
        .map(|(name, df)| {
            let count = seen.entry(name.clone()).or_insert(0);
            let unique = if *count == 0 { name.clone() } else { format!("{} ({})", name, *count + 1) };
            *count += 1;
            (unique, df)
        })
        .collect();

    data_loader::export_to_excel_multi(&sheets, &path).map_err(|e| e.to_string())?;

    let n = sheets.len();
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Exporté {} feuille(s) dans {}", n, path),
    );
    Ok(format!("Exporté {} feuille(s) dans {}", n, path))
}

/// Export a single saved aggregation to CSV or XLSX.
#[tauri::command]
pub fn export_aggregation(
    state: State<'_, AppState>,
    aggregation_id: String,
    format: String,
    path: String,
) -> Result<String, String> {
    let (name, df) = crate::commands::aggregation::aggregation_as_df(&aggregation_id)
        .map_err(|e| e.to_string())?;
    match format.as_str() {
        "csv" => data_loader::export_to_csv(&df, &path).map_err(|e| e.to_string())?,
        "xlsx" => data_loader::export_to_excel(&df, &path, &name).map_err(|e| e.to_string())?,
        _ => return Err(format!("Format non supporté : {}", format)),
    }
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;
    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Agrégation exportée : {} → {} ({})", name, path, format),
    );
    Ok(format!("Exporté vers {}", path))
}

#[tauri::command]
pub fn export_data(
    state: State<'_, AppState>,
    dataset: String,
    format: String,
    path: String,
) -> Result<String, String> {
    let mut app = state.inner.lock().map_err(|e| e.to_string())?;

    let df = dataset_df(&app, &dataset);

    let df = df.ok_or_else(|| format!("Dataset '{}' is not available", dataset))?;

    match format.as_str() {
        "csv" => data_loader::export_to_csv(df, &path).map_err(|e| e.to_string())?,
        "xlsx" => {
            data_loader::export_to_excel(df, &path, &dataset).map_err(|e| e.to_string())?
        }
        _ => return Err(format!("Unsupported export format: {}", format)),
    }

    logger::add_log(
        &mut app.logs,
        LogLevel::Success,
        format!("Exported '{}' to {} ({})", dataset, path, format),
    );

    Ok(format!("Exported to {}", path))
}
