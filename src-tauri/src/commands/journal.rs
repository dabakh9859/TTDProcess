use tauri::State;

use crate::core::types::LogEntry;
use crate::state::AppState;

#[tauri::command]
pub fn get_logs(state: State<'_, AppState>) -> Result<Vec<LogEntry>, String> {
    let data = state.inner.lock().map_err(|e| e.to_string())?;
    Ok(data.logs.clone())
}

#[tauri::command]
pub fn clear_logs(state: State<'_, AppState>) -> Result<String, String> {
    let mut data = state.inner.lock().map_err(|e| e.to_string())?;
    data.logs.clear();
    Ok("Logs cleared".to_string())
}
