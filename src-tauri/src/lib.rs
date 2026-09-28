mod ai;
mod commands;
mod core;
mod ml;
mod state;
mod utils;

use ai::AiSidecar;
use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .manage(AiSidecar::new())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            // Import commands
            commands::import::get_sheet_names,
            commands::import::preview_file,
            commands::import::load_data,
            commands::import::get_column_stats,
            // Table commands
            commands::table::get_table_page,
            commands::table::viz_list_sources,
            commands::table::viz_load_file,
            commands::table::viz_unload_file,
            commands::table::get_datasets_info,
            commands::table::update_cell,
            // Calculation commands (TTD classic)
            commands::calculations::run_pipeline,
            commands::calculations::recompute_from_stage,
            commands::calculations::set_sap_flow_params,
            // Calculation commands (TTD+)
            commands::calculations::set_ttdplus_params,
            commands::calculations::run_ttdplus_pipeline,
            commands::calculations::list_diurnal_regression_diagnostics,
            commands::calculations::get_diurnal_regression_diagnostic,
            // Calculs avancés — Jh → Jhp → Qh → Qd
            commands::calculations::list_advanced_sensors,
            commands::calculations::compute_advanced_chain,
            // ML commands
            commands::ml::train_model,
            commands::ml::predict,
            commands::ml::list_models,
            commands::ml::delete_model,
            commands::ml::save_model,
            commands::ml::load_model,
            // Cleaning commands
            commands::cleaning::detect_outliers,
            commands::cleaning::fill_gaps,
            commands::cleaning::validate_detection,
            // Aggregation
            commands::aggregation::aggregate_data,
            commands::aggregation::list_aggregation_sources,
            commands::aggregation::list_aggregations,
            commands::aggregation::load_aggregation,
            commands::aggregation::delete_aggregation,
            commands::aggregation::rename_aggregation,
            // Environmental data (DataEnv Tm method)
            commands::env_data::load_env_data,
            commands::env_data::load_env_data_sliced,
            commands::env_data::load_env_data_multi,
            commands::env_data::get_env_info,
            commands::env_data::clear_env_data,
            commands::env_data::preview_env_conditions,
            commands::env_data::env_column_quick_stats,
            // Cleaning v2 (two-path: classical + ML)
            commands::cleaning_v2::cleaning_list_columns,
            commands::cleaning_v2::cleaning_detect_only,
            commands::cleaning_v2::cleaning_run_classical,
            commands::cleaning_v2::gap_filling_run_classical,
            commands::cleaning_v2::cleaning_train_ml,
            commands::cleaning_v2::cleaning_detect_ml,
            commands::cleaning_v2::cleaning_apply_ml,
            commands::cleaning_v2::cleaning_mark_manual_nan,
            commands::cleaning_v2::cleaning_time_gaps,
            commands::cleaning_v2::cleaning_reindex_time,
            commands::cleaning_v2::cleaning_origin_flags,
            commands::cleaning_v2::export_origin_flags,
            commands::cleaning_v2::cleaning_reset_ml,
            commands::cleaning_v2::cleaning_reset_to_raw,
            commands::cleaning_v2::cleaning_lock_permanent,
            commands::cleaning_v2::cleaning_unlock_permanent,
            commands::cleaning_v2::clear_session,
            commands::cleaning_v2::cleaning_ml_status,
            commands::cleaning_v2::get_app_status,
            commands::cleaning_v2::get_cleaning_info,
            commands::cleaning_v2::get_cleaning_pre_rows,
            // Scenario commands
            commands::scenarios::save_scenario,
            commands::scenarios::load_scenario,
            commands::scenarios::list_scenarios,
            commands::scenarios::delete_scenario,
            commands::scenarios::list_scenario_datasets,
            // Visualisation overlays (multi-scenario comparison)
            commands::scenarios::viz_overlay_load,
            commands::scenarios::viz_overlay_remove,
            commands::scenarios::viz_overlay_list,
            // Export
            commands::export::export_data,
            commands::export::export_data_multi,
            commands::export::export_data_multi_files,
            commands::export::export_aggregation,
            // Journal
            commands::journal::get_logs,
            commands::journal::clear_logs,
            // AI sidecar (SAITS deep-learning imputation)
            commands::ai::ai_health,
            commands::ai::ai_train,
            commands::ai::ai_predict,
            commands::ai::ai_model_save,
            commands::ai::ai_model_load,
            commands::ai::ai_model_delete,
            commands::ai::ai_inspect_files,
            commands::ai::ai_list_models,
            commands::ai::ai_list_env_columns,
            commands::ai::cleaning_apply_saits,
            commands::ai::cleaning_apply_saits_clean,
            commands::ai::cleaning_restore_cells,
            commands::ai::cleaning_detect_saits,
            commands::ai::cleaning_apply_cells,
            commands::ai::cleaning_complete_saits,
            commands::ai::cleaning_commit_completion,
            commands::ai::cleaning_discard_completion,
        ])
        .setup(|_app| {
            #[cfg(debug_assertions)]
            {
                // Could add dev-tools initialization here
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
