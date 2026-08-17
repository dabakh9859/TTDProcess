// GENERATED from src-tauri/src/lib.rs - do not edit by hand.
// Regenerate whenever you add or rename a #[tauri::command].
//
// Commands marked UNUSED are exposed by Rust but were never called by the
// previous frontend. Check what they actually do before wiring them up.

export type Command =
  // Import commands
  | 'get_sheet_names'
  | 'preview_file'
  | 'load_data'
  | 'get_column_stats'
  // Table commands
  | 'get_table_page'
  | 'viz_list_sources'
  | 'viz_load_file'
  | 'viz_unload_file'
  | 'get_datasets_info'
  | 'update_cell'
  // Calculation commands (TTD classic)
  | 'run_pipeline'
  | 'recompute_from_stage'
  | 'set_sap_flow_params'
  // Calculation commands (TTD+)
  | 'set_ttdplus_params'
  | 'run_ttdplus_pipeline'
  | 'list_diurnal_regression_diagnostics'
  | 'get_diurnal_regression_diagnostic'
  // Calculs avancés — Jh → Jhp → Qh → Qd
  | 'list_advanced_sensors'
  | 'compute_advanced_chain'
  // ML commands
  | 'train_model'   // UNUSED
  | 'predict'   // UNUSED
  | 'list_models'   // UNUSED
  | 'save_model'   // UNUSED
  | 'load_model'   // UNUSED
  // Cleaning commands
  | 'detect_outliers'   // UNUSED
  | 'fill_gaps'   // UNUSED
  | 'validate_detection'   // UNUSED
  // Aggregation
  | 'aggregate_data'
  | 'list_aggregation_sources'
  | 'list_aggregations'
  | 'load_aggregation'
  | 'delete_aggregation'
  | 'rename_aggregation'
  // Environmental data (DataEnv Tm method)
  | 'load_env_data'
  | 'load_env_data_sliced'
  | 'load_env_data_multi'
  | 'get_env_info'
  | 'clear_env_data'
  | 'preview_env_conditions'   // UNUSED
  | 'env_column_quick_stats'   // UNUSED
  // Cleaning v2 (two-path: classical + ML)
  | 'cleaning_list_columns'
  | 'cleaning_detect_only'
  | 'cleaning_run_classical'
  | 'gap_filling_run_classical'
  | 'cleaning_train_ml'
  | 'cleaning_detect_ml'
  | 'cleaning_apply_ml'
  | 'cleaning_mark_manual_nan'
  | 'cleaning_time_gaps'
  | 'cleaning_reindex_time'
  | 'cleaning_reset_ml'
  | 'cleaning_reset_to_raw'
  | 'cleaning_lock_permanent'
  | 'cleaning_unlock_permanent'
  | 'clear_session'   // UNUSED
  | 'cleaning_ml_status'
  | 'get_app_status'
  | 'get_cleaning_info'
  | 'get_cleaning_pre_rows'
  // Scenario commands
  | 'save_scenario'
  | 'load_scenario'
  | 'list_scenarios'
  | 'delete_scenario'
  | 'list_scenario_datasets'
  // Visualisation overlays (multi-scenario comparison)
  | 'viz_overlay_load'
  | 'viz_overlay_remove'
  | 'viz_overlay_list'
  // Export
  | 'export_data'
  | 'export_data_multi'
  | 'export_aggregation'
  // Journal
  | 'get_logs'   // UNUSED
  | 'clear_logs'   // UNUSED
  // AI sidecar (SAITS deep-learning imputation)
  | 'ai_health'
  | 'ai_train'
  | 'ai_predict'
  | 'ai_model_save'
  | 'ai_model_load'
  | 'ai_inspect_files'
  | 'ai_list_models'
  | 'ai_list_env_columns'   // UNUSED
  | 'cleaning_apply_saits'   // UNUSED
  | 'cleaning_apply_saits_clean'   // UNUSED
  | 'cleaning_restore_cells'   // UNUSED
  | 'cleaning_detect_saits'
  | 'cleaning_apply_cells'
  | 'cleaning_complete_saits'
  | 'cleaning_commit_completion'
  | 'cleaning_discard_completion'
