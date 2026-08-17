import ViewStub from './ViewStub'

export default function DetectionView() {
  return (
    <ViewStub
      labelKey="sidebar.dataCleaning"
      tab="detection"
      namespace="detection"
      keyCount={371}
      charts={9}
      commands={[
        'cleaning_detect_only',
        'cleaning_run_classical',
        'cleaning_train_ml',
        'cleaning_detect_ml',
        'cleaning_apply_ml',
        'cleaning_detect_saits',
        'cleaning_apply_cells',
        'cleaning_mark_manual_nan',
        'cleaning_reset_ml',
        'cleaning_reset_to_raw',
        'cleaning_lock_permanent',
        'cleaning_unlock_permanent',
        'cleaning_list_columns',
        'cleaning_ml_status',
        'get_cleaning_info',
        'ai_list_models',
        'ai_model_load',
        'load_env_data',
        'get_env_info',
        'clear_env_data',
        'get_sheet_names',
        'get_table_page',
      ]}
    />
  )
}
