import ViewStub from './ViewStub'

export default function GapFillingView() {
  return (
    <ViewStub
      labelKey="sidebar.gapFilling"
      tab="gapfilling"
      namespace="gap"
      keyCount={161}
      charts={3}
      commands={[
        'gap_filling_run_classical',
        'cleaning_complete_saits',
        'cleaning_commit_completion',
        'cleaning_discard_completion',
        'cleaning_time_gaps',
        'cleaning_reindex_time',
        'cleaning_train_ml',
        'cleaning_apply_ml',
        'cleaning_reset_ml',
        'cleaning_reset_to_raw',
        'cleaning_list_columns',
        'cleaning_ml_status',
        'get_cleaning_info',
        'get_cleaning_pre_rows',
        'ai_train',
        'ai_list_models',
        'list_scenarios',
        'list_scenario_datasets',
        'viz_load_file',
        'viz_unload_file',
        'load_env_data',
        'get_env_info',
        'clear_env_data',
        'preview_file',
        'get_sheet_names',
        'get_table_page',
      ]}
    />
  )
}
