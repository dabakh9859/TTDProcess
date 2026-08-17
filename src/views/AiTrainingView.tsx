import ViewStub from './ViewStub'

export default function AiTrainingView() {
  return (
    <ViewStub
      labelKey="sidebar.aiTraining"
      tab="aiTraining"
      namespace="(textes en dur)"
      keyCount={0}
      charts={7}
      commands={[
        'ai_health',
        'ai_train',
        'ai_predict',
        'ai_model_save',
        'ai_model_load',
        'ai_list_models',
        'ai_inspect_files',
        'cleaning_list_columns',
        'cleaning_ml_status',
        'load_env_data',
        'get_env_info',
        'get_sheet_names',
        'get_table_page',
      ]}
    />
  )
}
