import ViewStub from './ViewStub'

export default function EnvDataView() {
  return (
    <ViewStub
      labelKey="sidebar.envdata"
      tab="envdata"
      namespace="envdata"
      keyCount={62}
      charts={0}
      commands={[
        'clear_env_data',
        'get_env_info',
        'get_sheet_names',
        'get_table_page',
        'load_env_data_multi',
        'load_env_data_sliced',
        'preview_file',
      ]}
    />
  )
}
