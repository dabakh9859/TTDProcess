import ViewStub from './ViewStub'

export default function CalculationsView() {
  return (
    <ViewStub
      labelKey="sidebar.calculsRaw"
      tab="calculs"
      namespace="calculs"
      keyCount={194}
      charts={2}
      commands={[
        'run_pipeline',
        'recompute_from_stage',
        'set_sap_flow_params',
        'set_ttdplus_params',
        'run_ttdplus_pipeline',
        'list_diurnal_regression_diagnostics',
        'get_diurnal_regression_diagnostic',
        'load_env_data',
        'get_env_info',
        'clear_env_data',
        'get_cleaning_info',
        'get_sheet_names',
        'get_table_page',
      ]}
    />
  )
}
