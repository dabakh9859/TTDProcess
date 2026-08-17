import ViewStub from './ViewStub'

export default function ExportView() {
  return (
    <ViewStub
      labelKey="sidebar.export"
      tab="export"
      namespace="export"
      keyCount={52}
      charts={0}
      commands={[
        'export_data',
        'export_data_multi',
        'export_aggregation',
        'get_datasets_info',
        'list_aggregations',
      ]}
    />
  )
}
