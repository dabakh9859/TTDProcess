import ViewStub from './ViewStub'

export default function AggregationView() {
  return (
    <ViewStub
      labelKey="sidebar.aggregation"
      tab="agregation"
      namespace="agg"
      keyCount={68}
      charts={0}
      commands={[
        'aggregate_data',
        'list_aggregation_sources',
        'list_aggregations',
        'load_aggregation',
        'delete_aggregation',
        'rename_aggregation',
      ]}
    />
  )
}
