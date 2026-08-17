import ViewStub from './ViewStub'

export default function VisualizationView() {
  return (
    <ViewStub
      labelKey="sidebar.visualization"
      tab="visualisation"
      namespace="viz"
      keyCount={135}
      charts={1}
      commands={[
        'viz_list_sources',
        'viz_load_file',
        'viz_unload_file',
        'viz_overlay_load',
        'viz_overlay_remove',
        'viz_overlay_list',
        'list_scenarios',
        'get_sheet_names',
        'get_table_page',
      ]}
    />
  )
}
