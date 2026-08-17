import ViewStub from './ViewStub'

export default function TableView() {
  return (
    <ViewStub
      labelKey="sidebar.table"
      tab="tableau"
      namespace="table"
      keyCount={48}
      charts={1}
      commands={[
        'get_table_page',
        'update_cell',
      ]}
    />
  )
}
