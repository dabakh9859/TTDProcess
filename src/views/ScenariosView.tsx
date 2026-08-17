import ViewStub from './ViewStub'

export default function ScenariosView() {
  return (
    <ViewStub
      labelKey="sidebar.scenarios"
      tab="scenarios"
      namespace="scenarios"
      keyCount={22}
      charts={3}
      commands={[
        'save_scenario',
        'load_scenario',
        'list_scenarios',
        'delete_scenario',
      ]}
    />
  )
}
