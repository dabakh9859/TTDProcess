import ViewStub from './ViewStub'

export default function AdvancedCalculationsView() {
  return (
    <ViewStub
      labelKey="sidebar.calculsAdvanced"
      tab="calculsAdvanced"
      namespace="calculsAdv"
      keyCount={23}
      charts={0}
      commands={[
        'compute_advanced_chain',
        'list_advanced_sensors',
      ]}
    />
  )
}
