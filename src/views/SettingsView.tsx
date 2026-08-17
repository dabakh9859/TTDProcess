import ViewStub from './ViewStub'

export default function SettingsView() {
  return (
    <ViewStub
      labelKey="sidebar.settings"
      tab="parametres"
      namespace="settings"
      keyCount={27}
      charts={0}
      commands={[
        'set_sap_flow_params',
      ]}
    />
  )
}
