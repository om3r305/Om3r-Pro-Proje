from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "supabase" / "functions" / "brian-control-center" / "index.ts"
TEST = ROOT / "tests" / "test_brian2026_dashboard_mobile_tr.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


src = CONTROL.read_text(encoding="utf-8")

old_constants = '''const ALPHA_COLLECTORS = [
  "brian-alpha-decision-compiler-v2",
  "brian-missed-opportunity-auditor-v3",
  "brian-official-macro-eye",
];
const CLOUD_COLLECTORS = [
  { id: "brian-universe-collector", label: "Piyasa Evreni", cadenceSeconds: 900 },
  { id: "brian-live-shadow", label: "Phase 3.7 Shadow", cadenceSeconds: 300 },
  { id: "brian-sensor-mesh", label: "Sensör Ağı", cadenceSeconds: 600 },
  { id: "brian-derivatives-eye", label: "Türev Piyasa Gözü", cadenceSeconds: 300 },
  { id: "brian-intrabar-eye", label: "Intrabar Gözü", cadenceSeconds: 120 },
  { id: "brian-alpha-decision-compiler-v2", label: "ALPHA Karar Motoru", cadenceSeconds: 60 },
  { id: "brian-missed-opportunity-auditor-v3", label: "Sonuç Denetçisi", cadenceSeconds: 300 },
  { id: "brian-official-macro-eye", label: "Resmî Makro Gözü", cadenceSeconds: 600 },
  { id: "brian-fx-eye", label: "Döviz Gözü", cadenceSeconds: 3600 },
] as const;
'''

new_constants = '''const ALPHA_COLLECTORS = [
  "brian-alpha-decision-compiler-v2",
  "brian-missed-opportunity-auditor-v3",
  // Production v3 currently preserves the historical collector_id for append-only continuity.
  "brian-missed-opportunity-auditor-v2",
  "brian-official-macro-eye",
];
type CloudComponentConfig = {
  id: string;
  label: string;
  cadenceSeconds: number;
  collectorIds: readonly string[];
};
const CLOUD_COLLECTORS: readonly CloudComponentConfig[] = [
  // These three services publish canonical output tables but do not write brian_collector_runs.
  { id: "brian-universe-collector", label: "Piyasa Evreni", cadenceSeconds: 900, collectorIds: [] },
  { id: "brian-live-shadow", label: "Phase 3.7 Shadow", cadenceSeconds: 300, collectorIds: [] },
  { id: "brian-sensor-mesh", label: "Sensör Ağı", cadenceSeconds: 600, collectorIds: [] },
  // Keep canonical dashboard ids while accepting the collector ids actually emitted in production.
  { id: "brian-derivatives-eye", label: "Türev Piyasa Gözü", cadenceSeconds: 300, collectorIds: ["brian-derivatives-eye", "phase39-binance-usdm-derivatives"] },
  { id: "brian-intrabar-eye", label: "Intrabar Gözü", cadenceSeconds: 120, collectorIds: ["brian-intrabar-eye"] },
  { id: "brian-alpha-decision-compiler-v2", label: "ALPHA Karar Motoru", cadenceSeconds: 60, collectorIds: ["brian-alpha-decision-compiler-v2"] },
  { id: "brian-missed-opportunity-auditor-v3", label: "Sonuç Denetçisi", cadenceSeconds: 300, collectorIds: ["brian-missed-opportunity-auditor-v3", "brian-missed-opportunity-auditor-v2"] },
  { id: "brian-official-macro-eye", label: "Resmî Makro Gözü", cadenceSeconds: 600, collectorIds: ["brian-official-macro-eye"] },
  { id: "brian-fx-eye", label: "Döviz Gözü", cadenceSeconds: 3600, collectorIds: ["brian-fx-eye", "phase39-ecb-fx"] },
];
'''

old_health = '''async function loadCloudHealth() {
  const runs = await supabase.from("brian_collector_runs")
    .select("collector_id,status,started_at,finished_at,degraded_sources,error_class,error_message,metadata")
    .in("collector_id", CLOUD_COLLECTORS.map((item) => item.id)).order("started_at", { ascending: false }).limit(160);
  if (runs.error) throw runs.error;
  const latest = new Map<string, CollectorRun>();
  for (const row of (runs.data ?? []) as CollectorRun[]) if (!latest.has(row.collector_id)) latest.set(row.collector_id, row);
  const now = Date.now();
  const components = CLOUD_COLLECTORS.map((item) => {
    const row = latest.get(item.id);
    const runAge = ageSeconds(row?.started_at, now);
    const fresh = runAge != null && runAge <= item.cadenceSeconds * 3 + 90;
    const successish = row?.status === "SUCCESS" || row?.status === "DEGRADED";
    return {
      id: item.id,
      label: item.label,
      cadence_seconds: item.cadenceSeconds,
      state: !row ? "NO_DATA" : fresh && successish ? row.status : fresh ? "ERROR" : "STALE",
      last_status: row?.status ?? null,
      last_started_at: row?.started_at ?? null,
      last_finished_at: row?.finished_at ?? null,
      age_seconds: runAge,
      error_class: row?.error_class ?? null,
      error_message: row?.error_message ?? null,
      degraded_sources: row?.degraded_sources ?? [],
    };
  });
  const healthy = components.filter((row) => row.state === "SUCCESS" || row.state === "DEGRADED").length;
  return {
    mode: "SERVER_SIDE_CRON",
    browser_independent: true,
    continues_when_page_closed: true,
    healthy_components: healthy,
    total_components: components.length,
    overall: healthy === components.length ? "ONLINE" : healthy >= Math.ceil(components.length * 0.7) ? "DEGRADED" : "ERROR",
    components,
  };
}
'''

new_health = '''async function loadCloudHealth() {
  // Dashboard health must follow the service's canonical production output, not assume that every
  // worker writes the same collector-run id. Universe, Phase 3.7 and Sensor Mesh are output-table
  // driven; derivatives/FX/auditor also retain historical collector ids for append-only continuity.
  const [universe, phase37, sensor] = await Promise.all([
    supabase.from("brian_universe_snapshots")
      .select("observed_at").order("observed_at", { ascending: false }).limit(1).maybeSingle(),
    supabase.from("brian_live_shadow_ticks")
      .select("observed_at").eq("experiment_id", SOURCE_EXPERIMENT_ID)
      .order("observed_at", { ascending: false }).limit(1).maybeSingle(),
    supabase.from("brian_sensor_observations")
      .select("observed_at").eq("available", true)
      .order("observed_at", { ascending: false }).limit(1).maybeSingle(),
  ]);

  const latestByComponent = new Map<string, CollectorRun>();
  const collectorErrors = new Map<string, string>();
  await Promise.all(CLOUD_COLLECTORS.filter((item) => item.collectorIds.length > 0).map(async (item) => {
    const result = await supabase.from("brian_collector_runs")
      .select("collector_id,status,started_at,finished_at,degraded_sources,error_class,error_message,metadata")
      .in("collector_id", [...item.collectorIds])
      .order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (result.error) collectorErrors.set(item.id, result.error.message);
    else if (result.data) latestByComponent.set(item.id, result.data as CollectorRun);
  }));

  type TimestampProbe = {
    data: { observed_at?: string | null } | null;
    error: { message: string } | null;
  };
  const outputProbes = new Map<string, TimestampProbe>([
    ["brian-universe-collector", universe as TimestampProbe],
    ["brian-live-shadow", phase37 as TimestampProbe],
    ["brian-sensor-mesh", sensor as TimestampProbe],
  ]);
  const now = Date.now();
  const components = CLOUD_COLLECTORS.map((item) => {
    const probe = outputProbes.get(item.id);
    if (probe) {
      const observedAt = probe.data?.observed_at ? String(probe.data.observed_at) : null;
      const runAge = ageSeconds(observedAt, now);
      const fresh = runAge != null && runAge <= item.cadenceSeconds * 3 + 90;
      return {
        id: item.id,
        label: item.label,
        cadence_seconds: item.cadenceSeconds,
        source_kind: "OUTPUT_DATA",
        source_id: item.id,
        state: probe.error ? "ERROR" : !observedAt ? "NO_DATA" : fresh ? "SUCCESS" : "STALE",
        last_status: probe.error ? "ERROR" : observedAt ? "SUCCESS" : null,
        last_started_at: observedAt,
        last_finished_at: observedAt,
        age_seconds: runAge,
        error_class: probe.error ? "HEALTH_PROBE_ERROR" : null,
        error_message: probe.error?.message ?? null,
        degraded_sources: [],
      };
    }

    const row = latestByComponent.get(item.id);
    const queryError = collectorErrors.get(item.id) ?? null;
    const runAge = ageSeconds(row?.started_at, now);
    const fresh = runAge != null && runAge <= item.cadenceSeconds * 3 + 90;
    const successish = row?.status === "SUCCESS" || row?.status === "DEGRADED";
    return {
      id: item.id,
      label: item.label,
      cadence_seconds: item.cadenceSeconds,
      source_kind: "COLLECTOR_RUN",
      source_id: row?.collector_id ?? null,
      state: queryError ? "ERROR" : !row ? "NO_DATA" : fresh && successish ? row.status : fresh ? "ERROR" : "STALE",
      last_status: row?.status ?? null,
      last_started_at: row?.started_at ?? null,
      last_finished_at: row?.finished_at ?? null,
      age_seconds: runAge,
      error_class: queryError ? "HEALTH_PROBE_ERROR" : row?.error_class ?? null,
      error_message: queryError ?? row?.error_message ?? null,
      degraded_sources: row?.degraded_sources ?? [],
    };
  });
  const healthy = components.filter((row) => row.state === "SUCCESS" || row.state === "DEGRADED").length;
  const anyDegraded = components.some((row) => row.state === "DEGRADED");
  return {
    mode: "SERVER_SIDE_CRON",
    browser_independent: true,
    continues_when_page_closed: true,
    healthy_components: healthy,
    total_components: components.length,
    overall: healthy === components.length && !anyDegraded
      ? "ONLINE"
      : healthy >= Math.ceil(components.length * 0.7) ? "DEGRADED" : "ERROR",
    components,
  };
}
'''

src = replace_once(src, old_constants, new_constants, "control constants")
src = replace_once(src, old_health, new_health, "cloud health")
CONTROL.write_text(src, encoding="utf-8")

test = TEST.read_text(encoding="utf-8")
old_test = '''def test_control_center_is_connected_to_current_auditor_and_learning_chain():
    src = text(CONTROL)
    assert '\"brian-missed-opportunity-auditor-v3\"' in src
    assert '\"brian-missed-opportunity-auditor-v2\"' not in src
    assert 'brian_sensor_reliability_shadow_snapshots' in src
'''
new_test = '''def test_control_center_is_connected_to_current_auditor_and_learning_chain():
    src = text(CONTROL)
    assert '\"brian-missed-opportunity-auditor-v3\"' in src
    # v3 production keeps the v2 collector_id in append-only run telemetry, so health accepts both.
    assert '[\"brian-missed-opportunity-auditor-v3\", \"brian-missed-opportunity-auditor-v2\"]' in src
    assert 'brian_sensor_reliability_shadow_snapshots' in src
'''
test = replace_once(test, old_test, new_test, "dashboard auditor test")

needle = '''def test_dashboard_keeps_existing_control_actions_and_shadow_boundary():
'''
health_test = '''def test_control_center_health_uses_canonical_outputs_and_production_collector_aliases():
    src = text(CONTROL)
    for table in ["brian_universe_snapshots", "brian_live_shadow_ticks", "brian_sensor_observations"]:
        assert table in src
    assert '"phase39-binance-usdm-derivatives"' in src
    assert '"phase39-ecb-fx"' in src
    assert 'source_kind: "OUTPUT_DATA"' in src
    assert 'source_kind: "COLLECTOR_RUN"' in src
    assert 'healthy === components.length && !anyDegraded' in src


'''
if health_test not in test:
    test = replace_once(test, needle, health_test + needle, "health regression insertion")
TEST.write_text(test, encoding="utf-8")

print("dashboard health patch applied")
