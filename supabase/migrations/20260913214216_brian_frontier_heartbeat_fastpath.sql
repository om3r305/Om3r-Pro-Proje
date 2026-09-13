create index if not exists brian_collector_runs_collector_started_idx
  on public.brian_collector_runs (collector_id, started_at desc);

create or replace function public.brian_frontier_heartbeat_snapshot()
returns jsonb
language sql
security definer
set search_path = pg_catalog, public
as $$
with latest_alpha as (
  select observed_at, decision_id, asset_id, action, direction,
         evidence_score, estimated_round_trip_cost_bps
  from public.brian_alpha_decisions
  order by observed_at desc
  limit 1
),
latest_world as (
  select status, started_at, finished_at, input_events, event_frames,
         entity_observations, narrative_snapshots, asset_impacts
  from public.brian_world_brain_runs
  order by started_at desc
  limit 1
),
latest_runs as (
  select distinct on (collector_id)
         collector_id, status, started_at, finished_at, error_class,
         error_message, observed_records, stored_records
  from public.brian_collector_runs
  where collector_id = any(array[
    'brian-world-brain-v1',
    'brian-world-discovery-eye-v1',
    'brian-alpha-decision-compiler-v2',
    'brian-intrabar-eye',
    'brian-evolution-orchestrator-v1',
    'brian-evolution-researcher-v1',
    'brian-evolution-sandbox-v1',
    'brian-evolution-ocean-worker-v1',
    'brian-evolution-treasury-v1',
    'brian-evolution-alpha-edge-challenger-v1'
  ]::text[])
  order by collector_id, started_at desc
),
collector_json as (
  select coalesce(
    jsonb_object_agg(collector_id, to_jsonb(latest_runs) - 'collector_id'),
    '{}'::jsonb
  ) as value
  from latest_runs
)
select jsonb_build_object(
  'control', public.brian_system_control_status(),
  'alpha', (select to_jsonb(latest_alpha) from latest_alpha),
  'world_run', (select to_jsonb(latest_world) from latest_world),
  'collectors', (select value from collector_json),
  'dip_touched', false,
  'shadow_only', true,
  'live_execution', false
);
$$;

revoke all on function public.brian_frontier_heartbeat_snapshot() from public, anon, authenticated;
grant execute on function public.brian_frontier_heartbeat_snapshot() to service_role;
