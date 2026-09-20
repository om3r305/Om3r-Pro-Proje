-- APPLY TO: brian-market-intelligence (Core) only.
-- Realtime owns fast scheduling; Core retains stateful/slow work.

create table if not exists public.brian_scheduler_state (
  scheduler_id text primary key,
  owner_project text not null,
  last_seen_at timestamptz not null,
  last_action text,
  status text not null default 'ONLINE',
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create index if not exists brian_scheduler_state_seen_idx
  on public.brian_scheduler_state(last_seen_at desc);

-- The scheduler bridge implementation is installed in the live Core database and
-- called only by the authenticated brian-core-scheduler-bridge Edge Function.
-- Keep the migrated Core pg_cron jobs disabled; scheduling now belongs to brian-realtime.
do $$
declare
  r record;
begin
  for r in
    select jobid
    from cron.job
    where jobname = any(array[
      'brian-core-recovery-dispatcher-v1',
      'brian-official-primary-eye-v2-5m',
      'brian-source-observer-v2-2m',
      'brian-source-registry-v2-2m',
      'brian-meeting-backend-sync-1m',
      'brian-pgnet-watchdog-59s',
      'brian-core-alpha-sync-3m',
      'brian-core-world-sync-6m',
      'brian-core-treasury-sync-4m',
      'brian-core-discovery-sync-10m',
      'brian-dip-multiasset-v873-shadow-20s',
      'brian-multiasset-opportunity-10m',
      'brian-frontier-heartbeat-cache-1m',
      'brian-breaking-scout-2m'
    ]::text[])
  loop
    perform cron.alter_job(r.jobid, active := false);
  end loop;
end $$;

-- brian_system_control_status and brian_private.refresh_frontier_heartbeat_cache
-- are intentionally scheduler-owner aware in production:
-- an ONLINE brian-realtime-core-scheduler heartbeat counts migrated logical jobs
-- as active even though their legacy Core cron rows are disabled.
