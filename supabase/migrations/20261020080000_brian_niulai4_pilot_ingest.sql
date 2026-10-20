-- NIULAI4 observation pilot: results/health sink on CORE + read-only status for the UI.
--
-- The pilot itself runs OUTSIDE Supabase (a long-lived local process). It may push only small
-- heartbeats and interim/final summaries here, through brian-niulai4-pilot-ingest, authenticated by
-- a dedicated pilot key whose SHA-256 is stored below. This migration creates NO key: the auth row
-- starts with a NULL hash and enabled = false, so every ingest is refused until the operator
-- explicitly configures one. Nothing here starts the pilot, touches existing tables, or changes
-- brian_prediction_control (wallet_tracking_enabled stays as it is).

-- ---------------------------------------------------------------------------------------------
-- 1. Pilot ingest credential (hash only, single row, disabled by default)
-- ---------------------------------------------------------------------------------------------
create table if not exists public.brian_niulai4_pilot_auth (
  auth_id text primary key default 'pilot-ingest-v1' check (auth_id = 'pilot-ingest-v1'),
  ingest_key_sha256 text check (ingest_key_sha256 is null or ingest_key_sha256 ~ '^[0-9a-f]{64}$'),
  enabled boolean not null default false,
  configured_at timestamptz,
  updated_at timestamptz not null default now(),
  check (not enabled or (ingest_key_sha256 is not null and configured_at is not null))
);
insert into public.brian_niulai4_pilot_auth (auth_id) values ('pilot-ingest-v1') on conflict (auth_id) do nothing;
alter table public.brian_niulai4_pilot_auth enable row level security;
revoke all on public.brian_niulai4_pilot_auth from public, anon, authenticated, service_role;
grant select on public.brian_niulai4_pilot_auth to service_role;

-- ---------------------------------------------------------------------------------------------
-- 2. Runs, heartbeats, reports (append-mostly, shadow-only by constraint)
-- ---------------------------------------------------------------------------------------------
create table if not exists public.brian_niulai4_pilot_runs (
  run_id text primary key check (run_id ~ '^[A-Za-z0-9_.:-]{8,80}$'),
  rules_version text not null check (rules_version in ('niulai4-pilot.v2')),
  rules_fingerprint text not null check (rules_fingerprint ~ '^[0-9a-f]{64}$'),
  host_label text check (host_label is null or length(host_label) <= 40),
  started_at timestamptz not null,
  last_heartbeat_at timestamptz,
  ended_at timestamptz,
  state text not null default 'RUNNING' check (state in ('RUNNING', 'STOPPED', 'COMPLETED')),
  stop_reason text check (stop_reason is null or length(stop_reason) <= 64),
  created_at timestamptz not null default now(),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  trade_signal boolean not null default false check (not trade_signal),
  check ((state = 'RUNNING') = (ended_at is null))
);

create table if not exists public.brian_niulai4_pilot_heartbeats (
  run_id text not null references public.brian_niulai4_pilot_runs(run_id),
  seq integer not null check (seq >= 0),
  observed_at timestamptz not null,
  received_at timestamptz not null default now(),
  health jsonb not null check (jsonb_typeof(health) = 'object' and octet_length(health::text) <= 16384),
  primary key (run_id, seq)
);
create index if not exists brian_niulai4_pilot_heartbeats_time_idx
  on public.brian_niulai4_pilot_heartbeats(run_id, observed_at desc);

create table if not exists public.brian_niulai4_pilot_reports (
  run_id text not null references public.brian_niulai4_pilot_runs(run_id),
  seq integer not null check (seq >= 0),
  kind text not null check (kind in ('INTERIM', 'FINAL')),
  observed_at timestamptz not null,
  received_at timestamptz not null default now(),
  summary jsonb not null check (
    jsonb_typeof(summary) = 'object'
    and octet_length(summary::text) <= 262144
    and summary->>'tradeSignal' = 'false'
  ),
  primary key (run_id, seq)
);
create index if not exists brian_niulai4_pilot_reports_time_idx
  on public.brian_niulai4_pilot_reports(run_id, observed_at desc);

alter table public.brian_niulai4_pilot_runs enable row level security;
alter table public.brian_niulai4_pilot_heartbeats enable row level security;
alter table public.brian_niulai4_pilot_reports enable row level security;
revoke all on public.brian_niulai4_pilot_runs, public.brian_niulai4_pilot_heartbeats, public.brian_niulai4_pilot_reports
  from public, anon, authenticated, service_role;
grant select on public.brian_niulai4_pilot_runs, public.brian_niulai4_pilot_heartbeats, public.brian_niulai4_pilot_reports
  to service_role;

-- ---------------------------------------------------------------------------------------------
-- 3. Ingest: the only write path. Validates everything; never trusts the client's clock blindly.
-- ---------------------------------------------------------------------------------------------
create or replace function public.brian_record_niulai4_pilot_v1(p_payload jsonb)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_kind text := p_payload->>'kind';
  v_run text := p_payload->>'run_id';
  v_seq integer;
  v_obs timestamptz;
  v_run_row public.brian_niulai4_pilot_runs%rowtype;
  v_other text;
  v_last timestamptz;
begin
  if p_payload is null or jsonb_typeof(p_payload) <> 'object' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'PAYLOAD_NOT_OBJECT');
  end if;
  if v_kind not in ('START', 'HEARTBEAT', 'REPORT', 'STOP') then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'UNKNOWN_KIND');
  end if;
  if v_run is null or v_run !~ '^[A-Za-z0-9_.:-]{8,80}$' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_RUN_ID');
  end if;
  begin
    v_obs := (p_payload->>'observed_at')::timestamptz;
  exception when others then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_OBSERVED_AT');
  end;
  -- The pilot's clock is NTP-corrected; anything far from the server clock is a bug, not data.
  if v_obs is null or v_obs > now() + interval '2 minutes' or v_obs < now() - interval '1 hour' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'OBSERVED_AT_OUT_OF_RANGE');
  end if;
  if coalesce(p_payload->>'shadow_only', '') <> 'true' or coalesce(p_payload->>'trade_signal', '') <> 'false' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'NOT_SHADOW_ONLY');
  end if;

  select * into v_run_row from public.brian_niulai4_pilot_runs where run_id = v_run for update;

  if v_kind = 'START' then
    if found then
      return jsonb_build_object('status', 'DUPLICATE', 'run_id', v_run);
    end if;
    -- One live run at a time: another RUNNING run with a recent heartbeat blocks a new start.
    select run_id into v_other from public.brian_niulai4_pilot_runs
      where state = 'RUNNING' and coalesce(last_heartbeat_at, started_at) > now() - interval '10 minutes'
      limit 1;
    if v_other is not null then
      return jsonb_build_object('status', 'REJECTED', 'reason', 'ANOTHER_RUN_ACTIVE', 'active_run_id', v_other);
    end if;
    -- A silent RUNNING run is closed as STOPPED/HEARTBEAT_LOST before the new one starts.
    update public.brian_niulai4_pilot_runs
      set state = 'STOPPED', stop_reason = 'HEARTBEAT_LOST', ended_at = coalesce(last_heartbeat_at, started_at)
      where state = 'RUNNING';
    begin
      insert into public.brian_niulai4_pilot_runs (run_id, rules_version, rules_fingerprint, host_label, started_at, last_heartbeat_at)
      values (v_run, p_payload->>'rules_version', p_payload->>'rules_fingerprint',
              nullif(left(coalesce(p_payload->>'host_label', ''), 40), ''), v_obs, v_obs);
    exception when check_violation or not_null_violation then
      return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_RUN_FIELDS');
    end;
    return jsonb_build_object('status', 'STORED', 'kind', v_kind, 'run_id', v_run);
  end if;

  if not found then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'UNKNOWN_RUN');
  end if;
  if v_run_row.state <> 'RUNNING' then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'RUN_NOT_RUNNING', 'state', v_run_row.state);
  end if;
  if coalesce(p_payload->>'rules_fingerprint', '') <> v_run_row.rules_fingerprint then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'FINGERPRINT_MISMATCH');
  end if;

  if v_kind = 'STOP' then
    update public.brian_niulai4_pilot_runs
      set state = case when p_payload->>'stop_reason' = 'DURATION_ELAPSED' then 'COMPLETED' else 'STOPPED' end,
          stop_reason = left(coalesce(p_payload->>'stop_reason', 'UNKNOWN'), 64),
          ended_at = greatest(v_obs, started_at),
          last_heartbeat_at = greatest(coalesce(last_heartbeat_at, v_obs), v_obs)
      where run_id = v_run;
    return jsonb_build_object('status', 'STORED', 'kind', v_kind, 'run_id', v_run);
  end if;

  begin
    v_seq := (p_payload->>'seq')::integer;
  exception when others then
    v_seq := null;
  end;
  if v_seq is null or v_seq < 0 then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_SEQ');
  end if;

  if v_kind = 'HEARTBEAT' then
    -- Rate limit: at most one heartbeat per 30 s per run (the pilot sends one per 60 s).
    select max(received_at) into v_last from public.brian_niulai4_pilot_heartbeats where run_id = v_run;
    if v_last is not null and v_last > now() - interval '30 seconds' then
      return jsonb_build_object('status', 'RATE_LIMITED', 'retry_after_seconds', 30);
    end if;
    begin
      insert into public.brian_niulai4_pilot_heartbeats (run_id, seq, observed_at, health)
      values (v_run, v_seq, v_obs, p_payload->'health')
      on conflict (run_id, seq) do nothing;
    exception when check_violation or not_null_violation then
      return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_HEALTH');
    end;
    update public.brian_niulai4_pilot_runs set last_heartbeat_at = greatest(coalesce(last_heartbeat_at, v_obs), v_obs)
      where run_id = v_run;
    -- Bounded storage for this pilot's own rows only: keep 30 days of heartbeats.
    delete from public.brian_niulai4_pilot_heartbeats where run_id = v_run and observed_at < now() - interval '30 days';
    return jsonb_build_object('status', 'STORED', 'kind', v_kind, 'run_id', v_run, 'seq', v_seq);
  end if;

  -- REPORT: at most one per 5 minutes per run; FINAL only once.
  if p_payload->>'report_kind' not in ('INTERIM', 'FINAL') then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_REPORT_KIND');
  end if;
  select max(received_at) into v_last from public.brian_niulai4_pilot_reports where run_id = v_run;
  if v_last is not null and v_last > now() - interval '5 minutes' then
    return jsonb_build_object('status', 'RATE_LIMITED', 'retry_after_seconds', 300);
  end if;
  if p_payload->>'report_kind' = 'FINAL'
     and exists (select 1 from public.brian_niulai4_pilot_reports where run_id = v_run and kind = 'FINAL') then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'FINAL_ALREADY_STORED');
  end if;
  begin
    insert into public.brian_niulai4_pilot_reports (run_id, seq, kind, observed_at, summary)
    values (v_run, v_seq, p_payload->>'report_kind', v_obs, p_payload->'summary')
    on conflict (run_id, seq) do nothing;
  exception when check_violation or not_null_violation then
    return jsonb_build_object('status', 'REJECTED', 'reason', 'BAD_SUMMARY');
  end;
  return jsonb_build_object('status', 'STORED', 'kind', v_kind, 'run_id', v_run, 'seq', v_seq);
end;
$$;
revoke all on function public.brian_record_niulai4_pilot_v1(jsonb) from public, anon, authenticated;
grant execute on function public.brian_record_niulai4_pilot_v1(jsonb) to service_role;

-- ---------------------------------------------------------------------------------------------
-- 4. Read-only status for the UI. States are derived from facts, never assumed:
--    NOT_CONFIGURED  no enabled ingest key                     (nothing can arrive)
--    NOT_STARTED     key configured, no run recorded
--    RUNNING         RUNNING run, heartbeat <= 180 s old
--    STALE           RUNNING run, heartbeat older than 180 s   (process down or sink failing)
--    STOPPED/COMPLETED  last run ended (stop_reason says why)
-- ---------------------------------------------------------------------------------------------
create or replace function public.brian_niulai4_pilot_status_v1(p_as_of timestamptz default now())
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with a as (
    select coalesce(bool_or(enabled and ingest_key_sha256 is not null), false) as configured
    from public.brian_niulai4_pilot_auth
  ),
  r as (
    select * from public.brian_niulai4_pilot_runs
    where started_at <= p_as_of
    order by started_at desc, run_id desc limit 1
  ),
  hb as (
    select h.* from public.brian_niulai4_pilot_heartbeats h, r
    where h.run_id = r.run_id and h.observed_at <= p_as_of
    order by h.observed_at desc, h.seq desc limit 1
  ),
  rep as (
    select p.* from public.brian_niulai4_pilot_reports p, r
    where p.run_id = r.run_id and p.observed_at <= p_as_of
    order by (p.kind = 'FINAL') desc, p.observed_at desc, p.seq desc limit 1
  ),
  counts as (
    select (select count(*) from public.brian_niulai4_pilot_heartbeats h, r where h.run_id = r.run_id) as heartbeats,
           (select count(*) from public.brian_niulai4_pilot_reports p, r where p.run_id = r.run_id) as reports
  )
  select jsonb_build_object(
    'as_of', p_as_of,
    'status', case
      when not (select configured from a) and not exists (select 1 from r) then 'NOT_CONFIGURED'
      when not exists (select 1 from r) then 'NOT_STARTED'
      when (select state from r) = 'RUNNING'
        and coalesce((select observed_at from hb), (select started_at from r)) > p_as_of - interval '180 seconds' then 'RUNNING'
      when (select state from r) = 'RUNNING' then 'STALE'
      else (select state from r)
    end,
    'ingest_configured', (select configured from a),
    'stale_after_seconds', 180,
    'run', (select jsonb_build_object(
      'run_id', run_id, 'rules_version', rules_version, 'rules_fingerprint', rules_fingerprint,
      'host_label', host_label, 'state', state, 'stop_reason', stop_reason,
      'started_at', started_at, 'ended_at', ended_at, 'last_heartbeat_at', last_heartbeat_at) from r),
    'last_heartbeat', (select jsonb_build_object('seq', seq, 'observed_at', observed_at, 'received_at', received_at,
      'age_seconds', round(extract(epoch from (p_as_of - observed_at))::numeric, 1), 'health', health) from hb),
    'latest_report', (select jsonb_build_object('seq', seq, 'kind', kind, 'observed_at', observed_at,
      'received_at', received_at, 'summary', summary) from rep),
    'counts', (select to_jsonb(c) from counts c),
    'shadow_only', true, 'live_execution', false, 'trade_signal', false
  );
$$;
revoke all on function public.brian_niulai4_pilot_status_v1(timestamptz) from public, anon, authenticated;
grant execute on function public.brian_niulai4_pilot_status_v1(timestamptz) to service_role;

-- ---------------------------------------------------------------------------------------------
-- 5. Read-only Prediction Intelligence status for the UI (wraps existing functions; no writes).
--    status: DISABLED (module off) | NO_DATA (never collected) | STALE (no cycle in 40 min,
--    schedule is every 15 min) | DEGRADED (last cycle had non-OK sources) | OK
-- ---------------------------------------------------------------------------------------------
create or replace function public.brian_prediction_status_v1(p_as_of timestamptz default now())
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with c as (select * from public.brian_prediction_control where control_id = 'prediction-intelligence'),
  last_h as (
    select max(observed_at) as at from public.brian_prediction_source_health where observed_at <= p_as_of
  ),
  last_cycle as (
    select h.status, h.source from public.brian_prediction_source_health h, last_h
    where last_h.at is not null and h.observed_at > last_h.at - interval '2 minutes' and h.observed_at <= last_h.at
  ),
  snaps as (
    select count(*) as total, max(received_at) as last_at,
           count(*) filter (where received_at > p_as_of - interval '24 hours') as last_24h
    from public.brian_prediction_market_snapshots where received_at <= p_as_of
  ),
  res as (
    select count(*) as total, max(received_at) as last_at from public.brian_prediction_resolutions where received_at <= p_as_of
  )
  select jsonb_build_object(
    'as_of', p_as_of,
    'status', case
      when not coalesce((select module_enabled from c), false) then 'DISABLED'
      when (select at from last_h) is null then 'NO_DATA'
      when (select at from last_h) < p_as_of - interval '40 minutes' then 'STALE'
      when exists (select 1 from last_cycle where status <> 'OK') then 'DEGRADED'
      else 'OK'
    end,
    'stale_after_seconds', 2400,
    'control', (select jsonb_build_object(
      'module_enabled', module_enabled, 'wallet_tracking_enabled', wallet_tracking_enabled,
      'max_markets', max_markets, 'tag_slugs', to_jsonb(tag_slugs), 'max_storage_mb', max_storage_mb,
      'updated_at', updated_at) from c),
    'last_cycle_at', (select at from last_h),
    'last_cycle_sources', coalesce((select jsonb_agg(jsonb_build_object('source', source, 'status', status) order by source) from last_cycle), '[]'::jsonb),
    'snapshots', (select jsonb_build_object('total', total, 'last_24h', last_24h, 'last_received_at', last_at) from snaps),
    'resolutions', (select jsonb_build_object('total', total, 'last_received_at', last_at) from res),
    'storage', jsonb_build_object(
      'bytes', public.brian_prediction_storage_bytes(),
      'cap_bytes', (select max_storage_mb::bigint * 1024 * 1024 from c)),
    'health_24h', public.brian_prediction_health_summary_v1(p_as_of, interval '24 hours'),
    'shadow_only', true, 'live_execution', false, 'trade_signal', false
  );
$$;
revoke all on function public.brian_prediction_status_v1(timestamptz) from public, anon, authenticated;
grant execute on function public.brian_prediction_status_v1(timestamptz) to service_role;
