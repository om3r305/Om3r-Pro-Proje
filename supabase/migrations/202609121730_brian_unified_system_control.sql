-- Brian unified command control: one operator surface for MAIN/ALPHA/Evolution.
-- DIP is deliberately excluded from every job selector and control operation.
-- SHADOW ONLY. No exchange/order/withdrawal capability is introduced.

create table if not exists public.brian_system_job_registry (
  job_name text primary key,
  component text not null,
  operator_managed boolean not null default true,
  created_at timestamptz not null default now()
);

alter table public.brian_system_job_registry enable row level security;
revoke all on public.brian_system_job_registry from anon, authenticated;
revoke insert, update, delete, truncate, references, trigger on public.brian_system_job_registry from service_role;
grant select on public.brian_system_job_registry to service_role;

insert into public.brian_system_job_registry(job_name, component, operator_managed) values
  ('brian-alpha-calibration-challenger-v1-5m','ALPHA',true),
  ('brian-alpha-decision-compiler-1m','ALPHA',true),
  ('brian-alpha-reliability-feature-freeze-1m','ALPHA',true),
  ('brian-compact-retention-hourly','MAINTENANCE',true),
  ('brian-control-center-report-5m','CONTROL',true),
  ('brian-derivatives-eye-5m','SENSORS',true),
  ('brian-evolution-alpha-edge-2m','EVOLUTION',true),
  ('brian-evolution-experiment-runner-hourly','EVOLUTION',true),
  ('brian-evolution-ocean-5m','OCEAN',true),
  ('brian-evolution-orchestrator-10m','EVOLUTION',true),
  ('brian-evolution-promotion-council-hourly','EVOLUTION',true),
  ('brian-evolution-researcher-30m','RESEARCH',true),
  ('brian-evolution-sandbox-30m','RESEARCH',true),
  ('brian-evolution-template-generator-30m','RESEARCH',true),
  ('brian-evolution-treasury-1m','TREASURY',true),
  ('brian-fx-eye-hourly','SENSORS',true),
  ('brian-intrabar-eye-1m','SENSORS',true),
  ('brian-live-shadow-5m','ALPHA',true),
  ('brian-missed-opportunity-auditor-v3-5m','RESEARCH',true),
  ('brian-official-macro-eye-10m','WORLD',true),
  ('brian-retention-vacuum-daily','MAINTENANCE',true),
  ('brian-sensor-mesh-5m','SENSORS',true),
  ('brian-sensor-reliability-calibration-5m','SENSORS',true),
  ('brian-sensor-reliability-shadow-hourly','SENSORS',true),
  ('brian-storage-health-6h','MAINTENANCE',true),
  ('brian-universe-collector-15m','WORLD',true),
  ('brian-world-brain-5m','WORLD',true),
  ('brian-world-discovery-eye-10m','WORLD',true)
on conflict (job_name) do update
set component=excluded.component,
    operator_managed=excluded.operator_managed;

-- Runtime configuration table was introduced by 1720. Re-declare defensively so a
-- forward-only recovery can still apply this migration safely.
create table if not exists public.brian_evolution_runtime_config (
  config_key text primary key,
  config_value text not null,
  updated_at timestamptz not null default now()
);

insert into public.brian_evolution_runtime_config(config_key, config_value)
values
  ('brian_system_enabled','true'),
  ('treasury_target_equity_usd','10000')
on conflict (config_key) do nothing;

create or replace function public.brian_system_control_status()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public, cron
as $$
declare
  v_total integer := 0;
  v_active integer := 0;
  v_missing text[] := '{}';
  v_target numeric := 10000;
  v_latest record;
  v_enabled boolean := false;
begin
  select count(*)::integer,
         count(*) filter (where j.active)::integer,
         coalesce(array_agg(r.job_name order by r.job_name) filter (where j.jobid is null), '{}')
    into v_total, v_active, v_missing
  from public.brian_system_job_registry r
  left join cron.job j on j.jobname=r.job_name
  where r.operator_managed;

  select case when lower(config_value)='true' then true else false end
    into v_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  select nullif(config_value,'')::numeric
    into v_target
  from public.brian_evolution_runtime_config
  where config_key='treasury_target_equity_usd';

  select snapshot_id, observed_at, starting_equity_usd, cash_usd, equity_usd,
         deployment_pct, positions
    into v_latest
  from public.brian_treasury_shadow_snapshots
  order by observed_at desc, created_at desc
  limit 1;

  return jsonb_build_object(
    'system_enabled', coalesce(v_enabled,false),
    'managed_jobs_total', coalesce(v_total,0),
    'managed_jobs_active', coalesce(v_active,0),
    'missing_jobs', coalesce(to_jsonb(v_missing),'[]'::jsonb),
    'treasury_target_equity_usd', coalesce(v_target,10000),
    'treasury', case when v_latest.snapshot_id is null then null else jsonb_build_object(
      'snapshot_id',v_latest.snapshot_id,
      'observed_at',v_latest.observed_at,
      'starting_equity_usd',v_latest.starting_equity_usd,
      'cash_usd',v_latest.cash_usd,
      'equity_usd',v_latest.equity_usd,
      'deployment_pct',v_latest.deployment_pct,
      'open_positions',jsonb_array_length(coalesce(v_latest.positions,'[]'::jsonb))
    ) end,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false
  );
end;
$$;

create or replace function public.brian_set_system_enabled(p_enabled boolean)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public, cron
as $$
declare
  v_changed integer := 0;
  v_total integer := 0;
begin
  -- Only names explicitly registered above are touched. No brian-dip-* job can ever
  -- enter this operation accidentally.
  update cron.job j
     set active=p_enabled
    from public.brian_system_job_registry r
   where r.operator_managed
     and j.jobname=r.job_name
     and j.jobname not like 'brian-dip-%'
     and j.active is distinct from p_enabled;
  get diagnostics v_changed = row_count;

  select count(*)::integer into v_total
  from public.brian_system_job_registry
  where operator_managed;

  insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
  values('brian_system_enabled',case when p_enabled then 'true' else 'false' end,clock_timestamp())
  on conflict(config_key) do update
  set config_value=excluded.config_value,updated_at=excluded.updated_at;

  return jsonb_build_object(
    'status',case when p_enabled then 'RUNNING' else 'STOPPED' end,
    'changed_jobs',v_changed,
    'managed_jobs_total',v_total,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false
  );
end;
$$;

create or replace function public.brian_set_treasury_target(p_amount numeric)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
begin
  if p_amount is null or p_amount < 100 or p_amount > 1000000 then
    raise exception 'TREASURY_TARGET_OUT_OF_RANGE';
  end if;
  if round(p_amount,2) <> p_amount then
    raise exception 'TREASURY_TARGET_TOO_PRECISE';
  end if;

  insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
  values('treasury_target_equity_usd',p_amount::text,clock_timestamp())
  on conflict(config_key) do update
  set config_value=excluded.config_value,updated_at=excluded.updated_at;

  return jsonb_build_object('treasury_target_equity_usd',p_amount,'shadow_only',true,'live_execution',false);
end;
$$;

create or replace function public.brian_rebase_treasury_if_safe(p_amount numeric)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_latest record;
  v_at timestamptz := clock_timestamp();
  v_cycle_id text;
  v_snapshot_id text;
  v_system_enabled boolean := false;
begin
  if p_amount is null or p_amount < 100 or p_amount > 1000000 then
    raise exception 'TREASURY_TARGET_OUT_OF_RANGE';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('brian_evolution_treasury_chain_v1'));

  select case when lower(config_value)='true' then true else false end into v_system_enabled
  from public.brian_evolution_runtime_config where config_key='brian_system_enabled';
  if coalesce(v_system_enabled,false) then
    raise exception 'TREASURY_REBASE_REQUIRES_STOPPED_SYSTEM';
  end if;

  select snapshot_id, observed_at, starting_equity_usd, equity_usd, positions
    into v_latest
  from public.brian_treasury_shadow_snapshots
  order by observed_at desc, created_at desc
  limit 1;

  if v_latest.snapshot_id is not null and jsonb_array_length(coalesce(v_latest.positions,'[]'::jsonb)) > 0 then
    raise exception 'TREASURY_REBASE_OPEN_POSITIONS';
  end if;

  if v_latest.observed_at is not null and v_at <= v_latest.observed_at then
    v_at := v_latest.observed_at + interval '1 microsecond';
  end if;

  v_cycle_id := 'treasury-rebase:' || replace(gen_random_uuid()::text,'-','');
  v_snapshot_id := 'treasury-snapshot:' || replace(gen_random_uuid()::text,'-','');

  insert into public.brian_treasury_shadow_snapshots(
    snapshot_id,cycle_id,observed_at,previous_snapshot_id,
    starting_equity_usd,cash_usd,equity_usd,realized_pnl_usd,cumulative_costs_usd,
    deployment_usd,deployment_pct,cash_reserve_pct,positions,action_count,
    promotion_gate_open,promotion_gate_ref,promotion_gate_reason,
    treasury_version,gate_version,blocked_reasons,metadata,
    evidence_class,shadow_only,live_execution
  ) values (
    v_snapshot_id,v_cycle_id,v_at,v_latest.snapshot_id,
    p_amount,p_amount,p_amount,0,0,
    0,0,1,'[]'::jsonb,0,
    false,null,'operator shadow treasury rebase',
    'brian.treasury-shadow.v3','brian.treasury-promotion-gate.v2','{}',
    jsonb_build_object(
      'event','OPERATOR_TREASURY_REBASE',
      'previous_starting_equity_usd',v_latest.starting_equity_usd,
      'previous_equity_usd',v_latest.equity_usd,
      'dip_touched',false,
      'shadow_only',true,
      'live_execution',false
    ),
    'PROSPECTIVE_EVOLUTION_SHADOW',true,false
  );

  perform public.brian_set_treasury_target(p_amount);

  return jsonb_build_object(
    'status','REBASED',
    'snapshot_id',v_snapshot_id,
    'starting_equity_usd',p_amount,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false
  );
end;
$$;

revoke all on function public.brian_system_control_status() from public, anon, authenticated;
revoke all on function public.brian_set_system_enabled(boolean) from public, anon, authenticated;
revoke all on function public.brian_set_treasury_target(numeric) from public, anon, authenticated;
revoke all on function public.brian_rebase_treasury_if_safe(numeric) from public, anon, authenticated;
grant execute on function public.brian_system_control_status() to service_role;
grant execute on function public.brian_set_system_enabled(boolean) to service_role;
grant execute on function public.brian_set_treasury_target(numeric) to service_role;
grant execute on function public.brian_rebase_treasury_if_safe(numeric) to service_role;
