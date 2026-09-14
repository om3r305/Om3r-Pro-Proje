-- Brian core resilience final migration (2026-09-14)
-- Goals:
-- 1) keep DIP isolated and untouched,
-- 2) serialize heavy core collectors behind one backpressure-aware dispatcher,
-- 3) make collector lease acquire retry-safe,
-- 4) keep operator start/stop/restart semantics correct,
-- 5) prevent stuck PostgREST idle-in-transaction sessions from accumulating.

create schema if not exists brian_private;

create table if not exists brian_private.core_dispatch_state (
  service_id text primary key,
  last_attempt_at timestamptz,
  last_request_id bigint,
  last_http_status integer,
  last_response_at timestamptz,
  attempt_count bigint not null default 0
);

alter role authenticator set idle_in_transaction_session_timeout = '45s';

-- Retry-safe acquire: if a successful HTTP response is lost, the same owner token
-- can call acquire again without turning its own live lease into false contention.
create or replace function public.brian_acquire_collector_lease(
  p_collector_id text,
  p_owner_token text,
  p_lease_seconds integer
)
returns boolean
language plpgsql
security definer
set search_path to 'pg_catalog', 'public'
as $function$
declare
  v_now timestamptz := clock_timestamp();
  v_prior_owner text;
  v_prior_lease_until timestamptz;
  v_had_prior boolean;
  v_rows integer;
  v_acquired boolean;
begin
  if p_collector_id is null or length(trim(p_collector_id))=0 then
    raise exception 'BRIAN_LEASE: p_collector_id is required';
  end if;
  if p_owner_token is null or length(trim(p_owner_token))=0 then
    raise exception 'BRIAN_LEASE: p_owner_token is required';
  end if;
  if p_lease_seconds is null or p_lease_seconds<=0 then
    raise exception 'BRIAN_LEASE: p_lease_seconds must be positive';
  end if;

  select owner_token, lease_until
    into v_prior_owner, v_prior_lease_until
  from public.brian_collector_leases
  where collector_id=p_collector_id
  for update;
  v_had_prior := found;

  if v_had_prior
     and v_prior_owner = p_owner_token
     and v_prior_lease_until > v_now then
    update public.brian_collector_leases
      set lease_until=v_now+make_interval(secs=>p_lease_seconds),
          updated_at=v_now
    where collector_id=p_collector_id and owner_token=p_owner_token;

    insert into public.brian_collector_lease_events(
      collector_id,owner_token,event,observed_at,lease_until,metadata
    ) values (
      p_collector_id,p_owner_token,'RENEWED',v_now,
      v_now+make_interval(secs=>p_lease_seconds),
      jsonb_build_object('lease_seconds',p_lease_seconds,'via','acquire_retry')
    );
    return true;
  end if;

  insert into public.brian_collector_leases(
    collector_id,owner_token,acquired_at,lease_until,updated_at
  ) values (
    p_collector_id,p_owner_token,v_now,
    v_now+make_interval(secs=>p_lease_seconds),v_now
  )
  on conflict(collector_id) do update
    set owner_token=excluded.owner_token,
        acquired_at=excluded.acquired_at,
        lease_until=excluded.lease_until,
        updated_at=excluded.updated_at
  where public.brian_collector_leases.lease_until<=v_now;

  get diagnostics v_rows=row_count;
  v_acquired := v_rows>0;

  insert into public.brian_collector_lease_events(
    collector_id,owner_token,event,observed_at,lease_until,metadata
  ) values (
    p_collector_id,p_owner_token,
    case
      when not v_acquired then 'BLOCKED_ACTIVE'
      when v_had_prior and v_prior_lease_until<=v_now then 'EXPIRED_RECOVERY'
      else 'ACQUIRED'
    end,
    v_now,
    case when v_acquired then v_now+make_interval(secs=>p_lease_seconds) else v_prior_lease_until end,
    jsonb_build_object('lease_seconds',p_lease_seconds,'had_prior_lease',v_had_prior)
  );

  return v_acquired;
end;
$function$;

create or replace function brian_private.dispatch_stale_core()
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog', 'public', 'brian_private', 'net', 'vault', 'extensions'
as $function$
declare
  now_ts timestamptz := clock_timestamp();
  svc record;
  project_url text;
  anon_jwt text;
  cron_key text;
  dashboard_cron_key text;
  req_id bigint;
  queue_depth integer := 0;
  inflight record;
  system_enabled boolean := false;
begin
  if not pg_try_advisory_xact_lock(hashtextextended('brian-core-recovery-dispatcher-v3', 0)) then
    return jsonb_build_object('status','BUSY');
  end if;

  select coalesce(lower(config_value)='true',false)
    into system_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  if not coalesce(system_enabled,false) then
    return jsonb_build_object('status','SYSTEM_DISABLED','observed_at',now_ts);
  end if;

  update brian_private.core_dispatch_state s
  set last_http_status = r.status_code,
      last_response_at = r.created
  from net._http_response r
  where s.last_request_id = r.id
    and (s.last_response_at is null or s.last_response_at < r.created);

  select count(*)::int into queue_depth from net.http_request_queue;
  if queue_depth >= 4 then
    return jsonb_build_object('status','BACKPRESSURE_QUEUE','queue_depth',queue_depth,'observed_at',now_ts);
  end if;

  select d.service_id,d.last_attempt_at,d.last_request_id
  into inflight
  from brian_private.core_dispatch_state d
  where d.last_attempt_at is not null
    and d.last_attempt_at > coalesce(d.last_response_at,'epoch'::timestamptz)
    and d.last_attempt_at > now_ts - interval '180 seconds'
  order by d.last_attempt_at desc
  limit 1;

  if found then
    return jsonb_build_object(
      'status','BACKPRESSURE_INFLIGHT',
      'service_id',inflight.service_id,
      'request_id',inflight.last_request_id,
      'attempt_age_seconds',round(extract(epoch from (now_ts-inflight.last_attempt_at))::numeric,1),
      'observed_at',now_ts
    );
  end if;

  with services(service_id, collector_id, endpoint, body, max_age_seconds, retry_seconds, priority, use_dashboard_key, timeout_ms) as (
    values
      ('alpha','brian-alpha-decision-compiler-v2','brian-alpha-decision-compiler','{}'::jsonb,180,180,10,true,120000),
      ('treasury','brian-evolution-treasury-v1','brian-evolution-treasury','{}'::jsonb,300,180,20,false,90000),
      ('world','brian-world-brain-v1','brian-world-brain','{}'::jsonb,600,240,30,false,150000),
      ('evolution','brian-evolution-orchestrator-v1','brian-evolution-orchestrator','{}'::jsonb,900,180,40,false,90000),
      ('ocean','brian-evolution-ocean-worker-v1','brian-evolution-ocean-worker','{}'::jsonb,900,180,50,false,70000),
      ('discovery','brian-world-discovery-eye-v1','brian-world-discovery-eye','{}'::jsonb,1200,240,60,false,120000),
      ('researcher','brian-evolution-researcher-v1','brian-evolution-researcher','{}'::jsonb,1800,300,70,false,90000),
      ('sandbox','brian-evolution-sandbox-v1','brian-evolution-sandbox','{"action":"plan"}'::jsonb,1800,240,80,false,70000)
  ), candidates as (
    select s.*,
           l.last_success_at,
           d.last_attempt_at,
           d.last_request_id,
           case when l.last_success_at is null then 1e12
                else extract(epoch from (now_ts-l.last_success_at))/greatest(s.max_age_seconds,1)
           end as overdue_ratio
    from services s
    left join lateral (
      select max(r.finished_at) as last_success_at
      from public.brian_collector_runs r
      where r.collector_id=s.collector_id and r.status='SUCCESS'
    ) l on true
    left join brian_private.core_dispatch_state d on d.service_id=s.service_id
    where (l.last_success_at is null or l.last_success_at < now_ts - make_interval(secs => s.max_age_seconds))
      and (d.last_attempt_at is null or d.last_attempt_at < now_ts - make_interval(secs => s.retry_seconds))
  )
  select * into svc
  from candidates
  order by priority asc, overdue_ratio desc, last_attempt_at asc nulls first
  limit 1;

  if not found then
    return jsonb_build_object('status','IDLE','queue_depth',queue_depth,'observed_at',now_ts);
  end if;

  select decrypted_secret into project_url from vault.decrypted_secrets where name='brian_project_url' limit 1;
  select decrypted_secret into anon_jwt from vault.decrypted_secrets where name='brian_anon_jwt' limit 1;
  select decrypted_secret into cron_key from vault.decrypted_secrets where name='brian_cron_key' limit 1;
  select decrypted_secret into dashboard_cron_key from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1;

  if project_url is null or anon_jwt is null or cron_key is null or dashboard_cron_key is null then
    raise exception 'BRIAN_DISPATCH_SECRETS_UNAVAILABLE';
  end if;

  req_id := net.http_post(
    url := project_url || '/functions/v1/' || svc.endpoint,
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || anon_jwt,
      'apikey',anon_jwt,
      'x-brian-cron-key',case when svc.use_dashboard_key then dashboard_cron_key else cron_key end
    ),
    body := svc.body,
    timeout_milliseconds := svc.timeout_ms
  );

  insert into brian_private.core_dispatch_state(service_id,last_attempt_at,last_request_id,attempt_count)
  values (svc.service_id,now_ts,req_id,1)
  on conflict(service_id) do update set
    last_attempt_at=excluded.last_attempt_at,
    last_request_id=excluded.last_request_id,
    attempt_count=brian_private.core_dispatch_state.attempt_count+1;

  return jsonb_build_object(
    'status','ENQUEUED',
    'service_id',svc.service_id,
    'collector_id',svc.collector_id,
    'request_id',req_id,
    'last_success_at',svc.last_success_at,
    'queue_depth',queue_depth,
    'timeout_ms',svc.timeout_ms,
    'observed_at',now_ts
  );
end;
$function$;

-- Start/stop/restart keeps direct core cron jobs disabled and toggles the dispatcher instead.
create or replace function public.brian_set_system_enabled(p_enabled boolean)
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog', 'public', 'cron'
as $function$
declare
  v_job record;
  v_changed integer := 0;
  v_total integer := 0;
  v_dispatcher record;
  v_core_jobs text[] := array[
    'brian-alpha-decision-compiler-1m',
    'brian-evolution-orchestrator-10m',
    'brian-world-brain-5m',
    'brian-world-discovery-eye-10m',
    'brian-evolution-researcher-30m',
    'brian-evolution-sandbox-30m',
    'brian-evolution-treasury-1m',
    'brian-evolution-ocean-5m'
  ]::text[];
begin
  for v_job in
    select j.jobid, j.jobname, j.active
    from public.brian_system_job_registry r
    join cron.job j on j.jobname=r.job_name
    where r.operator_managed and j.jobname = any(v_core_jobs)
  loop
    if v_job.active then
      perform cron.alter_job(v_job.jobid, null, null, null, null, false);
      v_changed := v_changed + 1;
    end if;
  end loop;

  for v_job in
    select j.jobid, j.jobname, j.active
    from public.brian_system_job_registry r
    join cron.job j on j.jobname=r.job_name
    where r.operator_managed
      and j.jobname not like 'brian-dip-%'
      and not (j.jobname = any(v_core_jobs))
    order by j.jobid
  loop
    if v_job.active is distinct from p_enabled then
      perform cron.alter_job(v_job.jobid, null, null, null, null, p_enabled);
      v_changed := v_changed + 1;
    end if;
  end loop;

  select jobid,active into v_dispatcher
  from cron.job
  where jobname='brian-core-recovery-dispatcher-v1'
  limit 1;

  if found and v_dispatcher.active is distinct from p_enabled then
    perform cron.alter_job(v_dispatcher.jobid, null, null, null, null, p_enabled);
    v_changed := v_changed + 1;
  end if;

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
    'core_dispatcher_managed',true,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false
  );
end;
$function$;

-- Report dispatcher-covered jobs as effectively active while preserving truthful diagnostics.
create or replace function public.brian_system_control_status()
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog', 'public', 'cron'
as $function$
declare
  v_total integer := 0;
  v_active integer := 0;
  v_missing text[] := '{}';
  v_target numeric := 10000;
  v_latest record;
  v_enabled boolean := false;
  v_dispatcher_active boolean := false;
  v_core_direct_active integer := 0;
  v_core_count integer := 0;
  v_core_jobs text[] := array[
    'brian-alpha-decision-compiler-1m',
    'brian-evolution-orchestrator-10m',
    'brian-world-brain-5m',
    'brian-world-discovery-eye-10m',
    'brian-evolution-researcher-30m',
    'brian-evolution-sandbox-30m',
    'brian-evolution-treasury-1m',
    'brian-evolution-ocean-5m'
  ]::text[];
begin
  select case when lower(config_value)='true' then true else false end
    into v_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  select coalesce(active,false) into v_dispatcher_active
  from cron.job
  where jobname='brian-core-recovery-dispatcher-v1'
  limit 1;

  select count(*)::integer,
         coalesce(array_agg(r.job_name order by r.job_name) filter (where j.jobid is null), '{}')
    into v_total, v_missing
  from public.brian_system_job_registry r
  left join cron.job j on j.jobname=r.job_name
  where r.operator_managed;

  select count(*)::integer into v_core_count
  from public.brian_system_job_registry r
  where r.operator_managed and r.job_name = any(v_core_jobs);

  select count(*)::integer into v_core_direct_active
  from cron.job j
  where j.jobname = any(v_core_jobs) and j.active;

  select count(*)::integer into v_active
  from public.brian_system_job_registry r
  join cron.job j on j.jobname=r.job_name
  where r.operator_managed
    and not (r.job_name = any(v_core_jobs))
    and j.active;

  if v_dispatcher_active and coalesce(v_enabled,false) then
    v_active := v_active + v_core_count;
  else
    v_active := v_active + v_core_direct_active;
  end if;

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
    'core_dispatcher_active',coalesce(v_dispatcher_active,false),
    'core_direct_jobs_active',coalesce(v_core_direct_active,0),
    'core_dispatcher_managed',true,
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
$function$;

-- Ensure exactly one dispatcher exists, then enforce the safe topology with cron.alter_job
-- (direct writes to cron.job are intentionally avoided).
do $do$
declare
  v_job record;
  v_enabled boolean := false;
begin
  select coalesce(lower(config_value)='true',false)
    into v_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  if not exists (select 1 from cron.job where jobname='brian-core-recovery-dispatcher-v1') then
    perform cron.schedule(
      'brian-core-recovery-dispatcher-v1',
      '20 seconds',
      'select brian_private.dispatch_stale_core();'
    );
  end if;

  for v_job in
    select jobid,active from cron.job
    where jobname in (
      'brian-alpha-decision-compiler-1m',
      'brian-evolution-orchestrator-10m',
      'brian-world-brain-5m',
      'brian-world-discovery-eye-10m',
      'brian-evolution-researcher-30m',
      'brian-evolution-sandbox-30m',
      'brian-evolution-treasury-1m',
      'brian-evolution-ocean-5m'
    )
  loop
    if v_job.active then
      perform cron.alter_job(v_job.jobid, null, null, null, null, false);
    end if;
  end loop;

  for v_job in
    select jobid,active from cron.job
    where jobname='brian-core-recovery-dispatcher-v1'
  loop
    perform cron.alter_job(
      v_job.jobid,
      '20 seconds',
      'select brian_private.dispatch_stale_core();',
      null,
      null,
      v_enabled
    );
  end loop;
end
$do$;
