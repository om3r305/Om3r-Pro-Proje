-- Brian core recovery dispatcher hardening (2026-09-14)
-- Keeps DIP isolated while serializing stale core recovery work and preventing
-- PostgREST idle-in-transaction sessions from accumulating indefinitely.

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
begin
  if not pg_try_advisory_xact_lock(hashtextextended('brian-core-recovery-dispatcher-v3', 0)) then
    return jsonb_build_object('status','BUSY');
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

-- Core jobs are dispatched serially by brian-core-recovery-dispatcher-v1.
update cron.job
set active = false
where jobname in (
  'brian-alpha-decision-compiler-1m',
  'brian-evolution-orchestrator-10m',
  'brian-world-brain-5m',
  'brian-world-discovery-eye-10m',
  'brian-evolution-researcher-30m',
  'brian-evolution-sandbox-30m',
  'brian-evolution-treasury-1m',
  'brian-evolution-ocean-5m'
);

-- DIP remains independent and untouched.
update cron.job set active = true where jobname = 'brian-dip-multiasset-worker-15s';

do $do$
begin
  if exists (select 1 from cron.job where jobname='brian-core-recovery-dispatcher-v1') then
    update cron.job
      set active=true,
          schedule='20 seconds',
          command='select brian_private.dispatch_stale_core();'
      where jobname='brian-core-recovery-dispatcher-v1';
  else
    perform cron.schedule(
      'brian-core-recovery-dispatcher-v1',
      '20 seconds',
      'select brian_private.dispatch_stale_core();'
    );
  end if;
end
$do$;
