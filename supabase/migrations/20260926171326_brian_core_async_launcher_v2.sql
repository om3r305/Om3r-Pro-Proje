-- APPLY TO: brian-market-intelligence only.
-- Make Core lane launch non-blocking by queueing brian-core-launcher through pg_net.

create or replace function brian_private.launch_core_lane(p_service text)
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog','public','brian_private','net','vault','extensions'
as $function$
declare
  now_ts timestamptz:=clock_timestamp();
  svc record;
  last_success timestamptz;
  last_attempt timestamptz;
  project_url text;
  anon_jwt text;
  cron_key text;
  dashboard_cron_key text;
  req_id bigint;
  queue_depth integer := 0;
  enabled boolean:=false;
begin
  select coalesce(lower(config_value)='true',false)
  into enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  if not coalesce(enabled,false) then
    return jsonb_build_object('status','SYSTEM_DISABLED','service',p_service,'observed_at',now_ts);
  end if;

  select *
  into svc
  from (values
    ('alpha','brian-alpha-decision-compiler-v2',120,120,true),
    ('world','brian-world-brain-v1',240,180,false),
    ('treasury','brian-evolution-treasury-v1',180,180,false),
    ('discovery','brian-world-discovery-eye-v1',480,300,false),
    ('evolution','brian-evolution-orchestrator-v1',600,300,false),
    ('ocean','brian-evolution-ocean-worker-v1',600,300,false),
    ('researcher','brian-evolution-researcher-v1',1500,600,false),
    ('sandbox','brian-evolution-sandbox-v1',1500,600,false)
  ) s(service_id,collector_id,max_age_seconds,retry_seconds,use_dashboard_key)
  where s.service_id=p_service;

  if not found then
    return jsonb_build_object('status','UNKNOWN_SERVICE','service',p_service);
  end if;

  if not pg_try_advisory_xact_lock(hashtextextended('brian-core-launch-'||svc.service_id,0)) then
    return jsonb_build_object('status','BUSY','service',svc.service_id,'observed_at',now_ts);
  end if;

  select max(finished_at)
  into last_success
  from public.brian_collector_runs
  where collector_id=svc.collector_id and status in ('SUCCESS','DEGRADED');

  if last_success is not null
     and last_success>=now_ts-make_interval(secs=>svc.max_age_seconds) then
    return jsonb_build_object('status','FRESH','service',svc.service_id,'last_success_at',last_success,'observed_at',now_ts);
  end if;

  select max(last_attempt_at)
  into last_attempt
  from brian_private.core_dispatch_state
  where service_id=svc.service_id;

  if last_attempt is not null
     and last_attempt>=now_ts-make_interval(secs=>svc.retry_seconds) then
    return jsonb_build_object('status','RETRY_GUARD','service',svc.service_id,'last_attempt_at',last_attempt,'observed_at',now_ts);
  end if;

  select count(*)::int into queue_depth from net.http_request_queue;
  if queue_depth>=3 then
    return jsonb_build_object('status','BACKPRESSURE_QUEUE','service',svc.service_id,'queue_depth',queue_depth,'observed_at',now_ts);
  end if;

  select decrypted_secret into project_url from vault.decrypted_secrets where name='brian_project_url' limit 1;
  select decrypted_secret into anon_jwt from vault.decrypted_secrets where name='brian_anon_jwt' limit 1;
  select decrypted_secret into cron_key from vault.decrypted_secrets where name='brian_cron_key' limit 1;
  select decrypted_secret into dashboard_cron_key from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1;

  if project_url is null or anon_jwt is null or cron_key is null or dashboard_cron_key is null then
    raise exception 'BRIAN_CORE_LAUNCH_SECRETS_UNAVAILABLE';
  end if;

  req_id:=net.http_post(
    url:=project_url||'/functions/v1/brian-core-launcher',
    headers:=jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer '||anon_jwt,
      'apikey',anon_jwt,
      'x-brian-cron-key',cron_key,
      'x-brian-downstream-key',case when svc.use_dashboard_key then dashboard_cron_key else cron_key end
    ),
    body:=jsonb_build_object('service',svc.service_id),
    timeout_milliseconds:=30000
  );

  insert into brian_private.core_dispatch_state(
    service_id,last_attempt_at,last_request_id,attempt_count,last_http_status,last_response_at
  )
  values(svc.service_id,now_ts,req_id,1,null,null)
  on conflict(service_id) do update
  set last_attempt_at=excluded.last_attempt_at,
      last_request_id=excluded.last_request_id,
      last_http_status=null,
      last_response_at=null,
      attempt_count=brian_private.core_dispatch_state.attempt_count+1;

  return jsonb_build_object(
    'status','ENQUEUED','service',svc.service_id,'request_id',req_id,
    'queue_depth',queue_depth,'last_success_at',last_success,'observed_at',clock_timestamp()
  );
exception when others then
  return jsonb_build_object('status','LAUNCH_EXCEPTION','service',p_service,'error',sqlerrm,'observed_at',clock_timestamp());
end;
$function$;
