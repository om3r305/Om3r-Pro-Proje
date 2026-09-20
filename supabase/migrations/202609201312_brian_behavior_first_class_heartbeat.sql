CREATE OR REPLACE FUNCTION public.brian_external_capability_heartbeat_v1(p_capability text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
declare
  v_cap text := lower(trim(coalesce(p_capability,'')));
  v_collector text;
begin
  v_collector := case v_cap
    when 'market.universe' then 'brian-realtime-universe-heartbeat'
    when 'market.sensor-mesh' then 'brian-realtime-sensor-heartbeat'
    when 'market.intrabar' then 'brian-realtime-intrabar-heartbeat'
    when 'market.derivatives' then 'brian-realtime-derivatives-heartbeat'
    when 'market.crowd-behavior' then 'brian-realtime-crowd-behavior-heartbeat'
    when 'world.fx' then 'brian-realtime-fx-heartbeat'
    else null
  end;

  if v_collector is null then
    raise exception 'unsupported external capability: %',p_capability using errcode='22023';
  end if;

  insert into public.brian_external_capability_heartbeats(
    capability_id,collector_id,observed_at,status,source_project,metadata,updated_at
  )
  values(
    v_cap,v_collector,now(),'SUCCESS','brian-realtime',
    jsonb_build_object('verified_by','realtime-local-evidence','direct_alpha_influence',false),
    now()
  )
  on conflict(capability_id) do update
  set collector_id=excluded.collector_id,
      observed_at=excluded.observed_at,
      status=excluded.status,
      source_project=excluded.source_project,
      metadata=excluded.metadata,
      updated_at=excluded.updated_at;

  return jsonb_build_object('status','SUCCESS','capability_id',v_cap,'collector_id',v_collector,'observed_at',now());
end;
$function$
;

CREATE OR REPLACE FUNCTION brian_private.refresh_frontier_heartbeat_cache()
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public', 'brian_private', 'cron'
AS $function$
declare
  v_enabled boolean := false;
  v_total integer := 0;
  v_active integer := 0;
  v_dispatcher boolean := false;
  v_external_live boolean := false;
  v_treasury jsonb;
  v_alpha jsonb;
  v_behavior jsonb;
  v_behavior_pipeline jsonb;
  v_world jsonb;
  v_collectors jsonb;
  v_news jsonb;
  v_payload jsonb;
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
  v_external_jobs text[] := array[
    'brian-breaking-scout-2m',
    'brian-core-alpha-sync-3m',
    'brian-core-discovery-sync-10m',
    'brian-core-treasury-sync-4m',
    'brian-core-world-sync-6m',
    'brian-meeting-backend-sync-1m',
    'brian-multiasset-opportunity-10m'
  ]::text[];
begin
  if not pg_try_advisory_xact_lock(hashtextextended('brian-frontier-heartbeat-cache-v3',0)) then
    return jsonb_build_object('status','BUSY');
  end if;

  perform set_config('statement_timeout','4000',true);
  perform set_config('lock_timeout','1000',true);

  select coalesce(lower(config_value)='true',false)
  into v_enabled
  from public.brian_evolution_runtime_config
  where config_key='brian_system_enabled';

  select coalesce(active,false)
  into v_dispatcher
  from cron.job
  where jobname='brian-core-recovery-dispatcher-v1'
  limit 1;

  select exists(
    select 1
    from public.brian_scheduler_state
    where scheduler_id='brian-realtime-core-scheduler'
      and status='ONLINE'
      and last_seen_at > now()-interval '3 minutes'
  ) into v_external_live;

  select count(*)::int,
         count(*) filter (
           where case
             when r.job_name = any(v_core_jobs)
               then (v_dispatcher or v_external_live) and v_enabled
             when r.job_name = any(v_external_jobs)
               then v_external_live and v_enabled
             else coalesce(j.active,false)
           end
         )::int
  into v_total,v_active
  from public.brian_system_job_registry r
  left join cron.job j on j.jobname=r.job_name
  where r.operator_managed;

  select to_jsonb(t)
  into v_treasury
  from (
    select snapshot_id,observed_at,starting_equity_usd,cash_usd,equity_usd,deployment_pct,
           jsonb_array_length(coalesce(positions,'[]'::jsonb)) as open_positions
    from public.brian_treasury_shadow_snapshots
    order by observed_at desc,created_at desc
    limit 1
  ) t;

  select to_jsonb(a)
  into v_alpha
  from (
    select observed_at,asset_id,action,direction,evidence_score,estimated_round_trip_cost_bps
    from public.brian_alpha_decisions
    order by observed_at desc
    limit 1
  ) a;

  select coalesce(a.metadata->'crowd_behavior_context','{}'::jsonb)
         || jsonb_build_object(
              'asset_id',a.asset_id,
              'decision_observed_at',a.observed_at
            )
  into v_behavior
  from public.brian_alpha_decisions a
  where jsonb_typeof(a.metadata->'crowd_behavior_context')='object'
    and coalesce(a.metadata->'crowd_behavior_context'->>'state','') <> 'UNAVAILABLE'
  order by a.observed_at desc
  limit 1;

  select jsonb_build_object(
           'pipeline_status',h.status,
           'pipeline_observed_at',h.observed_at,
           'pipeline_collector_id',h.collector_id,
           'pipeline_source_project',h.source_project
         )
  into v_behavior_pipeline
  from public.brian_external_capability_heartbeats h
  where h.capability_id='market.crowd-behavior'
  limit 1;

  v_behavior := coalesce(v_behavior,'{}'::jsonb) || coalesce(v_behavior_pipeline,'{}'::jsonb);

  select to_jsonb(w)
  into v_world
  from (
    select status,started_at,finished_at,input_events,event_frames,entity_observations,narrative_snapshots,asset_impacts
    from public.brian_world_brain_runs
    where upper(coalesce(status,'')) not like 'SKIPPED%'
    order by started_at desc
    limit 1
  ) w;

  with tracked(collector_id) as (
    values
      ('brian-world-brain-v1'),
      ('brian-world-discovery-eye-v1'),
      ('brian-alpha-decision-compiler-v2'),
      ('brian-evolution-treasury-v1'),
      ('brian-evolution-orchestrator-v1'),
      ('brian-evolution-researcher-v1'),
      ('brian-evolution-sandbox-v1'),
      ('brian-evolution-ocean-worker-v1')
  ), latest as (
    select
      t.collector_id,
      r.status,r.started_at,r.finished_at,r.error_class,
      r.observed_records,r.stored_records,
      s.last_success_at
    from tracked t
    left join lateral (
      select cr.status,cr.started_at,cr.finished_at,cr.error_class,cr.observed_records,cr.stored_records
      from public.brian_collector_runs cr
      where cr.collector_id=t.collector_id
        and upper(coalesce(cr.status,'')) <> 'SKIPPED'
      order by cr.started_at desc
      limit 1
    ) r on true
    left join lateral (
      select max(cr.finished_at) as last_success_at
      from public.brian_collector_runs cr
      where cr.collector_id=t.collector_id
        and upper(coalesce(cr.status,'')) in ('SUCCESS','DEGRADED','ONLINE','OK')
    ) s on true
  )
  select coalesce(
    jsonb_object_agg(
      collector_id,
      (to_jsonb(latest)-'collector_id') ||
      jsonb_build_object(
        'activity_state',
        case
          when status='SUCCESS' and coalesce(observed_records,0)=0 and coalesce(stored_records,0)=0 then 'IDLE'
          when status='SUCCESS' and coalesce(observed_records,0)>0 and coalesce(stored_records,0)=0 then 'SCANNING_NO_NEW_DATA'
          else coalesce(status,'UNKNOWN')
        end
      )
    ),
    '{}'::jsonb
  )
  into v_collectors
  from latest;

  select brian_private.news_pipeline_health_v1() into v_news;

  v_payload := jsonb_build_object(
    'status','OK',
    'observed_at',now(),
    'control',jsonb_build_object(
      'system_enabled',v_enabled,
      'managed_jobs_total',v_total,
      'managed_jobs_active',v_active,
      'core_dispatcher_active',v_dispatcher,
      'realtime_scheduler_active',v_external_live,
      'treasury',v_treasury
    ),
    'alpha',v_alpha,
    'behavior',v_behavior,
    'world_run',v_world,
    'collectors',v_collectors,
    'news_pipeline',v_news,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false,
    'read_only',true,
    'source','heartbeat_cache'
  );

  insert into brian_private.frontier_heartbeat_cache(singleton,payload,updated_at)
  values(true,v_payload,now())
  on conflict(singleton) do update
  set payload=excluded.payload,updated_at=excluded.updated_at;

  return v_payload;
exception
  when query_canceled then
    return jsonb_build_object('status','DEGRADED','reason','HEARTBEAT_REFRESH_TIMEOUT','observed_at',now());
  when lock_not_available then
    return jsonb_build_object('status','DEGRADED','reason','HEARTBEAT_REFRESH_LOCK_TIMEOUT','observed_at',now());
end;
$function$
;
