CREATE OR REPLACE FUNCTION brian_private.news_pipeline_health_v1()
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public', 'brian_private'
AS $function$
declare
  v_now timestamptz := now();
  v_intel_at timestamptz;
  v_frame_event_at timestamptz;
  v_frame_written_at timestamptz;
  v_world_at timestamptz;
  v_scout_at timestamptz;
  v_scout_status text;
  v_ingest_at timestamptz;
  v_ingest_status text;
  v_backlog integer := 0;
  v_oldest_backlog_at timestamptz;
  v_backlog_age_seconds integer := 0;
  v_status text := 'UNKNOWN';
begin
  select first_observed_at into v_intel_at
  from public.brian_intel_events
  order by first_observed_at desc
  limit 1;

  select observed_at into v_frame_event_at
  from public.brian_world_event_frames
  order by observed_at desc
  limit 1;

  select created_at into v_frame_written_at
  from public.brian_world_event_frames
  order by created_at desc
  limit 1;

  select finished_at into v_world_at
  from public.brian_world_brain_runs
  where status='SUCCESS'
  order by started_at desc
  limit 1;

  select finished_at,status into v_scout_at,v_scout_status
  from public.brian_collector_runs
  where collector_id='brian-breaking-scout-v1'
  order by started_at desc
  limit 1;

  select finished_at,status into v_ingest_at,v_ingest_status
  from public.brian_collector_runs
  where collector_id='brian-core-intel-ingest-v1'
  order by started_at desc
  limit 1;

  if v_frame_event_at is not null then
    select count(*)::int,min(first_observed_at)
      into v_backlog,v_oldest_backlog_at
    from public.brian_intel_events
    where first_observed_at > v_frame_event_at;
  elsif v_intel_at is not null then
    v_backlog := 1;
    v_oldest_backlog_at := v_intel_at;
  end if;

  if v_oldest_backlog_at is not null then
    v_backlog_age_seconds := greatest(0,extract(epoch from (v_now-v_oldest_backlog_at))::int);
  end if;

  v_status := case
    when v_world_at is null then 'DEGRADED'
    when v_now-v_world_at > interval '12 minutes' then 'DEGRADED'
    when v_backlog > 0 and v_backlog_age_seconds > 720 then 'DEGRADED'
    else 'HEALTHY'
  end;

  return jsonb_build_object(
    'status',v_status,
    'observed_at',v_now,
    'latest_intel_at',v_intel_at,
    'latest_world_event_at',v_frame_event_at,
    'latest_world_write_at',v_frame_written_at,
    'latest_world_success_at',v_world_at,
    'backlog_events',v_backlog,
    'oldest_backlog_at',v_oldest_backlog_at,
    'backlog_age_seconds',v_backlog_age_seconds,
    'breaking_scout',jsonb_build_object(
      'status',v_scout_status,
      'last_run_at',v_scout_at
    ),
    'realtime_official_ingest',jsonb_build_object(
      'status',v_ingest_status,
      'last_run_at',v_ingest_at
    ),
    'stall_threshold_seconds',720,
    'pipeline_stalled',(v_backlog > 0 and v_backlog_age_seconds > 720),
    'read_only',true,
    'shadow_only',true,
    'live_execution',false
  );
end;
$function$;

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
      s.last_success_at
    from tracked t
    left join lateral (
      select cr.status,cr.started_at,cr.finished_at,cr.error_class
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
  select coalesce(jsonb_object_agg(collector_id,to_jsonb(latest)-'collector_id'),'{}'::jsonb)
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
$function$;