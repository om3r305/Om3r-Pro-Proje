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
    when 'news.direct-wire' then 'brian-direct-wire-eye-v1'
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
  v_scout_observed integer := 0;
  v_scout_stored integer := 0;
  v_ingest_at timestamptz;
  v_ingest_status text;
  v_ingest_observed integer := 0;
  v_ingest_stored integer := 0;
  v_wire_at timestamptz;
  v_wire_status text;
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

  select finished_at,status,observed_records,stored_records
    into v_scout_at,v_scout_status,v_scout_observed,v_scout_stored
  from public.brian_collector_runs
  where collector_id='brian-breaking-scout-v1'
  order by started_at desc
  limit 1;

  select finished_at,status,observed_records,stored_records
    into v_ingest_at,v_ingest_status,v_ingest_observed,v_ingest_stored
  from public.brian_collector_runs
  where collector_id='brian-core-intel-ingest-v1'
  order by started_at desc
  limit 1;

  select observed_at,status into v_wire_at,v_wire_status
  from public.brian_external_capability_heartbeats
  where capability_id='news.direct-wire'
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
    when coalesce(v_scout_status,'UNKNOWN') <> 'SUCCESS' then 'DEGRADED'
    when v_wire_at is null or v_now-v_wire_at > interval '6 minutes' or coalesce(v_wire_status,'UNKNOWN') <> 'SUCCESS' then 'DEGRADED'
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
      'activity_state',case
        when v_scout_status='SUCCESS' and v_scout_observed=0 and v_scout_stored=0 then 'IDLE'
        when v_scout_status='SUCCESS' and v_scout_observed>0 and v_scout_stored=0 then 'SCANNING_NO_NEW_DATA'
        else coalesce(v_scout_status,'UNKNOWN')
      end,
      'observed_records',v_scout_observed,
      'stored_records',v_scout_stored,
      'last_run_at',v_scout_at
    ),
    'direct_wire',jsonb_build_object(
      'status',coalesce(v_wire_status,'UNKNOWN'),
      'activity_state',case
        when v_wire_at is null then 'NO_HEARTBEAT'
        when v_now-v_wire_at > interval '6 minutes' then 'STALE'
        else 'LIVE'
      end,
      'last_run_at',v_wire_at
    ),
    'realtime_official_ingest',jsonb_build_object(
      'status',v_ingest_status,
      'activity_state',case
        when v_ingest_status='SUCCESS' and v_ingest_observed=0 and v_ingest_stored=0 then 'IDLE'
        when v_ingest_status='SUCCESS' and v_ingest_observed>0 and v_ingest_stored=0 then 'SCANNING_NO_NEW_DATA'
        else coalesce(v_ingest_status,'UNKNOWN')
      end,
      'observed_records',v_ingest_observed,
      'stored_records',v_ingest_stored,
      'last_run_at',v_ingest_at
    ),
    'stall_threshold_seconds',720,
    'pipeline_stalled',(v_backlog > 0 and v_backlog_age_seconds > 720),
    'read_only',true,
    'shadow_only',true,
    'live_execution',false
  );
end;
$function$
;
