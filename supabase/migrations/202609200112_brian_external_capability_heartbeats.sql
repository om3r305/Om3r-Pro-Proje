create table if not exists public.brian_external_capability_heartbeats (
  capability_id text primary key,
  collector_id text not null,
  observed_at timestamptz not null,
  status text not null,
  source_project text not null,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create index if not exists brian_external_capability_heartbeats_observed_idx
  on public.brian_external_capability_heartbeats(observed_at desc);

create or replace function public.brian_external_capability_heartbeat_v1(p_capability text)
returns jsonb
language plpgsql
security definer
set search_path='pg_catalog','public'
as $$
declare
  v_cap text := lower(trim(coalesce(p_capability,'')));
  v_collector text;
begin
  v_collector := case v_cap
    when 'market.universe' then 'brian-realtime-universe-heartbeat'
    when 'market.sensor-mesh' then 'brian-realtime-sensor-heartbeat'
    when 'market.intrabar' then 'brian-realtime-intrabar-heartbeat'
    when 'market.derivatives' then 'brian-realtime-derivatives-heartbeat'
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

  return jsonb_build_object(
    'status','SUCCESS',
    'capability_id',v_cap,
    'collector_id',v_collector,
    'observed_at',now()
  );
end;
$$;

revoke all on function public.brian_external_capability_heartbeat_v1(text) from public,anon,authenticated;
grant execute on function public.brian_external_capability_heartbeat_v1(text) to service_role;
