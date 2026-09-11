-- Brian Evolution OS Layer 6 hardening: serialize Ocean START/STOP commands.
-- GitHub-only until explicit rollout. MAIN/ALPHA only. DIP is outside scope.
--
-- The dashboard preflight is advisory evidence validation. The database is the final
-- concurrency boundary: two simultaneous START requests must never create two active
-- Ocean exams, and STOP must target an actually active run.

create or replace function public.brian_ocean_start_run(
  p_command_id text,
  p_run_id text,
  p_requested_at timestamptz,
  p_duration_hours integer,
  p_reason text default null,
  p_requested_by text default 'dashboard',
  p_metadata jsonb default '{}'::jsonb
)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_active_run_id text;
begin
  if nullif(trim(p_command_id),'') is null or nullif(trim(p_run_id),'') is null then
    raise exception 'OCEAN_CONTROL: command_id and run_id are required';
  end if;
  if p_requested_at is null then
    raise exception 'OCEAN_CONTROL: requested_at is required';
  end if;
  if p_duration_hours not in (24,48) then
    raise exception 'OCEAN_CONTROL: duration_hours must be 24 or 48';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('brian_evolution_ocean_control_v1'));

  select s.run_id into v_active_run_id
  from public.brian_ocean_run_commands s
  where s.command='START'
    and s.requested_at <= p_requested_at
    and s.requested_at + make_interval(hours => s.duration_hours) > p_requested_at
    and not exists (
      select 1
      from public.brian_ocean_run_commands t
      where t.run_id=s.run_id
        and t.command='STOP'
        and t.requested_at >= s.requested_at
        and t.requested_at <= p_requested_at
    )
  order by s.requested_at desc
  limit 1;

  if v_active_run_id is not null then
    raise exception 'OCEAN_RUN_ALREADY_ACTIVE:%', v_active_run_id;
  end if;

  insert into public.brian_ocean_run_commands(
    command_id,run_id,command,requested_at,duration_hours,reason,requested_by,metadata,
    evidence_class,shadow_only,live_execution
  ) values (
    p_command_id,p_run_id,'START',p_requested_at,p_duration_hours,
    nullif(left(coalesce(p_reason,''),500),''),coalesce(nullif(trim(p_requested_by),''),'dashboard'),coalesce(p_metadata,'{}'::jsonb),
    'PROSPECTIVE_EVOLUTION_SHADOW',true,false
  );

  return p_run_id;
end;
$$;

create or replace function public.brian_ocean_stop_run(
  p_command_id text,
  p_run_id text,
  p_requested_at timestamptz,
  p_reason text default 'operator stop',
  p_requested_by text default 'dashboard',
  p_metadata jsonb default '{}'::jsonb
)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_start_at timestamptz;
  v_duration_hours integer;
begin
  if nullif(trim(p_command_id),'') is null or nullif(trim(p_run_id),'') is null then
    raise exception 'OCEAN_CONTROL: command_id and run_id are required';
  end if;
  if p_requested_at is null then
    raise exception 'OCEAN_CONTROL: requested_at is required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('brian_evolution_ocean_control_v1'));

  select requested_at,duration_hours
    into v_start_at,v_duration_hours
  from public.brian_ocean_run_commands
  where run_id=p_run_id and command='START'
  limit 1;

  if v_start_at is null
     or p_requested_at < v_start_at
     or p_requested_at >= v_start_at + make_interval(hours => v_duration_hours)
     or exists(
       select 1 from public.brian_ocean_run_commands
       where run_id=p_run_id and command='STOP' and requested_at <= p_requested_at
     ) then
    raise exception 'OCEAN_RUN_NOT_ACTIVE:%', p_run_id;
  end if;

  insert into public.brian_ocean_run_commands(
    command_id,run_id,command,requested_at,duration_hours,reason,requested_by,metadata,
    evidence_class,shadow_only,live_execution
  ) values (
    p_command_id,p_run_id,'STOP',p_requested_at,null,
    coalesce(nullif(left(coalesce(p_reason,''),500),''),'operator stop'),coalesce(nullif(trim(p_requested_by),''),'dashboard'),coalesce(p_metadata,'{}'::jsonb),
    'PROSPECTIVE_EVOLUTION_SHADOW',true,false
  );

  return p_run_id;
end;
$$;

-- Force all service-role command writes through the serialized control functions.
revoke insert on public.brian_ocean_run_commands from service_role;
revoke all on function public.brian_ocean_start_run(text,text,timestamptz,integer,text,text,jsonb) from public,anon,authenticated;
revoke all on function public.brian_ocean_stop_run(text,text,timestamptz,text,text,jsonb) from public,anon,authenticated;
grant execute on function public.brian_ocean_start_run(text,text,timestamptz,integer,text,text,jsonb) to service_role;
grant execute on function public.brian_ocean_stop_run(text,text,timestamptz,text,text,jsonb) to service_role;
