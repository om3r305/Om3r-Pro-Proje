-- Brian session-window resume semantics.
-- Historical evidence stays append-only. START/New Session creates a fresh display window;
-- PAUSE -> Start resumes the same logical session. SHADOW ONLY; no execution changes.

-- MAIN / Control Center: when the latest session is paused, the existing Start action resumes
-- the same session_id and original tracking parameters. Explicit restart remains the only path
-- that creates a new tracking session after one already exists.
create or replace function public.brian_dashboard_start_session(
  p_event_id text,
  p_session_id text,
  p_starting_equity numeric,
  p_policy_scope text,
  p_source_experiment_id text
) returns public.brian_dashboard_session_events
language plpgsql
security definer
set search_path=pg_catalog,public
as $$
declare
  latest public.brian_dashboard_session_events%rowtype;
  origin public.brian_dashboard_session_events%rowtype;
  inserted public.brian_dashboard_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-dashboard-control',0));
  select * into latest
    from public.brian_dashboard_session_events
    order by requested_at desc,event_id desc
    limit 1;

  if latest.event_kind='START' then
    raise exception 'BRIAN_DASHBOARD: an active shadow session already exists';
  end if;

  if latest.event_kind='PAUSE' then
    select * into origin
      from public.brian_dashboard_session_events
      where session_id=latest.session_id and event_kind='START'
      order by requested_at asc,event_id asc
      limit 1;
    if origin.session_id is null then
      raise exception 'BRIAN_DASHBOARD: paused session origin missing';
    end if;
    insert into public.brian_dashboard_session_events(
      event_id,session_id,event_kind,starting_equity,policy_scope,source_experiment_id,
      requested_by,note,shadow_only,live_execution
    ) values (
      p_event_id,origin.session_id,'START',origin.starting_equity,origin.policy_scope,origin.source_experiment_id,
      'monster-coins-pro-pwa','resume existing tracking session',true,false
    ) returning * into inserted;
    return inserted;
  end if;

  insert into public.brian_dashboard_session_events(
    event_id,session_id,event_kind,starting_equity,policy_scope,source_experiment_id,shadow_only,live_execution
  ) values (
    p_event_id,p_session_id,'START',p_starting_equity,p_policy_scope,p_source_experiment_id,true,false
  ) returning * into inserted;
  return inserted;
end;
$$;

revoke all on function public.brian_dashboard_start_session(text,text,numeric,text,text) from public,anon,authenticated;
grant execute on function public.brian_dashboard_start_session(text,text,numeric,text,text) to service_role;

-- DIP: allow multiple START markers for the same session so a PAUSE can be resumed without
-- inventing a new session_id. The earliest START remains the immutable session origin.
drop index if exists public.brian_dip_session_start_unique;
create index if not exists brian_dip_session_start_idx
  on public.brian_dip_session_events(session_id,requested_at asc,event_id asc)
  where event_kind='START';

create or replace function public.brian_dip_resume_session(
  p_event_id text,
  p_session_id text,
  p_engine_token_sha256 text
) returns public.brian_dip_session_events
language plpgsql
security definer
set search_path=pg_catalog,public
as $$
declare
  latest public.brian_dip_session_events%rowtype;
  origin public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into latest
    from public.brian_dip_session_events
    order by requested_at desc,event_id desc
    limit 1;

  if latest.event_kind is distinct from 'PAUSE' or latest.session_id is distinct from p_session_id then
    raise exception 'BRIAN_DIP: no matching paused dip session';
  end if;

  select * into origin
    from public.brian_dip_session_events
    where session_id=p_session_id and event_kind='START'
    order by requested_at asc,event_id asc
    limit 1;
  if origin.session_id is null then
    raise exception 'BRIAN_DIP: paused session origin missing';
  end if;
  if origin.engine_token_sha256 is distinct from p_engine_token_sha256 then
    raise exception 'BRIAN_DIP: resume engine token mismatch';
  end if;

  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,
    requested_by,evidence_class,shadow_only,live_execution
  ) values (
    p_event_id,origin.session_id,'START',origin.starting_equity,origin.trade_notional,origin.config,origin.engine_token_sha256,
    'monster-coins-pro-dip-pwa','AGGRESSIVE_DIP_SHADOW',true,false
  ) returning * into inserted;

  -- Lease is operational state, not evidence. Resume the same owner/token and bump generation.
  insert into public.brian_dip_engine_leases(
    session_id,engine_token_sha256,lease_generation,claimed_at,heartbeat_at
  ) values (
    origin.session_id,origin.engine_token_sha256,1,now(),now()
  )
  on conflict (session_id) do update
    set engine_token_sha256=excluded.engine_token_sha256,
        lease_generation=public.brian_dip_engine_leases.lease_generation+1,
        claimed_at=excluded.claimed_at,
        heartbeat_at=excluded.heartbeat_at;

  return inserted;
end;
$$;

revoke all on function public.brian_dip_resume_session(text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_resume_session(text,text,text) to service_role;

-- Old cached clients must not silently create a second DIP session after PAUSE. They must
-- refresh and use the resume endpoint, or choose explicit Restart/New Session.
create or replace function public.brian_dip_start_session(
  p_event_id text,
  p_session_id text,
  p_starting_equity numeric,
  p_trade_notional numeric,
  p_config jsonb,
  p_engine_token_sha256 text
) returns public.brian_dip_session_events
language plpgsql
security definer
set search_path=pg_catalog,public
as $$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into latest
    from public.brian_dip_session_events
    order by requested_at desc,event_id desc
    limit 1;
  if latest.event_kind='START' then
    raise exception 'BRIAN_DIP: an active dip session already exists';
  end if;
  if latest.event_kind='PAUSE' then
    raise exception 'BRIAN_DIP: paused session must resume or explicitly restart';
  end if;
  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,
    shadow_only,live_execution
  ) values (
    p_event_id,p_session_id,'START',p_starting_equity,p_trade_notional,p_config,p_engine_token_sha256,
    true,false
  ) returning * into inserted;
  return inserted;
end;
$$;

revoke all on function public.brian_dip_start_session(text,text,numeric,numeric,jsonb,text) from public,anon,authenticated;
grant execute on function public.brian_dip_start_session(text,text,numeric,numeric,jsonb,text) to service_role;
