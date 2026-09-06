-- Brian 2026 session lifecycle contract
-- START after PAUSE resumes the same append-only session.
-- RESTART remains the only operation that opens a fresh session window.
-- No historical rows are deleted or rewritten. SHADOW ONLY.

create or replace function public.brian_dashboard_start_session(
  p_event_id text,
  p_session_id text,
  p_starting_equity numeric,
  p_policy_scope text,
  p_source_experiment_id text
)
returns public.brian_dashboard_session_events
language plpgsql
security definer
set search_path to 'pg_catalog', 'public'
as $function$
declare
  latest public.brian_dashboard_session_events%rowtype;
  original public.brian_dashboard_session_events%rowtype;
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
    select * into original
    from public.brian_dashboard_session_events
    where session_id=latest.session_id and event_kind='START'
    order by requested_at asc,event_id asc
    limit 1;

    if original.session_id is null then
      raise exception 'BRIAN_DASHBOARD: paused session start missing';
    end if;

    insert into public.brian_dashboard_session_events(
      event_id,session_id,event_kind,starting_equity,policy_scope,source_experiment_id,shadow_only,live_execution
    ) values (
      p_event_id,original.session_id,'START',original.starting_equity,original.policy_scope,original.source_experiment_id,true,false
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
$function$;

create or replace function public.brian_dip_start_session(
  p_event_id text,
  p_session_id text,
  p_starting_equity numeric,
  p_trade_notional numeric,
  p_config jsonb,
  p_engine_token_sha256 text
)
returns public.brian_dip_session_events
language plpgsql
security definer
set search_path to 'pg_catalog', 'public'
as $function$
declare
  latest public.brian_dip_session_events%rowtype;
  original public.brian_dip_session_events%rowtype;
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
    select * into original
    from public.brian_dip_session_events
    where session_id=latest.session_id and event_kind='START'
    order by requested_at asc,event_id asc
    limit 1;

    if original.session_id is null then
      raise exception 'BRIAN_DIP: paused session start missing';
    end if;

    insert into public.brian_dip_session_events(
      event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,shadow_only,live_execution
    ) values (
      p_event_id,original.session_id,'START',original.starting_equity,original.trade_notional,original.config,original.engine_token_sha256,true,false
    ) returning * into inserted;
    return inserted;
  end if;

  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,shadow_only,live_execution
  ) values (
    p_event_id,p_session_id,'START',p_starting_equity,p_trade_notional,p_config,p_engine_token_sha256,true,false
  ) returning * into inserted;
  return inserted;
end;
$function$;
