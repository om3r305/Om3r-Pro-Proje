-- Brian DIP V8: make START/PAUSE/RESTART ordering deterministic.
-- PostgreSQL now() is transaction-stable, so a restart used to write PAUSE and START
-- with the exact same requested_at timestamp. Readers then fell back to event_id order
-- and could select the PAUSE row (config=NULL) as the latest event.

create or replace function public.brian_dip_restart_session(
  p_pause_event_id text,
  p_start_event_id text,
  p_new_session_id text,
  p_starting_equity numeric,
  p_trade_notional numeric,
  p_config jsonb,
  p_engine_token_sha256 text
)
returns public.brian_dip_session_events
language plpgsql
security definer
set search_path to 'pg_catalog','public'
as $function$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
  event_ts timestamptz;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));

  select * into latest
  from public.brian_dip_session_events
  order by requested_at desc,
           case when event_kind='START' then 1 else 0 end desc,
           event_id desc
  limit 1;

  event_ts := clock_timestamp();

  if latest.event_kind='START' then
    insert into public.brian_dip_session_events(
      event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,
      config,engine_token_sha256,shadow_only,live_execution
    ) values(
      p_pause_event_id,latest.session_id,'PAUSE',event_ts,null,null,null,null,true,false
    );
  end if;

  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,
    config,engine_token_sha256,shadow_only,live_execution
  ) values(
    p_start_event_id,p_new_session_id,'START',event_ts + interval '1 microsecond',
    p_starting_equity,p_trade_notional,p_config,p_engine_token_sha256,true,false
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
set search_path to 'pg_catalog','public'
as $function$
declare
  latest public.brian_dip_session_events%rowtype;
  original public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));

  select * into latest
  from public.brian_dip_session_events
  order by requested_at desc,
           case when event_kind='START' then 1 else 0 end desc,
           event_id desc
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
      event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,
      config,engine_token_sha256,shadow_only,live_execution
    ) values(
      p_event_id,original.session_id,'START',clock_timestamp(),original.starting_equity,
      original.trade_notional,original.config,original.engine_token_sha256,true,false
    ) returning * into inserted;

    return inserted;
  end if;

  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,
    config,engine_token_sha256,shadow_only,live_execution
  ) values(
    p_event_id,p_session_id,'START',clock_timestamp(),p_starting_equity,
    p_trade_notional,p_config,p_engine_token_sha256,true,false
  ) returning * into inserted;

  return inserted;
end;
$function$;

create or replace function public.brian_dip_pause_session(
  p_event_id text,
  p_session_id text
)
returns public.brian_dip_session_events
language plpgsql
security definer
set search_path to 'pg_catalog','public'
as $function$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));

  select * into latest
  from public.brian_dip_session_events
  order by requested_at desc,
           case when event_kind='START' then 1 else 0 end desc,
           event_id desc
  limit 1;

  if latest.event_kind is distinct from 'START'
     or latest.session_id is distinct from p_session_id then
    raise exception 'BRIAN_DIP: no matching active dip session';
  end if;

  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,
    config,engine_token_sha256,shadow_only,live_execution
  ) values(
    p_event_id,p_session_id,'PAUSE',clock_timestamp(),null,null,null,null,true,false
  ) returning * into inserted;

  return inserted;
end;
$function$;
