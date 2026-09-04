-- Brian Aggressive Dip Trader — isolated SHADOW-only paper scalper.
-- This module is intentionally separate from Phase 3.7, Control Center policy,
-- learning, collectors, and every live-execution surface.

create table if not exists public.brian_dip_session_events (
  event_id text primary key,
  session_id text not null,
  event_kind text not null check (event_kind in ('START','PAUSE')),
  requested_at timestamptz not null default now(),
  starting_equity numeric,
  trade_notional numeric,
  config jsonb,
  engine_token_sha256 text,
  requested_by text not null default 'monster-coins-pro-dip-pwa',
  evidence_class text not null default 'AGGRESSIVE_DIP_SHADOW' check (evidence_class='AGGRESSIVE_DIP_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_dip_session_event_shape check (
    (event_kind='START'
      and starting_equity > 0 and starting_equity <= 1000000
      and trade_notional > 0 and trade_notional <= starting_equity
      and jsonb_typeof(config)='object'
      and engine_token_sha256 ~ '^[0-9a-f]{64}$')
    or
    (event_kind='PAUSE'
      and starting_equity is null
      and trade_notional is null
      and config is null
      and engine_token_sha256 is null)
  )
);
create unique index if not exists brian_dip_session_start_unique
  on public.brian_dip_session_events(session_id) where event_kind='START';
create index if not exists brian_dip_session_events_time_idx
  on public.brian_dip_session_events(requested_at desc,event_id desc);
create index if not exists brian_dip_session_events_session_idx
  on public.brian_dip_session_events(session_id,requested_at asc);

create table if not exists public.brian_dip_events (
  event_id text primary key,
  session_id text not null,
  observed_at timestamptz not null,
  event_kind text not null check (event_kind in (
    'ENGINE_START','ENGINE_PAUSE','DIP_ARMED','DIP_NEW_LOW','BUY','SELL','SKIP_CHASE','NO_CASH','INFO'
  )),
  symbol text,
  price numeric,
  dip_low numeric,
  entry_price numeric,
  exit_price numeric,
  quantity numeric,
  notional numeric,
  fees numeric,
  realized_pnl numeric,
  cash_after numeric,
  equity_after numeric,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'AGGRESSIVE_DIP_SHADOW' check (evidence_class='AGGRESSIVE_DIP_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);
create index if not exists brian_dip_events_session_time_idx
  on public.brian_dip_events(session_id,observed_at desc,event_id desc);
create index if not exists brian_dip_events_symbol_time_idx
  on public.brian_dip_events(symbol,observed_at desc) where symbol is not null;

create table if not exists public.brian_dip_snapshots (
  snapshot_id text primary key,
  session_id text not null,
  observed_at timestamptz not null,
  cash numeric not null,
  equity numeric not null,
  realized_pnl numeric not null default 0,
  unrealized_pnl numeric not null default 0,
  trade_count integer not null default 0 check (trade_count >= 0),
  win_count integer not null default 0 check (win_count >= 0),
  loss_count integer not null default 0 check (loss_count >= 0),
  state jsonb not null,
  evidence_class text not null default 'AGGRESSIVE_DIP_SHADOW' check (evidence_class='AGGRESSIVE_DIP_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_dip_snapshot_finiteish check (cash > -1000000000 and equity > -1000000000)
);
create index if not exists brian_dip_snapshots_session_time_idx
  on public.brian_dip_snapshots(session_id,observed_at desc,snapshot_id desc);

alter table public.brian_dip_session_events enable row level security;
alter table public.brian_dip_events enable row level security;
alter table public.brian_dip_snapshots enable row level security;

revoke all on public.brian_dip_session_events from anon,authenticated;
revoke all on public.brian_dip_events from anon,authenticated;
revoke all on public.brian_dip_snapshots from anon,authenticated;
revoke update,delete,truncate,references,trigger on public.brian_dip_session_events from service_role;
revoke update,delete,truncate,references,trigger on public.brian_dip_events from service_role;
revoke update,delete,truncate,references,trigger on public.brian_dip_snapshots from service_role;
grant select,insert on public.brian_dip_session_events to service_role;
grant select,insert on public.brian_dip_events to service_role;
grant select,insert on public.brian_dip_snapshots to service_role;

drop trigger if exists brian_dip_session_events_append_only on public.brian_dip_session_events;
create trigger brian_dip_session_events_append_only
  before update or delete on public.brian_dip_session_events
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_dip_events_append_only on public.brian_dip_events;
create trigger brian_dip_events_append_only
  before update or delete on public.brian_dip_events
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_dip_snapshots_append_only on public.brian_dip_snapshots;
create trigger brian_dip_snapshots_append_only
  before update or delete on public.brian_dip_snapshots
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_dip_start_session(
  p_event_id text,
  p_session_id text,
  p_starting_equity numeric,
  p_trade_notional numeric,
  p_config jsonb,
  p_engine_token_sha256 text
) returns public.brian_dip_session_events
language plpgsql security definer set search_path=pg_catalog,public as $$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into latest from public.brian_dip_session_events
    order by requested_at desc,event_id desc limit 1;
  if latest.event_kind='START' then
    raise exception 'BRIAN_DIP: an active dip session already exists';
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

create or replace function public.brian_dip_pause_session(
  p_event_id text,
  p_session_id text
) returns public.brian_dip_session_events
language plpgsql security definer set search_path=pg_catalog,public as $$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into latest from public.brian_dip_session_events
    order by requested_at desc,event_id desc limit 1;
  if latest.event_kind is distinct from 'START' or latest.session_id is distinct from p_session_id then
    raise exception 'BRIAN_DIP: no matching active dip session';
  end if;
  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,
    shadow_only,live_execution
  ) values (
    p_event_id,p_session_id,'PAUSE',null,null,null,null,true,false
  ) returning * into inserted;
  return inserted;
end;
$$;

create or replace function public.brian_dip_restart_session(
  p_pause_event_id text,
  p_start_event_id text,
  p_new_session_id text,
  p_starting_equity numeric,
  p_trade_notional numeric,
  p_config jsonb,
  p_engine_token_sha256 text
) returns public.brian_dip_session_events
language plpgsql security definer set search_path=pg_catalog,public as $$
declare
  latest public.brian_dip_session_events%rowtype;
  inserted public.brian_dip_session_events%rowtype;
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into latest from public.brian_dip_session_events
    order by requested_at desc,event_id desc limit 1;
  if latest.event_kind='START' then
    insert into public.brian_dip_session_events(
      event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,
      shadow_only,live_execution
    ) values (
      p_pause_event_id,latest.session_id,'PAUSE',null,null,null,null,true,false
    );
  end if;
  insert into public.brian_dip_session_events(
    event_id,session_id,event_kind,starting_equity,trade_notional,config,engine_token_sha256,
    shadow_only,live_execution
  ) values (
    p_start_event_id,p_new_session_id,'START',p_starting_equity,p_trade_notional,p_config,p_engine_token_sha256,
    true,false
  ) returning * into inserted;
  return inserted;
end;
$$;

revoke all on function public.brian_dip_start_session(text,text,numeric,numeric,jsonb,text) from public,anon,authenticated;
revoke all on function public.brian_dip_pause_session(text,text) from public,anon,authenticated;
revoke all on function public.brian_dip_restart_session(text,text,text,numeric,numeric,jsonb,text) from public,anon,authenticated;
grant execute on function public.brian_dip_start_session(text,text,numeric,numeric,jsonb,text) to service_role;
grant execute on function public.brian_dip_pause_session(text,text) to service_role;
grant execute on function public.brian_dip_restart_session(text,text,text,numeric,numeric,jsonb,text) to service_role;
