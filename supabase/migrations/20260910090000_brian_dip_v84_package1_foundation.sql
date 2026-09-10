-- Brian DIP V8.4 Package 1 foundation.
-- ADDITIVE ONLY. Do not ALTER, DROP, rewrite, schedule or redeploy any V8.3 object.
-- SHADOW ONLY. This migration intentionally registers the first release as PREPARED,
-- not SEALED, so an accidental deployment cannot start V8.4 execution.

create table if not exists public.brian_dip_v84_releases (
  release_id text primary key,
  status text not null check (status in ('PREPARED','SEALED','RETIRED')),
  logic_hash text not null,
  strategy_manifest_hash text not null,
  calibration_family_id text not null,
  db_contract_version text not null,
  manifest jsonb not null,
  created_at timestamptz not null default clock_timestamp(),
  sealed_at timestamptz,
  check ((status='SEALED' and sealed_at is not null and logic_hash<>'UNSEALED_GITHUB_ONLY') or status<>'SEALED')
);
alter table public.brian_dip_v84_releases enable row level security;
revoke all on public.brian_dip_v84_releases from public,anon,authenticated;
grant select,insert,update on public.brian_dip_v84_releases to service_role;

insert into public.brian_dip_v84_releases(release_id,status,logic_hash,strategy_manifest_hash,calibration_family_id,db_contract_version,manifest)
values(
  'dip-v84-package1-evidence-20260910.1','PREPARED','UNSEALED_GITHUB_ONLY',
  '1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0',
  'dip-v84-package1-evidence-family-1','brian-dip-v84-db-1',
  '{"symbol":"ETHUSDT","shadow_only":true,"live_execution":false,"browser_execution":false,"server_authoritative":true,"max_shadow_leverage":1,"warm_risk_promotion_enabled":false,"target_policy":"NEAREST_STRUCTURAL_THEN_ECONOMIC_GATE","live_l2":false,"skip_obstacle":false}'::jsonb
) on conflict (release_id) do nothing;

create or replace function public.brian_dip_v84_reject_mutation() returns trigger
language plpgsql set search_path='pg_catalog','public' as $$
begin raise exception 'V84_APPEND_ONLY_EVIDENCE'; end $$;
revoke all on function public.brian_dip_v84_reject_mutation() from public,anon,authenticated;
grant execute on function public.brian_dip_v84_reject_mutation() to service_role;

create table if not exists public.brian_dip_v84_session_events (
  event_id bigint generated always as identity primary key,
  session_id text not null,
  event_kind text not null check (event_kind in ('START','PAUSE')),
  requested_at timestamptz not null default clock_timestamp(),
  starting_equity numeric,
  trade_notional numeric,
  config jsonb not null default '{}'::jsonb,
  check (event_kind<>'START' or (starting_equity>0 and trade_notional>0))
);
create index if not exists brian_dip_v84_session_latest_idx on public.brian_dip_v84_session_events(requested_at desc,event_id desc);
create index if not exists brian_dip_v84_session_start_idx on public.brian_dip_v84_session_events(session_id,requested_at,event_id) where event_kind='START';
alter table public.brian_dip_v84_session_events enable row level security;
revoke all on public.brian_dip_v84_session_events from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_session_events to service_role;
drop trigger if exists brian_dip_v84_session_append_only on public.brian_dip_v84_session_events;
create trigger brian_dip_v84_session_append_only before update or delete on public.brian_dip_v84_session_events for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_runtime (
  session_id text primary key,
  state_version bigint not null default 0 check(state_version>=0),
  runtime jsonb not null,
  snapshot jsonb,
  last_commit_id text,
  last_commit_hash text,
  updated_at timestamptz not null default clock_timestamp(),
  shadow_only boolean not null default true check(shadow_only),
  live_execution boolean not null default false check(not live_execution)
);
alter table public.brian_dip_v84_runtime enable row level security;
revoke all on public.brian_dip_v84_runtime from public,anon,authenticated;
grant select,insert,update on public.brian_dip_v84_runtime to service_role;

create table if not exists public.brian_dip_v84_decisions (
  occurrence_id text primary key,
  session_id text not null,
  episode_id text not null,
  parent_occurrence_id text,
  symbol text not null check(symbol='ETHUSDT'),
  decision_at timestamptz not null,
  signal_at timestamptz not null,
  due_at timestamptz not null,
  setup text not null check(setup in ('SWEEP_RECLAIM','FAILED_BREAK','BOS_RETEST','EARLY_REVERSAL')),
  direction text not null check(direction in ('UP','DOWN')),
  regime text not null,
  venue text not null check(venue='SHADOW_PERP'),
  entry_price numeric not null check(entry_price>0),
  target_price numeric,
  invalidation_price numeric,
  target_role text not null check(target_role in ('L1_EXECUTABLE','L1_BLOCKING','NO_FORWARD_LEVEL')),
  l1_id text,
  raw_conviction double precision,
  forecast_probability double precision,
  release_id text not null,
  logic_hash text not null,
  strategy_manifest_hash text not null,
  calibration_family_id text not null,
  db_contract_version text not null,
  policy_version text not null,
  decision_revision text not null,
  metric_version text not null,
  resolver_version text not null,
  entry_guard_version text not null,
  target_planner_version text not null,
  execution_model_version text not null,
  cost_model_version text not null,
  market_data_contract_version text not null,
  evidence jsonb not null,
  checked_until timestamptz,
  last_price numeric,
  resolved_at timestamptz,
  resolution jsonb,
  hit boolean,
  shadow_only boolean not null default true check(shadow_only),
  live_execution boolean not null default false check(not live_execution),
  check(due_at>signal_at and due_at<=signal_at+interval '90 minutes'),
  check((target_role='NO_FORWARD_LEVEL' and target_price is null) or (target_role<>'NO_FORWARD_LEVEL' and target_price>0)),
  check(invalidation_price is null or invalidation_price>0),
  check(target_price is null or invalidation_price is null or
    (direction='UP' and invalidation_price<entry_price and entry_price<target_price) or
    (direction='DOWN' and target_price<entry_price and entry_price<invalidation_price))
);
create index if not exists brian_dip_v84_decisions_pending_idx on public.brian_dip_v84_decisions(release_id,decision_at) where resolved_at is null and target_price is not null and invalidation_price is not null;
create index if not exists brian_dip_v84_decisions_episode_idx on public.brian_dip_v84_decisions(release_id,calibration_family_id,episode_id,decision_at);
alter table public.brian_dip_v84_decisions enable row level security;
revoke all on public.brian_dip_v84_decisions from public,anon,authenticated;
grant select,insert,update on public.brian_dip_v84_decisions to service_role;

create or replace function public.brian_dip_v84_immutable_decision() returns trigger
language plpgsql set search_path='pg_catalog','public' as $$
begin
  if tg_op='DELETE' then raise exception 'V84_DECISION_EVIDENCE_IMMUTABLE'; end if;
  if (to_jsonb(new)-array['checked_until','last_price','resolved_at','resolution','hit']) is distinct from (to_jsonb(old)-array['checked_until','last_price','resolved_at','resolution','hit']) then raise exception 'V84_DECISION_INPUT_IMMUTABLE'; end if;
  if old.resolved_at is not null and to_jsonb(new) is distinct from to_jsonb(old) then raise exception 'V84_RESOLUTION_IMMUTABLE'; end if;
  if new.checked_until<old.checked_until or new.checked_until>least(new.due_at,clock_timestamp()) or new.resolved_at>clock_timestamp() then raise exception 'V84_INVALID_RESOLUTION_TIME'; end if;
  return new;
end $$;
revoke all on function public.brian_dip_v84_immutable_decision() from public,anon,authenticated;
grant execute on function public.brian_dip_v84_immutable_decision() to service_role;
drop trigger if exists brian_dip_v84_decision_immutable on public.brian_dip_v84_decisions;
create trigger brian_dip_v84_decision_immutable before update or delete on public.brian_dip_v84_decisions for each row execute function public.brian_dip_v84_immutable_decision();

create table if not exists public.brian_dip_v84_ledger (
  transition_id text primary key,
  session_id text not null,
  occurrence_id text,
  episode_id text,
  recorded_at timestamptz not null default clock_timestamp(),
  event_kind text not null check(event_kind in ('SESSION_START','BUY','SELL','SHORT_OPEN','SHORT_CLOSE')),
  release_id text not null,
  calibration_family_id text not null,
  payload jsonb not null,
  account_after jsonb not null,
  shadow_only boolean not null default true check(shadow_only),
  live_execution boolean not null default false check(not live_execution),
  unique(session_id,occurrence_id,event_kind)
);
create index if not exists brian_dip_v84_ledger_session_time_idx on public.brian_dip_v84_ledger(session_id,recorded_at desc);
create index if not exists brian_dip_v84_ledger_episode_entry_idx on public.brian_dip_v84_ledger(episode_id) where event_kind in ('BUY','SHORT_OPEN');
alter table public.brian_dip_v84_ledger enable row level security;
revoke all on public.brian_dip_v84_ledger from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_ledger to service_role;
drop trigger if exists brian_dip_v84_ledger_append_only on public.brian_dip_v84_ledger;
create trigger brian_dip_v84_ledger_append_only before update or delete on public.brian_dip_v84_ledger for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_execution_outcomes (
  outcome_id text primary key,
  session_id text not null,
  occurrence_id text not null unique,
  episode_id text not null,
  setup text not null,
  direction text not null check(direction in ('UP','DOWN')),
  regime text not null,
  release_id text not null,
  calibration_family_id text not null,
  closed_at timestamptz not null,
  resolution_reason text not null,
  is_win boolean not null,
  ambiguous_conservative_loss boolean not null default false,
  realized_pnl numeric not null,
  shadow_only boolean not null default true check(shadow_only),
  live_execution boolean not null default false check(not live_execution)
);
create index if not exists brian_dip_v84_exec_cal_idx on public.brian_dip_v84_execution_outcomes(release_id,calibration_family_id,setup,direction,regime,closed_at desc);
alter table public.brian_dip_v84_execution_outcomes enable row level security;
revoke all on public.brian_dip_v84_execution_outcomes from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_execution_outcomes to service_role;
drop trigger if exists brian_dip_v84_exec_outcome_append_only on public.brian_dip_v84_execution_outcomes;
create trigger brian_dip_v84_exec_outcome_append_only before update or delete on public.brian_dip_v84_execution_outcomes for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_candidate_evaluations (
  evaluation_id text primary key,
  session_id text not null,
  occurrence_id text,
  episode_id text,
  evaluated_at timestamptz not null,
  release_id text not null,
  strategy_manifest_hash text not null,
  candidate_set jsonb not null,
  first_blocking_veto text,
  veto_stage text,
  spread_bps double precision,
  atr_1m double precision,
  atr_5m double precision,
  atr_15m double precision,
  eval_offset_ms_from_seal bigint,
  payload jsonb not null default '{}'::jsonb
);
alter table public.brian_dip_v84_candidate_evaluations enable row level security;
revoke all on public.brian_dip_v84_candidate_evaluations from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_candidate_evaluations to service_role;
drop trigger if exists brian_dip_v84_candidate_eval_append_only on public.brian_dip_v84_candidate_evaluations;
create trigger brian_dip_v84_candidate_eval_append_only before update or delete on public.brian_dip_v84_candidate_evaluations for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_level_events (
  level_event_id text primary key,
  session_id text not null,
  occurrence_id text not null,
  episode_id text not null,
  release_id text not null,
  observed_at timestamptz not null,
  level_id text,
  event_kind text not null,
  payload jsonb not null
);
alter table public.brian_dip_v84_level_events enable row level security;
revoke all on public.brian_dip_v84_level_events from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_level_events to service_role;
drop trigger if exists brian_dip_v84_level_event_append_only on public.brian_dip_v84_level_events;
create trigger brian_dip_v84_level_event_append_only before update or delete on public.brian_dip_v84_level_events for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_counterfactual_outcomes (
  counterfactual_id text primary key,
  session_id text not null,
  occurrence_id text not null,
  episode_id text not null,
  release_id text not null,
  horizon_minutes integer not null check(horizon_minutes in (90,120,180,240)),
  recorded_at timestamptz not null default clock_timestamp(),
  payload jsonb not null
);
alter table public.brian_dip_v84_counterfactual_outcomes enable row level security;
revoke all on public.brian_dip_v84_counterfactual_outcomes from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_counterfactual_outcomes to service_role;
drop trigger if exists brian_dip_v84_counterfactual_append_only on public.brian_dip_v84_counterfactual_outcomes;
create trigger brian_dip_v84_counterfactual_append_only before update or delete on public.brian_dip_v84_counterfactual_outcomes for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_evaluation_protocols (
  evaluation_protocol_id text primary key,
  release_id text not null,
  strategy_manifest_hash text not null,
  starts_at timestamptz not null,
  ends_at timestamptz not null,
  primary_metrics jsonb not null,
  secondary_metrics jsonb not null default '[]'::jsonb,
  minimum_requirements jsonb not null,
  allowed_exclusions jsonb not null default '[]'::jsonb,
  failure_criteria jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default clock_timestamp(),
  check(ends_at>starts_at)
);
alter table public.brian_dip_v84_evaluation_protocols enable row level security;
revoke all on public.brian_dip_v84_evaluation_protocols from public,anon,authenticated;
grant select,insert on public.brian_dip_v84_evaluation_protocols to service_role;
drop trigger if exists brian_dip_v84_protocol_append_only on public.brian_dip_v84_evaluation_protocols;
create trigger brian_dip_v84_protocol_append_only before update or delete on public.brian_dip_v84_evaluation_protocols for each row execute function public.brian_dip_v84_reject_mutation();

create table if not exists public.brian_dip_v84_worker_leases (
  lease_key text primary key,
  owner_id text,
  lease_generation bigint not null default 0 check(lease_generation>=0),
  acquired_at timestamptz,
  heartbeat_at timestamptz,
  expires_at timestamptz,
  release_id text
);
alter table public.brian_dip_v84_worker_leases enable row level security;
revoke all on public.brian_dip_v84_worker_leases from public,anon,authenticated;
grant select,insert,update on public.brian_dip_v84_worker_leases to service_role;

create or replace function public.brian_dip_v84_acquire_lease(p_lease_key text,p_owner_id text,p_lease_seconds integer)
returns table(acquired boolean,lease_generation bigint)
language plpgsql set search_path='pg_catalog','public' as $$
declare r public.brian_dip_v84_worker_leases%rowtype; now_at timestamptz:=clock_timestamp();
begin
  if coalesce(length(p_lease_key),0)=0 or coalesce(length(p_owner_id),0)=0 or p_lease_seconds not between 5 and 120 then raise exception 'V84_INVALID_LEASE_REQUEST'; end if;
  insert into public.brian_dip_v84_worker_leases(lease_key) values(p_lease_key) on conflict do nothing;
  select * into r from public.brian_dip_v84_worker_leases where lease_key=p_lease_key for update;
  if r.owner_id is null or r.expires_at is null or r.expires_at<=now_at then
    update public.brian_dip_v84_worker_leases set owner_id=p_owner_id,lease_generation=r.lease_generation+1,acquired_at=now_at,heartbeat_at=now_at,expires_at=now_at+make_interval(secs=>p_lease_seconds),release_id='dip-v84-package1-evidence-20260910.1' where lease_key=p_lease_key returning * into r;
    return query select true,r.lease_generation;return;
  end if;
  return query select false,r.lease_generation;
end $$;
revoke all on function public.brian_dip_v84_acquire_lease(text,text,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_acquire_lease(text,text,integer) to service_role;

create or replace function public.brian_dip_v84_renew_lease(p_lease_key text,p_owner_id text,p_lease_generation bigint,p_lease_seconds integer)
returns boolean language plpgsql set search_path='pg_catalog','public' as $$
declare n integer;
begin
  if p_lease_seconds not between 5 and 120 then return false; end if;
  update public.brian_dip_v84_worker_leases set heartbeat_at=clock_timestamp(),expires_at=clock_timestamp()+make_interval(secs=>p_lease_seconds)
  where lease_key=p_lease_key and owner_id=p_owner_id and lease_generation=p_lease_generation and expires_at>clock_timestamp() and release_id='dip-v84-package1-evidence-20260910.1';
  get diagnostics n=row_count;return n=1;
end $$;
revoke all on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) to service_role;

create or replace function public.brian_dip_v84_release_lease(p_lease_key text,p_owner_id text,p_lease_generation bigint)
returns boolean language plpgsql set search_path='pg_catalog','public' as $$
declare n integer;
begin
  update public.brian_dip_v84_worker_leases set owner_id=null,heartbeat_at=clock_timestamp(),expires_at=clock_timestamp()
  where lease_key=p_lease_key and owner_id=p_owner_id and lease_generation=p_lease_generation;
  get diagnostics n=row_count;return n=1;
end $$;
revoke all on function public.brian_dip_v84_release_lease(text,text,bigint) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_release_lease(text,text,bigint) to service_role;

create or replace function public.brian_dip_v84_initialize_session() returns trigger
language plpgsql set search_path='pg_catalog','public' as $$
declare r jsonb; rel public.brian_dip_v84_releases%rowtype;
begin
  if new.event_kind<>'START' then return new; end if;
  select * into rel from public.brian_dip_v84_releases where release_id=new.config->>'release_id';
  if not found or rel.status<>'SEALED' then raise exception 'V84_RELEASE_NOT_SEALED'; end if;
  if new.config->>'release_id' is distinct from rel.release_id
     or new.config->>'logic_hash' is distinct from rel.logic_hash
     or new.config->>'strategy_manifest_hash' is distinct from rel.strategy_manifest_hash
     or new.config->>'calibration_family_id' is distinct from rel.calibration_family_id
     or new.config->>'db_contract_version' is distinct from rel.db_contract_version
     or new.config->>'policy_version' is distinct from 'dip-v84-l1-evidence-20260910.1'
     or new.config->>'engine_version' is distinct from 'brian-dip-v84'
     or new.config->>'decision_revision' is distinct from 'dip-v84-l1-occurrence-20260910.1'
     or new.config->>'metric_version' is distinct from 'target-before-invalidation-v84.1'
     or new.config->'symbols' is distinct from '["ETHUSDT"]'::jsonb
     or new.config->>'shadow_only' is distinct from 'true'
     or new.config->>'live_execution' is distinct from 'false'
     or new.config->>'browser_execution' is distinct from 'false'
     or new.config->>'server_authoritative' is distinct from 'true'
     or new.config->>'allow_shadow_short' is distinct from 'true'
     or coalesce((new.config->>'max_shadow_leverage')::integer,0)<>1
     or coalesce(new.config->>'execution_mode','') not in ('OBSERVE','SHADOW_PAPER')
  then raise exception 'V84_INVALID_SESSION_CONTRACT'; end if;
  if exists(select 1 from public.brian_dip_v84_runtime where session_id=new.session_id) then return new; end if;
  r=jsonb_build_object('start',new.starting_equity,'cash',new.starting_equity,'realized',0,'trades',0,'wins',0,'losses',0,'pos',null,'latestThesis',null,'lastOccurrence',null,'lastSnapshotHour',null,'marketCursor',0,'lastClosedAt',null);
  insert into public.brian_dip_v84_runtime(session_id,runtime) values(new.session_id,r);
  insert into public.brian_dip_v84_ledger(transition_id,session_id,event_kind,release_id,calibration_family_id,payload,account_after)
  values('v84-start-'||new.session_id,new.session_id,'SESSION_START',rel.release_id,rel.calibration_family_id,jsonb_build_object('starting_equity',new.starting_equity),r);
  return new;
end $$;
revoke all on function public.brian_dip_v84_initialize_session() from public,anon,authenticated;
grant execute on function public.brian_dip_v84_initialize_session() to service_role;
drop trigger if exists brian_dip_v84_session_init on public.brian_dip_v84_session_events;
create trigger brian_dip_v84_session_init before insert on public.brian_dip_v84_session_events for each row execute function public.brian_dip_v84_initialize_session();

create or replace function public.brian_dip_v84_execution_calibration(p_release_id text,p_calibration_family_id text,p_setup text,p_direction text,p_regime text)
returns table(state text,samples integer,episodes integer,days integer,wins integer,losses integer,p double precision,lower95 double precision,upper95 double precision,ambiguous_losses integer,unavailable_reason text)
language plpgsql stable set search_path='pg_catalog','public' as $$
declare rel public.brian_dip_v84_releases%rowtype; n integer; h integer; ep integer; dy integer; amb integer; pp double precision; z double precision:=1.96; d double precision; centre double precision; width double precision;
begin
  select * into rel from public.brian_dip_v84_releases where release_id=p_release_id;
  if not found or rel.status<>'SEALED' or rel.calibration_family_id is distinct from p_calibration_family_id then
    return query select 'UNAVAILABLE',0,0,0,0,0,null::double precision,null::double precision,null::double precision,0,'RELEASE_OR_FAMILY_MISMATCH';return;
  end if;
  with firsts as (
    select distinct on(episode_id) episode_id,is_win,ambiguous_conservative_loss,closed_at
    from public.brian_dip_v84_execution_outcomes
    where release_id=p_release_id and calibration_family_id=p_calibration_family_id and setup=p_setup and direction=p_direction and regime=p_regime
    order by episode_id,closed_at,occurrence_id
  ) select count(*)::integer,count(*) filter(where is_win)::integer,count(distinct episode_id)::integer,count(distinct closed_at::date)::integer,count(*) filter(where ambiguous_conservative_loss)::integer into n,h,ep,dy,amb from firsts;
  if n<40 then return query select 'COLD_NEW_FAMILY',n,ep,dy,h,n-h,null::double precision,null::double precision,null::double precision,amb,null::text;return; end if;
  pp=h::double precision/n;d=1+z*z/n;centre=(pp+z*z/(2*n))/d;width=z*sqrt((pp*(1-pp)+z*z/(4*n))/n)/d;
  return query select 'WARM',n,ep,dy,h,n-h,pp,greatest(0.0,centre-width),least(1.0,centre+width),amb,null::text;
end $$;
revoke all on function public.brian_dip_v84_execution_calibration(text,text,text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_execution_calibration(text,text,text,text,text) to service_role;

create or replace function public.brian_dip_v84_forecast_calibration(p_release_id text,p_calibration_family_id text,p_setup text,p_direction text,p_regime text)
returns table(samples integer,hits integer,ambiguous integer,p double precision)
language sql stable set search_path='pg_catalog','public' as $$
  with firsts as (
    select distinct on(episode_id) episode_id,hit,resolution
    from public.brian_dip_v84_decisions
    where release_id=p_release_id and calibration_family_id=p_calibration_family_id and setup=p_setup and direction=p_direction and regime=p_regime and resolved_at is not null
    order by episode_id,decision_at,occurrence_id
  )
  select count(*) filter(where hit is not null)::integer,count(*) filter(where hit is true)::integer,count(*) filter(where hit is null)::integer,
    case when count(*) filter(where hit is not null)>0 then count(*) filter(where hit is true)::double precision/count(*) filter(where hit is not null) else null end
  from firsts
$$;
revoke all on function public.brian_dip_v84_forecast_calibration(text,text,text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_forecast_calibration(text,text,text,text,text) to service_role;

create or replace function public.brian_dip_v84_commit(
  p_session_id text,p_expected_version bigint,p_owner_id text,p_lease_generation bigint,p_commit_id text,
  p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb
) returns jsonb
language plpgsql set search_path='pg_catalog','public' as $$
declare
  old_state public.brian_dip_v84_runtime%rowtype; latest public.brian_dip_v84_session_events%rowtype; start_row public.brian_dip_v84_session_events%rowtype; lease_row public.brian_dip_v84_worker_leases%rowtype;
  payload_hash text; e jsonb; d public.brian_dip_v84_decisions%rowtype; pos jsonb; old_pos jsonb; kind text; side text; margin numeric; fee_close numeric; gross numeric; funding numeric; expected_net numeric; delta_cash numeric:=0; delta_realized numeric:=0; opens integer:=0; closes integer:=0; wins integer:=0; losses integer:=0; ambiguous_loss boolean;
begin
  if p_commit_id is null or length(p_commit_id)>160 or jsonb_typeof(p_events) is distinct from 'array' or jsonb_array_length(p_events)>1 or octet_length(p_runtime::text)>120000 or octet_length(p_snapshot::text)>180000 or octet_length(coalesce(p_decision,'null'::jsonb)::text)>30000 then raise exception 'V84_INVALID_COMMIT'; end if;
  perform pg_advisory_xact_lock(hashtextextended('brian-dip-v84-control',0));
  select * into old_state from public.brian_dip_v84_runtime where session_id=p_session_id for update;if not found then raise exception 'V84_STATE_MISSING_RECONCILE';end if;
  payload_hash=md5(jsonb_build_array(p_runtime,p_snapshot,p_decision,p_events)::text);
  if old_state.last_commit_id=p_commit_id then if old_state.last_commit_hash<>payload_hash then raise exception 'V84_IDEMPOTENCY_PAYLOAD_CONFLICT';end if;return jsonb_build_object('status','ALREADY_COMMITTED','state_version',old_state.state_version);end if;
  if old_state.state_version<>p_expected_version then raise exception 'V84_STATE_VERSION_CONFLICT';end if;
  select * into latest from public.brian_dip_v84_session_events order by requested_at desc,event_id desc limit 1;if not found or latest.session_id is distinct from p_session_id then raise exception 'V84_STALE_SESSION';end if;
  if latest.event_kind='PAUSE' then select * into start_row from public.brian_dip_v84_session_events where session_id=p_session_id and event_kind='START' order by requested_at,event_id limit 1;if not found then raise exception 'V84_START_MISSING';end if;latest.config:=start_row.config;latest.starting_equity:=start_row.starting_equity;latest.trade_notional:=start_row.trade_notional;end if;
  if latest.config->>'release_id' is distinct from 'dip-v84-package1-evidence-20260910.1' or latest.config->>'strategy_manifest_hash' is distinct from '1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0' or latest.config->>'calibration_family_id' is distinct from 'dip-v84-package1-evidence-family-1' or latest.config->>'shadow_only' is distinct from 'true' or latest.config->>'live_execution' is distinct from 'false' or latest.config->>'browser_execution' is distinct from 'false' or latest.config->>'server_authoritative' is distinct from 'true' or coalesce((latest.config->>'max_shadow_leverage')::integer,0)<>1 then raise exception 'V84_SESSION_CONTRACT_MISMATCH';end if;
  select * into lease_row from public.brian_dip_v84_worker_leases where lease_key='brian-dip-v84-worker' for share;
  if lease_row.owner_id is distinct from p_owner_id or lease_row.lease_generation is distinct from p_lease_generation or lease_row.expires_at is null or lease_row.expires_at<=clock_timestamp() or lease_row.release_id is distinct from 'dip-v84-package1-evidence-20260910.1' then raise exception 'V84_LEASE_FENCE_LOST';end if;
  if p_snapshot->>'session_id' is distinct from p_session_id or p_snapshot#>>'{state,serverRuntime,release_id}' is distinct from 'dip-v84-package1-evidence-20260910.1' or p_snapshot#>>'{state,serverRuntime,shadow_only}' is distinct from 'true' or p_snapshot#>>'{state,serverRuntime,live_execution}' is distinct from 'false' then raise exception 'V84_SNAPSHOT_CONTRACT';end if;
  if jsonb_typeof(p_runtime) is distinct from 'object' or not(p_runtime?'pos') or (p_runtime->>'start')::numeric is distinct from (old_state.runtime->>'start')::numeric or (p_runtime->>'marketCursor')::bigint<(old_state.runtime->>'marketCursor')::bigint then raise exception 'V84_RUNTIME_CONTRACT';end if;
  pos=p_runtime->'pos';old_pos=old_state.runtime->'pos';
  for e in select value from jsonb_array_elements(p_events) loop
    kind=e->>'event_kind';if kind not in ('BUY','SELL','SHORT_OPEN','SHORT_CLOSE') or e#>>'{metadata,server_v84}' is distinct from 'true' or e#>>'{metadata,release_id}' is distinct from 'dip-v84-package1-evidence-20260910.1' then raise exception 'V84_INVALID_EVENT';end if;
    if kind in ('BUY','SHORT_OPEN') then
      opens=opens+1;if latest.event_kind<>'START' or latest.config->>'execution_mode' is distinct from 'SHADOW_PAPER' then raise exception 'V84_ENTRY_DISABLED';end if;if jsonb_typeof(pos) is distinct from 'object' or (pos->>'leverage')::integer<>1 or pos->>'release_id' is distinct from 'dip-v84-package1-evidence-20260910.1' then raise exception 'V84_POSITION_CONTRACT';end if;margin=(pos->>'margin')::numeric;delta_cash=delta_cash-margin-(e->>'fees')::numeric;
    else
      closes=closes+1;if old_pos='null'::jsonb then raise exception 'V84_CLOSE_WITHOUT_POSITION';end if;side=old_pos->>'side';fee_close=(e->>'fees')::numeric-(old_pos->>'fees_open')::numeric;if fee_close<0 then raise exception 'V84_NEGATIVE_CLOSE_FEE';end if;gross=case when side='LONG' then ((e->>'exit_price')::numeric-(old_pos->>'entry')::numeric)*(old_pos->>'qty')::numeric else ((old_pos->>'entry')::numeric-(e->>'exit_price')::numeric)*(old_pos->>'qty')::numeric end;funding=coalesce((e->>'funding_cashflow')::numeric,0);expected_net=gross-(e->>'fees')::numeric+funding;if abs((e->>'realized_pnl')::numeric-expected_net)>0.0000001 then raise exception 'V84_EXIT_PNL_MISMATCH';end if;delta_cash=delta_cash+(old_pos->>'margin')::numeric+gross-fee_close+funding;delta_realized=delta_realized+expected_net;if expected_net>0 then wins=wins+1;elsif expected_net<0 then losses=losses+1;end if;
    end if;
  end loop;
  if opens>1 or closes>1 or (old_pos='null'::jsonb and pos<>'null'::jsonb and opens<>1) or (old_pos<>'null'::jsonb and pos='null'::jsonb and closes<>1) or (opens=1 and old_pos<>'null'::jsonb) or (closes=1 and old_pos='null'::jsonb) then raise exception 'V84_ILLEGAL_POSITION_TRANSITION';end if;
  if old_pos<>'null'::jsonb and pos<>'null'::jsonb and (old_pos-array['checked_until','market_price','funding_accrued','funding_settlements']) is distinct from (pos-array['checked_until','market_price','funding_accrued','funding_settlements']) then raise exception 'V84_OPEN_PLAN_IMMUTABLE';end if;
  if abs((p_runtime->>'cash')::numeric-((old_state.runtime->>'cash')::numeric+delta_cash))>0.0000001 or abs((p_runtime->>'realized')::numeric-((old_state.runtime->>'realized')::numeric+delta_realized))>0.0000001 or (p_runtime->>'trades')::integer<>(old_state.runtime->>'trades')::integer+closes or (p_runtime->>'wins')::integer<>(old_state.runtime->>'wins')::integer+wins or (p_runtime->>'losses')::integer<>(old_state.runtime->>'losses')::integer+losses or (p_runtime->>'cash')::numeric<0 then raise exception 'V84_ACCOUNTING_MISMATCH';end if;
  if p_decision is not null and p_decision<>'null'::jsonb then
    d=jsonb_populate_record(null::public.brian_dip_v84_decisions,p_decision);if d.session_id<>p_session_id or d.release_id<>'dip-v84-package1-evidence-20260910.1' or d.strategy_manifest_hash<>'1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0' or d.calibration_family_id<>'dip-v84-package1-evidence-family-1' or d.shadow_only is distinct from true or d.live_execution is distinct from false then raise exception 'V84_INVALID_DECISION';end if;
    insert into public.brian_dip_v84_decisions select d.*;
  end if;
  for e in select value from jsonb_array_elements(p_events) loop
    insert into public.brian_dip_v84_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,release_id,calibration_family_id,payload,account_after)
    values(e->>'transition_id',p_session_id,e->>'occurrence_id',e->>'episode_id',e->>'event_kind','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1',e,p_runtime);
    if e->>'event_kind' in ('SELL','SHORT_CLOSE') then
      ambiguous_loss=coalesce((e#>>'{metadata,execution_ambiguous_loss}')::boolean,false);
      insert into public.brian_dip_v84_execution_outcomes(outcome_id,session_id,occurrence_id,episode_id,setup,direction,regime,release_id,calibration_family_id,closed_at,resolution_reason,is_win,ambiguous_conservative_loss,realized_pnl)
      values('outcome-'||(e->>'transition_id'),p_session_id,e->>'occurrence_id',e->>'episode_id',old_pos->>'setup',case when old_pos->>'side'='LONG' then 'UP' else 'DOWN' end,old_pos->>'regime','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1',clock_timestamp(),coalesce(e#>>'{metadata,exit_reason}','UNKNOWN'),case when ambiguous_loss then false else (e->>'realized_pnl')::numeric>0 end,ambiguous_loss,(e->>'realized_pnl')::numeric);
    end if;
  end loop;
  update public.brian_dip_v84_runtime set runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp() where session_id=p_session_id;
  return jsonb_build_object('status','COMMITTED','state_version',old_state.state_version+1);
end $$;
revoke all on function public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) to service_role;

-- No cron job is created here. No V8.4 release is sealed here.
-- Live activation requires an explicit later migration after GitHub/CI review.
