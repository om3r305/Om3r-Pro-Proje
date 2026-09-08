-- DIP V8.1 only. Preserve the recovered global retention policy and all learning evidence.
-- Current runtime is mutable; compact decisions and accounting transitions are durable evidence.
create table public.brian_dip_v8_runtime (
  session_id text primary key,
  state_version bigint not null default 0 check (state_version >= 0),
  runtime jsonb not null,
  snapshot jsonb,
  last_commit_id text,
  last_commit_hash text,
  updated_at timestamptz not null default now(),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);
alter table public.brian_dip_v8_runtime enable row level security;
revoke all on public.brian_dip_v8_runtime from public, anon, authenticated;
grant select, insert, update on public.brian_dip_v8_runtime to service_role;
alter table public.brian_dip_v8_runtime set (autovacuum_vacuum_scale_factor=0.02, autovacuum_vacuum_threshold=20, toast.autovacuum_vacuum_scale_factor=0.02, toast.autovacuum_vacuum_threshold=20);

create table public.brian_dip_v8_ledger (
  transition_id text primary key,
  session_id text not null,
  occurrence_id text,
  episode_id text,
  recorded_at timestamptz not null default clock_timestamp(),
  event_kind text not null check (event_kind in ('SESSION_START','BUY','SELL')),
  payload jsonb not null,
  account_after jsonb not null,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  unique (session_id, occurrence_id, event_kind)
);
create index brian_dip_v8_ledger_session_time on public.brian_dip_v8_ledger(session_id, recorded_at desc);
alter table public.brian_dip_v8_ledger enable row level security;
revoke all on public.brian_dip_v8_ledger from public, anon, authenticated;
grant select, insert on public.brian_dip_v8_ledger to service_role;
create trigger brian_dip_v8_ledger_append_only before update or delete on public.brian_dip_v8_ledger for each row execute function public.brian_reject_mutation();

create table public.brian_dip_v8_decisions (
  occurrence_id text primary key,
  session_id text not null,
  episode_id text not null,
  symbol text not null check (symbol='ETHUSDT'),
  decision_at timestamptz not null,
  due_at timestamptz not null,
  setup text not null,
  direction text not null check (direction in ('UP','DOWN')),
  regime text not null,
  venue text not null check (venue='SPOT'),
  entry_price numeric not null check (entry_price>0),
  target_price numeric not null check (target_price>0),
  invalidation_price numeric not null check (invalidation_price>0),
  raw_conviction double precision,
  calibrated_probability double precision,
  calibration_samples integer not null,
  policy_version text not null,
  metric_version text not null,
  evidence jsonb not null,
  checked_until timestamptz,
  last_price numeric,
  resolved_at timestamptz,
  resolution jsonb,
  hit boolean,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  check (due_at>decision_at and due_at<=decision_at+interval '90 minutes'),
  check ((direction='UP' and invalidation_price<entry_price and entry_price<target_price) or (direction='DOWN' and target_price<entry_price and entry_price<invalidation_price))
);
create index brian_dip_v8_decisions_pending on public.brian_dip_v8_decisions(decision_at) where resolved_at is null;
create index brian_dip_v8_decisions_calibration on public.brian_dip_v8_decisions(policy_version,setup,direction,regime,resolved_at desc) where resolved_at is not null;
create index brian_dip_v8_decisions_episode on public.brian_dip_v8_decisions(policy_version,episode_id,decision_at);
alter table public.brian_dip_v8_decisions enable row level security;
revoke all on public.brian_dip_v8_decisions from public, anon, authenticated;
grant select, insert, update on public.brian_dip_v8_decisions to service_role;

create function public.brian_dip_v8_immutable_decision() returns trigger language plpgsql security invoker set search_path=pg_catalog,public as $$
begin
  if tg_op='DELETE' then raise exception 'V8_DECISION_EVIDENCE_IMMUTABLE'; end if;
  if (to_jsonb(new)-array['checked_until','last_price','resolved_at','resolution','hit']) is distinct from (to_jsonb(old)-array['checked_until','last_price','resolved_at','resolution','hit']) then raise exception 'V8_DECISION_INPUT_IMMUTABLE'; end if;
  if old.resolved_at is not null and to_jsonb(new) is distinct from to_jsonb(old) then raise exception 'V8_RESOLUTION_IMMUTABLE'; end if;
  if new.checked_until<old.checked_until or new.checked_until>least(new.due_at,clock_timestamp()) or new.resolved_at>clock_timestamp() then raise exception 'V8_INVALID_RESOLUTION_TIME'; end if;
  return new;
end $$;
create trigger brian_dip_v8_decision_immutable before update or delete on public.brian_dip_v8_decisions for each row execute function public.brian_dip_v8_immutable_decision();
revoke all on function public.brian_dip_v8_immutable_decision() from public,anon,authenticated;
grant execute on function public.brian_dip_v8_immutable_decision() to service_role;

create function public.brian_dip_v8_initialize_session() returns trigger language plpgsql security invoker set search_path=pg_catalog,public as $$
declare r jsonb;
begin
  -- No restart may orphan an open V8 position, including a legacy UI restart request.
  if new.event_kind='START' and exists(select 1 from public.brian_dip_v8_runtime where session_id<>new.session_id and runtime->'pos'<>'null'::jsonb) then raise exception 'OPEN_V8_POSITION_BLOCKS_RESTART'; end if;
  if new.event_kind<>'START' or new.config->>'policy_version'<>'dip-v8-integrity-20260908.1' or new.config->>'policy_version' is null then return new; end if;
  if new.config->>'engine_version' is distinct from 'brian-dip-chart-reader-v8' or new.config->'symbols' is distinct from '["ETHUSDT"]'::jsonb or new.config->>'shadow_only' is distinct from 'true' or new.config->>'live_execution' is distinct from 'false' or new.config->>'browser_execution' is distinct from 'false' or new.config->>'server_authoritative' is distinct from 'true' or new.config->>'allow_shadow_short' is distinct from 'false' or coalesce(new.config->>'execution_mode','') not in ('OBSERVE','SHADOW_PAPER') then raise exception 'INVALID_V8_SHADOW_SESSION'; end if;
  if exists(select 1 from public.brian_dip_v8_runtime where session_id=new.session_id) then return new; end if;
  if exists(select 1 from public.brian_dip_session_events where session_id=new.session_id) then raise exception 'V8_STATE_MISSING_RECONCILE'; end if;
  r=jsonb_build_object('start',new.starting_equity,'cash',new.starting_equity,'realized',0,'trades',0,'wins',0,'losses',0,'pos',null,'lastLock',null,'latestThesis',null,'lastOccurrence',null,'lastSnapshotHour',null,'marketCursor',0,'lastClosedAt',null);
  insert into public.brian_dip_v8_runtime(session_id,runtime) values(new.session_id,r);
  insert into public.brian_dip_v8_ledger(transition_id,session_id,event_kind,payload,account_after) values('v8-start-'||new.session_id,new.session_id,'SESSION_START',jsonb_build_object('starting_equity',new.starting_equity,'policy_version',new.config->>'policy_version'),r);
  return new;
end $$;
revoke all on function public.brian_dip_v8_initialize_session() from public,anon,authenticated;
grant execute on function public.brian_dip_v8_initialize_session() to service_role;
create trigger brian_dip_v8_session_init before insert on public.brian_dip_session_events for each row execute function public.brian_dip_v8_initialize_session();

create function public.brian_dip_v8_commit(p_session_id text,p_expected_version bigint,p_owner_token text,p_commit_id text,p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb)
returns jsonb language plpgsql security invoker set search_path=pg_catalog,public as $$
declare
  old_state public.brian_dip_v8_runtime%rowtype;
  latest public.brian_dip_session_events%rowtype;
  lease_row public.brian_collector_leases%rowtype;
  payload_hash text;
  e jsonb;
  d public.brian_dip_v8_decisions%rowtype;
  buys integer=0; sells integer=0; wins integer=0; losses integer=0;
  delta_cash numeric=0; delta_realized numeric=0;
  pos jsonb; old_pos jsonb;
begin
  if p_commit_id is null or length(p_commit_id)>160 or jsonb_typeof(p_events) is distinct from 'array' or jsonb_array_length(p_events)>1 or octet_length(p_runtime::text)>100000 or octet_length(p_snapshot::text)>150000 or octet_length(coalesce(p_decision,'null'::jsonb)::text)>15000 then raise exception 'INVALID_V8_COMMIT'; end if;
  -- Same lock as START/PAUSE/RESTART: pause cannot race past the final entry fence.
  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into old_state from public.brian_dip_v8_runtime where session_id=p_session_id for update;
  if not found then raise exception 'V8_STATE_MISSING_RECONCILE'; end if;
  payload_hash=md5(jsonb_build_array(p_runtime,p_snapshot,p_decision,p_events)::text);
  if old_state.last_commit_id=p_commit_id then
    if old_state.last_commit_hash<>payload_hash then raise exception 'V8_IDEMPOTENCY_PAYLOAD_CONFLICT'; end if;
    return jsonb_build_object('status','ALREADY_COMMITTED','state_version',old_state.state_version);
  end if;
  if old_state.state_version<>p_expected_version then raise exception 'V8_STATE_VERSION_CONFLICT'; end if;
  select * into latest from public.brian_dip_session_events order by requested_at desc,event_id desc limit 1;
  if latest.session_id is distinct from p_session_id then raise exception 'STALE_V8_SESSION'; end if;
  select * into lease_row from public.brian_collector_leases where collector_id='brian-dip-shadow-worker-v8' for share;
  if lease_row.owner_token is distinct from p_owner_token or lease_row.lease_until is null or lease_row.lease_until<=clock_timestamp() then raise exception 'V8_LEASE_LOST'; end if;
  if p_snapshot->>'session_id' is distinct from p_session_id or p_snapshot#>>'{state,serverRuntime,policy_version}' is distinct from 'dip-v8-integrity-20260908.1' or p_snapshot#>>'{state,serverRuntime,shadow_only}' is distinct from 'true' or p_snapshot#>>'{state,serverRuntime,live_execution}' is distinct from 'false' then raise exception 'V8_SNAPSHOT_CONTRACT'; end if;
  if (p_runtime->>'start')::numeric is distinct from (old_state.runtime->>'start')::numeric or (p_runtime->>'marketCursor')::bigint<(old_state.runtime->>'marketCursor')::bigint then raise exception 'V8_STATE_RESET_OR_TIME_REGRESSION'; end if;
  if jsonb_typeof(p_runtime) is distinct from 'object' or not (p_runtime ? 'pos') or exists (
    select 1 from unnest(array['start','cash','realized','trades','wins','losses','marketCursor']) k
    where jsonb_typeof(p_runtime->k) is distinct from 'number'
  ) then raise exception 'V8_INVALID_RUNTIME_FIELDS'; end if;
  if p_snapshot#>'{state,v8}' is distinct from p_runtime
    or (p_snapshot->>'cash')::numeric is distinct from (p_runtime->>'cash')::numeric
    or (p_snapshot->>'realized_pnl')::numeric is distinct from (p_runtime->>'realized')::numeric
    or (p_snapshot->>'trade_count')::integer is distinct from (p_runtime->>'trades')::integer
    or (p_snapshot->>'win_count')::integer is distinct from (p_runtime->>'wins')::integer
    or (p_snapshot->>'loss_count')::integer is distinct from (p_runtime->>'losses')::integer
  then raise exception 'V8_SNAPSHOT_ACCOUNT_MISMATCH'; end if;
  pos=p_runtime->'pos'; old_pos=old_state.runtime->'pos';
  for e in select value from jsonb_array_elements(p_events) loop
    if coalesce(e->>'event_kind','') not in ('BUY','SELL') or e->>'occurrence_id' is null or e->>'transition_id' is distinct from ('v8-'||lower(e->>'event_kind')||'-'||(e->>'occurrence_id')) then raise exception 'INVALID_V8_EVENT'; end if;
    if e#>>'{metadata,server_v8}' is distinct from 'true' or exists (select 1 from unnest(array['quantity','notional','fees','realized_pnl','cash_after']) k where jsonb_typeof(e->k) is distinct from 'number') then raise exception 'V8_INVALID_EVENT_FIELDS'; end if;
    if e->>'event_kind'='BUY' then
      buys=buys+1;
      if latest.event_kind<>'START' or latest.config->>'execution_mode' is distinct from 'SHADOW_PAPER' then raise exception 'V8_ENTRY_DISABLED'; end if;
      if (p_snapshot->>'observed_at')::timestamptz<clock_timestamp()-interval '30 seconds' or (p_snapshot->>'observed_at')::timestamptz>clock_timestamp()+interval '2 seconds' then raise exception 'V8_STALE_ENTRY'; end if;
      delta_cash=delta_cash-(e->>'notional')::numeric-(e->>'fees')::numeric;
    else
      sells=sells+1; delta_realized=delta_realized+(e->>'realized_pnl')::numeric;
      delta_cash=delta_cash+(e->>'exit_price')::numeric*(e->>'quantity')::numeric-((e->>'fees')::numeric-(old_pos->>'fees_open')::numeric);
      if (e->>'realized_pnl')::numeric>0 then wins=wins+1; elsif (e->>'realized_pnl')::numeric<0 then losses=losses+1; end if;
    end if;
  end loop;
  if (old_pos='null'::jsonb and pos<>'null'::jsonb and buys<>1) or (old_pos<>'null'::jsonb and pos='null'::jsonb and sells<>1) or (buys=1 and old_pos<>'null'::jsonb) or (sells=1 and old_pos='null'::jsonb) or (sells=1 and pos<>'null'::jsonb) or (buys=1 and pos='null'::jsonb) then raise exception 'V8_ILLEGAL_POSITION_TRANSITION'; end if;
  if old_pos<>'null'::jsonb and pos<>'null'::jsonb and (old_pos-array['checked_until','market_price']) is distinct from (pos-array['checked_until','market_price']) then raise exception 'V8_OPEN_PLAN_IMMUTABLE'; end if;
  if buys=1 and (jsonb_typeof(pos) is distinct from 'object' or exists (
    select 1 from unnest(array['entry','stop','target','qty','notional','fees_open']) k
    where jsonb_typeof(pos->k) is distinct from 'number'
  ) or (pos->>'qty')::numeric<=0 or (pos->>'fees_open')::numeric<0
    or abs((pos->>'notional')::numeric-(pos->>'qty')::numeric*(pos->>'entry')::numeric)>0.0000001
  ) then raise exception 'V8_INVALID_POSITION_FIELDS'; end if;
  if buys=1 and ((pos->>'side') is distinct from 'LONG' or (pos->>'venue') is distinct from 'SPOT' or not ((pos->>'stop')::numeric<(pos->>'entry')::numeric and (pos->>'entry')::numeric<(pos->>'target')::numeric) or (pos->>'notional')::numeric>(old_state.runtime->>'cash')::numeric*0.20+0.0000001) then raise exception 'V8_POSITION_GEOMETRY_OR_CAP'; end if;
  if abs((p_runtime->>'cash')::numeric-((old_state.runtime->>'cash')::numeric+delta_cash))>0.0000001 or abs((p_runtime->>'realized')::numeric-((old_state.runtime->>'realized')::numeric+delta_realized))>0.0000001 or (p_runtime->>'trades')::integer<>(old_state.runtime->>'trades')::integer+sells or (p_runtime->>'wins')::integer<>(old_state.runtime->>'wins')::integer+wins or (p_runtime->>'losses')::integer<>(old_state.runtime->>'losses')::integer+losses then raise exception 'V8_ACCOUNTING_MISMATCH'; end if;
  if (p_runtime->>'cash')::numeric<0 then raise exception 'V8_NEGATIVE_CASH'; end if;
  if p_decision is not null and p_decision<>'null'::jsonb then
    d=jsonb_populate_record(null::public.brian_dip_v8_decisions,p_decision);
    if d.session_id<>p_session_id or d.policy_version<>'dip-v8-integrity-20260908.1' or d.metric_version<>'target-before-invalidation-v8.1' then raise exception 'INVALID_V8_DECISION'; end if;
    insert into public.brian_dip_v8_decisions(occurrence_id,session_id,episode_id,symbol,decision_at,due_at,setup,direction,regime,venue,entry_price,target_price,invalidation_price,raw_conviction,calibrated_probability,calibration_samples,policy_version,metric_version,evidence,shadow_only,live_execution)
    values(d.occurrence_id,d.session_id,d.episode_id,d.symbol,d.decision_at,d.due_at,d.setup,d.direction,d.regime,d.venue,d.entry_price,d.target_price,d.invalidation_price,d.raw_conviction,d.calibrated_probability,d.calibration_samples,d.policy_version,d.metric_version,d.evidence,true,false);
  end if;
  for e in select value from jsonb_array_elements(p_events) loop
    if e->>'event_kind'='BUY' and not exists(select 1 from public.brian_dip_v8_decisions where occurrence_id=e->>'occurrence_id' and session_id=p_session_id and entry_price=(pos->>'entry')::numeric and target_price=(pos->>'target')::numeric and invalidation_price=(pos->>'stop')::numeric) then raise exception 'V8_ENTRY_PLAN_NOT_RECORDED'; end if;
    if e->>'event_kind'='BUY' then
      select * into d from public.brian_dip_v8_decisions where occurrence_id=e->>'occurrence_id';
      if d.direction<>'UP' or (pos->>'notional')::numeric>(old_state.runtime->>'cash')::numeric*(case when d.calibration_samples<40 then 0.08 else 0.20 end)+0.0000001
        or (e->>'notional')::numeric is distinct from (pos->>'notional')::numeric
        or (e->>'quantity')::numeric is distinct from (pos->>'qty')::numeric
        or (e->>'fees')::numeric is distinct from (pos->>'fees_open')::numeric
        or e->>'occurrence_id' is distinct from pos->>'thesis_id'
      then raise exception 'V8_ENTRY_EVIDENCE_OR_CAP'; end if;
    elsif e->>'occurrence_id' is distinct from old_pos->>'thesis_id'
       or (e->>'quantity')::numeric is distinct from (old_pos->>'qty')::numeric
       or (e->>'exit_price')::numeric is null
       or abs((e->>'realized_pnl')::numeric-(((e->>'exit_price')::numeric-(old_pos->>'entry')::numeric)*(old_pos->>'qty')::numeric-(e->>'fees')::numeric))>0.0000001
    then raise exception 'V8_EXIT_ACCOUNT_MISMATCH'; end if;
    insert into public.brian_dip_v8_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,payload,account_after)
    values(e->>'transition_id',p_session_id,e->>'occurrence_id',e->>'episode_id',e->>'event_kind',e,jsonb_build_object('cash',p_runtime->'cash','realized',p_runtime->'realized','trades',p_runtime->'trades','wins',p_runtime->'wins','losses',p_runtime->'losses','pos',p_runtime->'pos'));
    -- Compatibility event view; retention may compact these rows, never the V8 ledger above.
    insert into public.brian_dip_events(event_id,session_id,observed_at,event_kind,symbol,price,entry_price,exit_price,quantity,notional,fees,realized_pnl,cash_after,equity_after,metadata,shadow_only,live_execution)
    values(e->>'transition_id',p_session_id,clock_timestamp(),e->>'event_kind','ETHUSDT',(e->>'price')::numeric,(e->>'entry_price')::numeric,(e->>'exit_price')::numeric,(e->>'quantity')::numeric,(e->>'notional')::numeric,(e->>'fees')::numeric,(e->>'realized_pnl')::numeric,(e->>'cash_after')::numeric,(e->>'equity_after')::numeric,e->'metadata',true,false);
  end loop;
  if p_runtime->>'lastSnapshotHour' is distinct from old_state.runtime->>'lastSnapshotHour' then
    insert into public.brian_dip_snapshots(snapshot_id,session_id,observed_at,cash,equity,realized_pnl,unrealized_pnl,trade_count,win_count,loss_count,state,shadow_only,live_execution)
    values(p_snapshot->>'snapshot_id',p_session_id,(p_snapshot->>'observed_at')::timestamptz,(p_snapshot->>'cash')::numeric,(p_snapshot->>'equity')::numeric,(p_snapshot->>'realized_pnl')::numeric,(p_snapshot->>'unrealized_pnl')::numeric,(p_snapshot->>'trade_count')::integer,(p_snapshot->>'win_count')::integer,(p_snapshot->>'loss_count')::integer,p_snapshot->'state',true,false);
  end if;
  if lease_row.lease_until<=clock_timestamp() then raise exception 'V8_LEASE_EXPIRED_DURING_COMMIT'; end if;
  update public.brian_dip_v8_runtime set runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp() where session_id=p_session_id;
  return jsonb_build_object('status','COMMITTED','state_version',p_expected_version+1);
end $$;
revoke all on function public.brian_dip_v8_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.brian_dip_v8_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb) to service_role;

create function public.brian_dip_v8_calibration(p_setup text,p_direction text,p_regime text)
returns table(hit boolean) language sql stable security invoker set search_path=pg_catalog,public as $$
  -- One first decision per structural episode across sessions: retries/re-entry are not extra lessons.
  select firsts.hit from (
    select distinct on (episode_id) episode_id, hit, decision_at, resolved_at
    from public.brian_dip_v8_decisions
    where setup=p_setup and direction=p_direction and regime=p_regime and venue='SPOT'
      and policy_version='dip-v8-integrity-20260908.1' and metric_version='target-before-invalidation-v8.1'
    order by episode_id,decision_at,occurrence_id
  ) firsts where resolved_at is not null order by resolved_at desc limit 500
$$;
revoke all on function public.brian_dip_v8_calibration(text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_v8_calibration(text,text,text) to service_role;

-- Preserve the post-recovery cadence, rather than silently re-enabling one-minute heavy writers.
do $$ declare j record; begin
  for j in select jobid,jobname from cron.job where jobname in ('brian-dip-shadow-worker-v7-1m','brian-dip-shadow-worker-v8-1m','brian-dip-foresight-v7-1m','brian-dip-foresight-v8-1m') loop
    perform cron.alter_job(j.jobid,schedule:=case when j.jobname like '%shadow-worker%' then '1-59/3 * * * *' else '4-59/5 * * * *' end,active:=false);
  end loop;
end $$;
comment on table public.brian_dip_v8_runtime is 'One mutable current state per session; hourly append-only compatibility snapshots. Not training evidence.';
comment on table public.brian_dip_v8_decisions is 'Compact immutable decision-time teaching evidence with separately frozen outcome; excluded from legacy raw retention.';
comment on table public.brian_dip_v8_ledger is 'Replayable compact shadow accounting evidence. No per-tick full-state append. Never substitute a missing read with starting equity.';
